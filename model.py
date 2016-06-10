import nltk
from nltk import FreqDist, ConditionalFreqDist, ConditionalProbDist, ELEProbDist, MLEProbDist, LaplaceProbDist
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from collections import defaultdict
import sys

class Model:
    def __init__(self, sents, context=3, max_suf_len=10, rare_max=3):
        self.tagged_sents = map(self.first2lower,sents)
        self.tagged_words = [wt for s in self.tagged_sents for wt in s]
        self.ngrams = dict([(i,[ng for ng in self.get_ngrams(i)]) for i in xrange(1,context+1)])
        self.tag_ngrams = dict([(i,self.get_tag_ngrams(i)) for i in xrange(1,context+1)])

        self.tags = [ug[0] for ug in self.tag_ngrams[1]]
        self.tokens = [ug[0][0] for ug in self.ngrams[1]]
        self.tagset = list(set(self.tags))
        self.tag2ind = dict([(self.tagset[i],i) for i in xrange(len(self.tagset))])

        self.max_suf_len = max_suf_len
        self.rare_max = rare_max

        self.t_freq = self.tag_freq()
        self.w_freq = self.word_freq()
        self.c_words = self.common_words()
        self.wt_freq = self.word_tag_freq()
        self.st_freq = self.suf_tag_freq()

        self.t_prob = self.tag_prob()
        self.wt_prob = self.word_tag_prob()
        self.st_prob = self.suf_tag_prob()
        self.sm_st_prob = self.smoothed_suf_tag_prob()

        """
        f = open('suftag.out','w')
        for suf in self.sm_st_prob:
            f.write('-{0}:\n'.format(suf))
            for tag in self.sm_st_prob[suf]:
                f.write('    {0} {1}\n'.format(tag, str(self.sm_st_prob[suf][tag])))
        f.close()
        """

    def get_ngrams(self, n):
        for s in self.tagged_sents:
            nones = [(None,None) for _ in xrange(n)]
            ss = nones + s + nones
            for ng in nltk.ngrams(ss,n):
                yield ng

    def first2lower(self, s):
        return [(s[0][0].lower(), s[0][1])] + s[1:]

    def get_tag_ngrams(self, n):
        return [tuple([t for (w,t) in ng]) for ng in self.ngrams[n]]

    def get_word_ngrams(self, n):
        return [tuple([w for (w,t) in ng]) for ng in self.ngrams[n]]

    def tag_freq(self):
        return FreqDist(self.tags)

    def word_freq(self):
        return FreqDist(self.tokens)

    def word_tag_freq(self):
        return ConditionalFreqDist(self.tagged_words)

    def two_ambig(self):
        fd = FreqDist([tuple(self.wt_freq[w].keys()) for w in self.wt_freq.keys() if len(self.wt_freq[w].keys()) == 2])
        print fd.keys()

    def tag_tag_freq(self):
        return ConditionalFreqDist(self.bigrams)

    def tag_tag_tag_freq(self):
        return ConditionalFreqDist([((t1,t2),t3) for (t1,t2,t3) in self.trigrams])

    def common_words(self):
        return [w for w in self.w_freq.samples() if self.w_freq[w] > self.rare_max]

    def suf_tag_freq(self):
        cfd = ConditionalFreqDist()
        for w in set(self.wt_freq.keys())-set(self.c_words):
            for t in self.wt_freq[w].keys():
                for suf_len in xrange(1,max(self.max_suf_len,len(w))):
                    suf = w[-suf_len:]
                    cfd[suf].inc(t, self.wt_freq[w][t])
                cfd[''].inc(t)
        return cfd

    def freq2prob(self, freq_dist):
        num_bins = max([freq_dist[w].B() for w in freq_dist] + [1])
        prob = ConditionalProbDist(freq_dist, LaplaceProbDist, num_bins)
        return prob

    def tag_prob(self):
        return MLEProbDist(self.t_freq)

    def word_tag_prob(self):
        return self.freq2prob(self.wt_freq)

    def tag_tag_prob(self):
        return self.freq2prob(self.tt_freq)

    def tag_tag_tag_prob(self):
        return self.freq2prob(self.ttt_freq)

    def suf_tag_prob(self):
        return self.freq2prob(self.st_freq)

    def smoothed_suf_tag_prob(self):
        try:
            avg_tag_prob = avg([self.t_prob.prob(t) for t in self.t_prob.samples()])
            theta = sum([(self.t_prob.prob(t) - avg_tag_prob)**2 for t in self.t_prob.samples()]) / len(self.t_prob.samples())
        except ZeroDivisionError:
            theta = 0.

        probs = {'': {}} #defaultdict(lambda: defaultdict(float))
        for suf in self.st_prob.keys():
            for tag in self.st_prob[suf].samples():
                prob = self.t_prob.prob(tag)
                probs[''][tag] = prob
                for i in range(1,self.max_suf_len+1):
                    suf_i = suf[-i:]
                    if suf_i not in probs:
                        probs[suf_i] = {}
                    probs[suf_i][tag] = (self.st_prob[suf_i].prob(tag) + theta * prob) / (1 + theta)
                    prob = probs[suf_i][tag]
        return probs

    def word_suf_tag_prob(self, w):
        suf = ''
        for i in range(self.max_suf_len,0,-1):
            if w[-i:] in self.sm_st_prob:
                suf = w[-i:]
                break
        return self.sm_st_prob[suf]

def avg(l):
    return sum(l) / len(l)
