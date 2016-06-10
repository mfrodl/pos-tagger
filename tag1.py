from nltk import TaggerI
from numpy import argmax, array, log
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.structure import SigmoidLayer, TanhLayer
import pickle
from collections import Counter
import sys

import model

class Tagger(TaggerI):
    def __init__(self, sents, load=False, context=3):
        self.context = context
        self.model = model.Model(sents, context)

        if load:
            self.load()
        else:
            self.nets = self.train(self.context)

        self.default_tag = self.model.t_freq.keys()[0]

    def save(self):
        #model_file = open('model.pkl', 'wb')
        #pickle.dump(self.model, model_file)
        #model_file.close()

        nets_file = open('nets.pkl', 'wb')
        pickle.dump(self.nets, nets_file)
        nets_file.close()

    def load(self):
        #model_file = open('model.pkl', 'rb')
        #self.model = pickle.load(model_file)
        #model_file.close()

        nets_file = open('nets.pkl', 'rb')
        self.nets = pickle.load(nets_file)
        nets_file.close()

    def tag(self, tokens):
        return self.neuro_tag(tokens, self.context)

    #""" BEST: 16.9. 16:19
    def neuro_tag(self, tokens, n):
        none_tokens = [None for _ in xrange(n/2)] + [tokens[0].lower()] + tokens[1:] + [None for _ in xrange((n-1)/2)]
        ngrams = [none_tokens[i:i+n] for i in xrange(len(tokens))]
        sent_tags = []
        out_probs = []
        next_probs = [0 for _ in xrange((n-1)/2*len(self.model.tagset))]
        for ng in reversed(ngrams):
            w = ng[n/2]
            tags = tuple(sorted(self.model.wt_freq[w].keys()))
            known = True
            # If a capitalized word is unknown, try to convert it to lowercase
            # first and find the corresponding tag probability distribution.
            if not tags and w[0].isupper():
                tags = tuple(sorted(self.model.wt_freq[w.lower()].keys()))

            # If the word is unkown, estimate the tag probability distribution
            # using a suffix.
            if not tags:
                known = False
                #print 'Unknown word "{0}", guessing by suffix...'.format(w)
                tags = tuple(sorted(self.model.word_suf_tag_prob(w).keys())) # TODO: optimize
                tag_probs = [self.model.word_suf_tag_prob(w)[t] for t in tags] # ...

            if len(tags) == 1:
                #print w, tags
                maxt = tags[0]
                out_probs = [1]
            else: # len(tags) >= 2, generate all pairs and choose the winning tag
                left, w, right = ng[:n/2], ng[n/2], ng[n/2+1:]
                if known:
                    tag_probs = [self.model.wt_prob[w].prob(t) for t in tags]
                counts = Counter()
                #print w, tags, tag_probs
                net_output = {}
                for i, j in all_pairs(tags):
                    t1, t2 = tags[i], tags[j]
                
                    if (t1,t2) in self.nets:
                        in_vect = [p for wl in left for p in self.prob_vect(wl)]
                        in_vect += [tag_probs[i], tag_probs[j]]
                        #in_vect += [p for wr in right for p in self.prob_vect(wr)] #list(next_probs)
                        in_vect += list(next_probs)
                        net_output[(t1,t2)] = list(self.nets[(t1,t2)].activate(in_vect))
                        tmax = (t1,t2)[argmax(net_output[(t1,t2)])]
                        #if t1 == tmax:
                        #    print '{0} > {1} ({2} > {3})'.format(t1,t2,net_output[(t1,t2)][argmax(net_output[(t1,t2)])],
                        #                                               net_output[(t1,t2)][1-argmax(net_output[(t1,t2)])])
                        #else:
                        #    print '{0} > {1} ({2} > {3})'.format(t2,t1,net_output[(t1,t2)][argmax(net_output[(t1,t2)])],
                        #                                               net_output[(t1,t2)][1-argmax(net_output[(t1,t2)])])
                        counts[tmax] += 1
                if counts:
                    maxt = counts.most_common(1)[0][0]
                    #print '{0}: {1} x {2}'.format(w, maxt, corr)
                    out_probs = [0 for _ in xrange(len(tags))]
                    out_probs[tags.index(maxt)] = 1.
                    for i in xrange(len(tags)):
                        tag = tags[i]
                        tag_pair = tuple(sorted((tag,maxt)))
                        if tag_pair in net_output:
                            min_val, max_val = min(net_output[tag_pair]), max(net_output[tag_pair])
                            out_probs[i] = min_val/max_val
                            #if out_probs[i] > 0.9:
                            #    print tokens
                            #    print tags, out_probs
                            #    raw_input()
                    #if maxt != corr:
                    #    print tokens
                    #    print tags, out_probs
                    #    raw_input()
                    
                    #print tags, out_probs
                    #print '===>', maxt
                    #raw_input()
                else:
                    #print tokens
                    #print w, tags, tag_probs
                    out_probs = []
                    maxt = self.default_tag # 77.92%
                    #maxt = tags[argmax(tag_probs)] # 77.51%

            sent_tags.append((maxt,))

            # Use the network output in the following input vector(s)
            pr_vect = [0 for _ in xrange(len(self.model.tagset))]
            for i in xrange(len(out_probs)):
                pr_vect[self.model.tag2ind[tags[i]]] = out_probs[i]
            #if net_output:
            #    pr_vect[self.model.tag2ind[tags[argmax(net_output)]]] *= 2 
            next_probs = pr_vect + next_probs[:-len(self.model.tagset)]

        sent_tags.reverse()
        tagged_sent = [(w,)+maxt for (w,maxt) in zip(tokens,sent_tags)]
        #print tagged_sent
        return tagged_sent
    #"""
  
    #""" BEST: 14.9. 17:53
    def train(self, n):
        train_sets = {}
        for ng in self.model.ngrams[n]:
            left, right = ng[:n/2], ng[n/2+1:]
            w, t0 = ng[n/2]
            tags = sorted(self.model.wt_freq[w].keys())
            if len(tags) < 2:
                continue
            #print tags
            for tag_pair in good_pairs(tags, t0):
                #print tag_pair
                tag_probs = [self.model.wt_prob[w].prob(t) for t in tag_pair]
                #if min(tag_probs) < 0.001:
                #     continue
                if tag_pair not in train_sets:
                    in_dim = (n-1) * len(self.model.tagset) + len(tag_pair)
                    train_sets[tag_pair] = ClassificationDataSet(in_dim, 1, nb_classes=len(tag_pair))

                in_vect = [p for (wl,_) in left for p in self.prob_vect(wl)]
                in_vect += tag_probs
                in_vect += [p for (wr,_) in right for p in self.prob_vect(wr)]
                klass = [tag_pair.index(t0)]
                train_sets[tag_pair].addSample(in_vect, klass)

        nets = {}
        i = 1
        for tag_pair in train_sets:
            sys.stdout.write('{0}/{1} '.format(i, len(train_sets)))
            nets[tag_pair] = self.train_net(train_sets[tag_pair], tag_pair)
            i += 1
        return nets
    #"""

    def train_net(self, train_data, tags):
        print 'Training the network... {0} ({1} items)'.format(tags, len(train_data))
        train_data._convertToOneOfMany()

        network = buildNetwork(train_data.indim, 8, train_data.outdim, bias=True, hiddenclass=TanhLayer)#SigmoidLayer)
        trainer = BackpropTrainer(network, dataset=train_data, momentum=0.1, verbose=False, weightdecay=0.01)
        trainer.trainEpochs(10)

        return network

    def my_eval(self, test_sents):
        untagged_sents = [[w for (w,_) in s] for s in test_sents]
        retagged_sents = [self.tag(s, True) for s in untagged_sents]

        # Will hold information about the number of known and uknown tokens
        counts = {False: 0., True: 0.}
        correct = {False: 0., True: 0.}

        num_sents = len(test_sents)
        for (i,ts,us) in zip(xrange(num_sents),test_sents,untagged_sents):
            print 'Evaluating sentence {0}/{1}...'.format(i+1,num_sents)
            rs = self.tag(us, mark_known=True, correct=[t for (_,t) in ts])
            for ((w,tt),(_,rt,known,tags,net_output)) in zip(ts,rs):
                counts[known] += 1
                if tt == rt:
                    correct[known] += 1
                #else:
                #    #print '{0}: is {1}, tagged as {2}'.format(w,tt,rt)
                #    #print '{0} {1}'.format(tags,net_output)

        known_accuracy = correct[True] / counts[True]
        unknown_accuracy = correct[False] / counts[False]
        overall_accuracy = (correct[True]+correct[False]) / (counts[True]+counts[False])

        return correct[True], counts[True], correct[False], counts[False], known_accuracy, unknown_accuracy, overall_accuracy

    def prob_vect(self, w):
        tag_probs = dict([(t,self.model.wt_prob[w].prob(t)) for t in self.model.wt_freq[w]])
        vect = [0 for _ in xrange(len(self.model.tagset))]
        for t in tag_probs:
            vect[self.model.tag2ind[t]] = tag_probs[t]
        return vect

    def probs2vect(self, probs):
        vect = [0 for _ in xrange(len(self.model.tagset))]
        for t in probs:
            vect[self.model.tag2ind[t]] = probs[t]
        return vect

    def unigram_tag(self, tokens):
        tagged_sent = []
        for w in tokens:
            try:
                t = self.model.wt_freq[w].keys()[0]
            except IndexError:
                t = self.default_tag
            tagged_sent.append((w,t))
        return tagged_sent

def argmax2(l):
    am1 = argmax(l)
    am2 = argmax(l[:am1] + [min(l)-1] + l[am1+1:])
    if am1 < am2:
        return am1, am2
    else:
        return am2, am1

def entropy(l):
    return -sum([x*log(x) for x in l if x >= 0.01])

def out2prob(l):
    #ll = [1.0/(1.0+exp(-2*x)) for x in l]
    ll = [exp(x) for x in l]
    sumll = sum(ll)
    return [x/sumll for x in ll]

def all_pairs(tags):
    return [(i,j) for i in xrange(len(tags)) for j in xrange(i,len(tags))]

def good_pairs(tags, t):
    return [(x,t) for x in tags[:tags.index(t)]] + [(t,x) for x in tags[tags.index(t)+1:]]
