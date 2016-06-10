import random

from sequence import TaggedSequence, feat_templates
from collections import Counter, defaultdict as dd
from math import log

class Model:
    def __init__(self, sents):
        print 'Computing model...'
        self.sents = sents

        self.generate_features()
        #print self.tag_index
        print self.num_feats, 'features'
        print 'Model computed.'

    def generate_features(self):
        self.feat = {}
        for label in feat_templates:
            self.feat[label] = set()

        self.tags = set()
        self.candidate_tags = dd(set)
        self.word_freq = Counter()
        self.tag_freq = Counter()
        for s in self.sents:
            ts = TaggedSequence(s)
            while not ts.end_of_seq():
                [tag] = ts.near_tag(0)
                self.tag_freq[tag] += 1
                self.tags.add(tag)

                for label, values in ts.all_features().items():
                    self.feat[label].update(values)

                [word] = ts.near_word(0)
                self.word_freq[word] += 1
                self.candidate_tags[word].add(tag)

                ts.shift()

        self.tags = sorted(list(self.tags))

        self.tag_index = {}
        for i in xrange(len(self.tags)):
            tag = self.tags[i]
            self.tag_index[tag] = i

        # Convert sets to lists
        self.feat = {label: list(value) for label, value in self.feat.items()}

        self.feat_index = {}
        offset = 0
        for label in sorted(self.feat):
            self.feat_index[label] = {}
            for i, value in zip(xrange(len(self.feat[label])), self.feat[label]):
                self.feat_index[label][value] = offset + i
            offset += len(self.feat[label])

        self.nums_feats = {label: len(self.feat[label]) for label in self.feat}
        self.num_feats = sum(self.nums_feats.values())

        # Generate ambiguity classes together with their abundances
        self.ambig_cls = Counter()
        for word in self.candidate_tags:
            cand_tags = tuple(sorted(self.candidate_tags[word]))
            if len(cand_tags) > 1:
                self.ambig_cls[cand_tags] += self.word_freq[word]

    def feat_by_num(self, num):
        for label in sorted(self.feat):
            if num >= self.nums_feats[label]:
                num -= self.nums_feats[label]
            else:
                print 'num', num, 'len', len(self.feat[label])
                value = self.feat[label][num]
                return label, value

    def compute_sim_matrix(self):
        sim_matrix = dd(float)
        counts = dd(float)
        """
        for t1 in self.tags:
            for t2 in self.tags:
                for left,right in set(self.tag_stats[t1]).union(set(self.tag_stats[t2])):
                    sim_matrix[t1][t2] += (self.tag_stats[t1][left,right] - self.tag_stats[t2][left,right])**2 / self.context_stats[left,right]**2
                    counts[t1][t2] += 1
        for t1 in sim_matrix:
            for t2 in sim_matrix[t1]:
                sim_matrix[t1][t2] /= counts[t1][t2]
        """
        t2_tags = set(self.tags)
        for t1 in self.tags:
            t2_tags.remove(t1)
            for t2 in t2_tags:
                for left,right in set(self.tag_stats[t1]).union(set(self.tag_stats[t2])):
                    sim_matrix[t1,t2] += abs(self.tag_stats[t1][left,right] - self.tag_stats[t2][left,right]) / self.context_stats[left,right]
                    counts[t1,t2] += 1
        for t1,t2 in sim_matrix:
            sim_matrix[t1,t2] /= counts[t1,t2]
            sim_matrix[t1,t2] = log(sim_matrix[t1,t2])
        return sim_matrix

    def reset_features(self):
        for lab in self.feat:
            for val in self.feat[lab]:
                for tag in self.feat[lab][val]:
                    self.feat[lab][val][tag] = 1

    # Computes the score of given state represented as TaggedSequence. The
    # score is the weighted sum of inidividual features.
    def state_score(self, st):
        tag = st.near_tag(0)
        score = 0

        for (lab,val) in zip(self.feat_labels, st.all_features()):
            score += self.feat[lab].get(val,{}).get(tag,0)

       	word_prefs = st.word_prefixes()
	for pref in word_prefs:
	    score += self.feat['wpref'].get(pref,{}).get(tag,0)

	word_sufs = st.word_suffixes()
	for suf in word_sufs:
	    score += self.feat['wsuf'].get(suf,{}).get(tag,0)

        return score
     
    def gen_tags(self, word):
        return sorted(self.candidate_tags[word])

    """
    def gen_tags2(self, word):
        tags = set()
        for i in xrange(1,min(9,len(word))+1): # Magic constant (9)
            suf = word[-i:]
            #print suf, self.word_suf_feat.get(suf)
            tags.update(self.feat['wsuf'].get(suf,{}))
        return tags if tags else self.tags
    """

def avg(l):
    return sum(l)/len(l)
