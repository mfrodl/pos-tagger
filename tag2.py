# -*- coding: utf-8 -*-

from cmath import exp, pi
from numpy import log
from collections import Counter, defaultdict as dd
from itertools import combinations
from nltk import TaggerI
from operator import mul, itemgetter
import pickle
import random
import re
import time

import model2
from sequence import TaggedSequence, SemiTaggedSequence
from perceptron import Perceptron

inf = float('inf')

class Tagger(TaggerI):
    def __init__(self, sents, load = False, save = False):
        print 'Initializing tagger...'
        #self.state_word_score = self.model.state_word_score
        #self.state_tag_score = self.model.state_tag_score

        self.model = model2.Model(sents)
        print len(self.model.tags), 'tags'
        self.gen_tags = self.model.gen_tags

        if load:
            print 'Loading the trained neurons...'
            self.neurons = pickle.load(open('neurons.pkl', 'rb'))
            print len(self.neurons), 'units'
            print 'Done.'
            #print 'Loading probabilities...'
            #self.prob_dists = pickle.load(open('probs.pkl', 'rb'))
            #print 'Done.'
        else:
            print 'Learning the training data...'
            self.learn(sents, save = save)

    def load(self, f):
        print 'Loading statistical model...'
        self.model = pickle.load(f)
        print 'Model loaded.'

    def dump(self, f):
        print 'Dumping statistical model...'
        pickle.dump(self.model, f)
        print 'Model dumped.'

    def tag(self, sent):
        """
        print '(HMM) Tagging: <{0}>'.format(' '.join(sent))
        vit = viterbi.Viterbi(self.model.state_score, self.gen_tags, self.model.gen_tags2)
        sent_tags = [st.near_tag(0) for st in vit.solve(sent)]
        return zip(sent, sent_tags)
        """
        #print sorted(self.neurons.keys())
        #print sorted(self.prob_dists.keys())
        sts = SemiTaggedSequence([], sent)
        #print sts.all_features()
        while not sts.end_of_seq():
            [word] = sts.near_word(0)
            cand_tags = self.gen_tags(word)
            #print '----------'
            #print sts
            #print word, cand_tags
            if len(cand_tags) == 1:
                [tag] = cand_tags
                #print '->', tag, '(unambiguous)'
            else:
                feat_vect = self.feat_vect(sts, short=True)
                if not cand_tags:
                    if word[0].isupper():
                        cand_tags = self.gen_tags(word[0].lower() + word[1:])
                    if not cand_tags:
                        #print 'Unknown word:', word
                        cand_tags = self.model.tags # TODO: Could be optimized using a suffix guesser?

                scores = {ct: self.neurons[ct].wsum(feat_vect) for ct in cand_tags}
                tag = argmax(scores)
                #print scores, '--->', tag

                #outputs = {}
                #prob_dist = ProbDist({})
                #for amb_pair in self.neurons:
                #    out = self.neurons[amb_pair].activate(feat_vect)
                #    #outputs[amb_pair] = out
                #    prob_dist *= self.prob_dists[amb_pair][out]
                #print prob_dist

                #tag = argmax(prob_dist.probs)
                #print tag

                #for amb_pair in self.prob_dist:
                #    print self.prob_dist[amb_pair]

                """"
                tag_scores = Counter()
                tag_nodes = set([tag for pair in set(self.neurons).intersection(cand_pairs) for tag in pair])
                print tag_nodes
                for cand_pair in set(cand_pairs).intersection(set(self.neurons.keys())):
                    out = self.neurons[cand_pair].activate(feat_vect)
                    print cand_pair, '->', cand_pair[out]
                    loser = cand_pair[1-out]
                    if loser in tag_nodes:
                        tag_nodes.remove(loser)
                print tag_nodes
                tag = tag_nodes.pop()
                """

            sts.tag(tag)

            sts.shift()
        #print sts.seq
        return sts.seq

    def default_tag(self, sent, tag='N'):
        print '(DEF) Tagging: <{0}>'.format(' '.join(sent))
        return [(w,tag) for w in sent]

    def features_to_vector(self, features, short=False):
        true_indices = []
        for label, values in features.items():
            for value in values:
                if value in self.model.feat_index[label]:
                    index = self.model.feat_index[label][value]
                    true_indices.append(index)

        if short:
            return true_indices
        else:
            return self.short_to_long(true_indices)

    def feat_vect(self, ts, short=False):
        return self.features_to_vector(ts.all_features(), short)

    def short_to_long(self, short_vector):
        long_vector = [1. for _ in xrange(self.model.num_feats)]
        for index in short_vector:
            long_vector[index] = -1. # True
        return long_vector

    def gen_samples(self, sents, visual=False):
        samples = dd(list)
        for sent in sents:
            ts = TaggedSequence(sent)
            while not ts.end_of_seq():
                # DATED:
                # Input vector contains -1's (True) at positions which
                # correspond to features present in this particular context.
                # The other components are set to 1 (False).
                in_vect = self.feat_vect(ts, short=True)

                [word] = ts.near_word(0)
                [tag] = ts.near_tag(0)

                if visual:
                    samples[tag].append((in_vect, ts.local_repr()))
                else:
                    samples[tag].append(in_vect)

                ts.shift()

        return samples

    def learn(self, sents, visual=False, save=False):
        samples = self.gen_samples(sents, visual)
        in_vects = dd(list)
        if visual:
            for tag in samples:
                for in_vect, rep in samples[tag]:
                    in_vects[tag].append(in_vect)
        else:
            in_vects = samples

        #ambig_pairs = sorted(self.model.ambig_cls.items(), key = itemgetter(1), reverse = True)
        #ambig_pairs = [tags for tags, _ in ambig_pairs]
        #ambig_pairs = filter(lambda x: len(x) == 2, ambig_pairs)

        self.neurons = {}
        #sample_freqs = {}
        #sample_rel_freqs = {}
        #self.prob_dists = {}
        #print len(ambig_pairs), 'ambiguity pairs'
        for neuron in self.model.tags:
            self.neurons[neuron] = Perceptron(self.model.num_feats, 0.1, short=True, guess_weights=True)
            #tag0, tag1 = cand_pair
            print 'Training the {0} neuron...'.format(neuron)
            train_samples = [(in_vect, 1) for in_vect in in_vects[neuron]]
            for tag in set(in_vects) - {neuron}:
                train_samples += [(in_vect, 0) for in_vect in in_vects[tag]]
            self.neurons[neuron].learn(train_samples)
            print 'Done.'

            # Keeping for possible future use
            # NOTE: delete if not used
            """  
            # Compute probability distributions for each half-space
            print 'Computing probability distribution for neuron <{0}>...'.format(neuron)
            sample_freqs[neuron] = [Counter(), Counter()]
            for tag in in_vects:
                for in_vect in in_vects[tag]:
                    out = self.neurons[neuron].activate(in_vect)
                    sample_freqs[neuron][out][tag] += 1

            self.prob_dists[neuron] = {}
            for out in [0, 1]:
                #print '*', out
                self.prob_dists[neuron][out] = ProbDist(sample_freqs[neuron][out])

            print neuron, self.prob_dists[neuron]
            """

        if save:
            print 'Saving the trained neurons...'
            f = open('neurons.pkl', 'wb')
            pickle.dump(self.neurons, f)
            f.close()
            print 'Done.'

            # Keeping for possible future use
            # NOTE: delete if not used
            """
            print 'Saving probability distributions...'
            f = open('probs.pkl', 'wb')
            pickle.dump(self.prob_dists, f)
            f.close()
            print 'Done.'
            """

    def inspect(self):
        for amb_pair in self.neurons:
            sorted_weights = sorted(self.neurons[amb_pair].weights.items(), key = itemgetter(1), reverse = True)
            for feat_num, score in sorted_weights[:5] + sorted_weights[-5:]:
                feat_num -= 1
                print amb_pair
                feat_lab, feat_val = self.model.feat_by_num(feat_num)
                print feat_num, ':', feat_lab, '=', feat_val, score
            raw_input()

    """
    def evaluate(self, sents):
        samples = self.gen_samples(sents)
        right = wrong = 0

        num_samples = sum(map(len, samples.values()))

        i = 1
        confusions = {}
        for tag in samples:
            confusions[tag] = dd(int)
            for features, in_vect in samples[tag]:
                #print 'Sample {0}/{1}'.format(i, num_samples)
                #long_vector = self.short_to_long(in_vect)
                long_vector = {i: -1. for i in in_vect}
                [word] = features['w0']
                #if word != 'to':
                #    continue
                candidate_clusters = filter_dict(self.model.gen_tags(word), self.clusters)
                if len(candidate_clusters) == 1:
                    cl = candidate_clusters.keys()[0]
                else:
                    cl_quads = self.cluster_quadrances(long_vector, candidate_clusters)

                    print 'CC before:', candidate_clusters
                    for cl, quad in cl_quads.items():
                        if quad > self.radiuses[cl]:
                            del candidate_clusters[cl]
                    print 'CC after:', candidate_clusters

                    cl = self.nearest_cluster(long_vector, candidate_clusters)

                confusions[tag][cl] += 1
                if cl == tag:
                    right += 1
                else:
                    wrong += 1
                    #if True:#tag == 'P' and cl == 'TO':
                    print 'Word {0}: TO = {1:.2f}, P = {2:.2f} (should be {3})'.format(\
                          word, self.quadrance(long_vector, self.clusters['TO']), self.quadrance(long_vector, self.clusters['P']), tag)
                    print '{0}/{1} Word {2} classified as {3} ({4:.2f}) should be {5} ({6:.2f}), farthest {7} ({8:.2f})'.format( \
                          i, num_samples, word, cl, self.quadrance(long_vector, self.clusters[cl]), tag,
                          self.quadrance(long_vector, self.clusters[tag]) if tag in self.clusters else inf, \
                          self.farthest_cluster(long_vector, self.clusters), \
                          self.quadrance(long_vector, self.clusters[self.farthest_cluster(long_vector, self.clusters)]) \
                    ),
                    print '(accuracy = {0:.2f}%)'.format(100 * (float(right) / float(right+wrong)))
                    #print self.nearest_clusters(long_vector, self.clusters)
                    print '{0}/{1} Word {2} classified as {3} should be {4}'.format(i, num_samples, word, cl, tag),
                    print '(accuracy = {0:.2f}%)'.format(100 * (float(right) / float(right+wrong)))
                i += 1
        print 'Right:', right
        print 'Wrong:', wrong
        print 'Accuracy: {:.02f}%'.format(100 * (float(right) / float(right+wrong)))
        f = open('confusions', 'w')
        f.write('      ')
        for tag in confusions:
            f.write('{0:>{1}}'.format(tag, len(tag)+2))
        f.write('\n')
        for tag in confusions:
            f.write('{0:>{1}}'.format(tag, 6))
            for cl in confusions:
                f.write('{0:>{1}}'.format(confusions[tag][cl] if cl in confusions[tag] else '', len(cl)+2))
            f.write('\n')
        f.close()
    """

def argmax(dictionary):
    return max(dictionary.items(), key = lambda x: x[1])[0]

def argmin(dictionary):
    return min(dictionary.items(), key = lambda x: x[1])[0]

def random_from_dict(dictionary):
    cum_prob = 0
    threshold = random.random()
    for key, prob in dictionary.items():
        cum_prob += prob
        if cum_prob > threshold:
            return key

def filter_dict(keys, dictionary):
    if keys:
        return {key: value for key, value in dictionary.items() if key in keys}
    else:
        return dictionary

def add_lists(list1, list2):
    assert len(list1) == len(list2), (len(list1), len(list2))
    return [x + y for x, y in zip(list1, list2)]
