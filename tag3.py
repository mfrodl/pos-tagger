from collections import Counter
from nltk import TaggerI
from nltk import DefaultTagger, AffixTagger, UnigramTagger, BigramTagger, TrigramTagger
from operator import itemgetter
import pickle
import time

class Tagger(TaggerI):
    def __init__(self, train_sents, load = False):
        if load:
            print 'Loading saved tagger...',
            self.load()
            print 'done.'
        else:
            time_start = time.time()

            print 'Training the tagger...'
            tag_counts = Counter([t for s in train_sents for w, t in s])
            default_tag = argmax(tag_counts)

            def_tgr = DefaultTagger(default_tag)
            af_tgr = AffixTagger(train_sents, affix_length=-3, backoff=def_tgr)
            uni_tgr = UnigramTagger(train_sents, backoff=af_tgr)
            bi_tgr = BigramTagger(train_sents, backoff=uni_tgr)
            tri_tgr = TrigramTagger(train_sents, backoff=bi_tgr)
            self.tgr = tri_tgr
            print 'Done.'

            time_stop = time.time()
            print 'Training time: {0:.2f}s'.format(time_stop - time_start)

    def load(self):
        tagger_file = open('baseline.pkl', 'rb')
        self.tgr = pickle.load(tagger_file)
        tagger_file.close()

    def save(self):
        tagger_file = open('baseline.pkl', 'wb')
        pickle.dump(self.tgr, tagger_file)
        tagger_file.close()

    def tag(self, sent):
        return self.tgr.tag(sent)

def argmax(d):
    return max(d.items(), key=itemgetter(1))[0]
