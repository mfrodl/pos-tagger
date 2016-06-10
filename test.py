from nltk.corpus import brown
import sys
import time

import tag1, tag2, tag3

def main(method, train_size, test_size, load = False, save = False, output = False):
    sents = brown.tagged_sents(simplify_tags = True)

    training_start_time = time.time()

    train_sents = []
    #if not load:
    print 'Retrieving training set...',
    train_sents = sents[:train_size]
    num_train_sents = len(train_sents)
    num_train_words = sum(map(len,train_sents))
    print 'done ({0} sentences, {1} tokens)'.format(num_train_sents, num_train_words)

    if len(train_sents) == 0:
        print 'Cannot learn from an empty training set.'
        sys.exit(1)

    print 'Retrieving evaluation set...',
    test_sents = sents[-test_size:]
    num_test_sents = len(test_sents)
    num_test_words = sum(map(len,test_sents))
    print 'done ({0} sentences, {1} tokens)'.format(num_test_sents, num_test_words)

    tagger = None
    if method == 1:
        tagger = tag1.Tagger(train_sents, load)
    elif method == 2:
        tagger = tag2.Tagger(train_sents, load)
    else:
        tagger = tag3.Tagger(train_sents, load)

    training_stop_time = time.time()

    if output:
        print 'Tagging the test sentences...'
        untagged_test_sents = [[w for (w,_) in s] for s in test_sents]
        tagged_test_sents = tagger.tag(untagged_test_sents)

        outf = open('a{0}.out'.format(train_size), 'w')
        for (ts,tts) in zip(test_sents,tagged_test_sents):
            for ((w,ct),(_,t)) in zip(ts,tts):
                outf.write('{0:<20}{1:<10}{2:<10}{3}\n'.format(w,ct,t,ct==t))
            outf.write('\n')
        outf.close()

    eval_start_time = time.time()

    print 'Evaluating the tagger...'
    accuracy = tagger.evaluate(test_sents)
    print 'Accuracy: {0:.2f}%'.format(100 * accuracy)

    eval_stop_time = time.time()

    if save:
        print 'Saving the tagger...'
        tagger.save()

    saving_stop_time = time.time()

    if not load:
        print 'Training time: {0:.2f}s'.format(training_stop_time - training_start_time)
    print 'Evaluation time: {0:.2f}s'.format(eval_stop_time - eval_start_time)
    if save:
        print 'Saving time: {0:.2f}s'.format(saving_stop_time - eval_stop_time)

def usage():
    print 'Usage: test.py [-l] [-s] [-o] -m<METHOD> <train_size> <test_size>'
    print
    print 'mandatory arguments:'
    print '<train_size>  number of sentences used for training'
    print '<test_size>   number of sentences used for testing'
    print
    print 'optional arguments:'
    print '-l, --load    load the trained tagger from file'
    print '-s, --save    save the tagger to file after training'
    print '-o, --output  save tagged test sentences to file'
    print '-m<METHOD>    select the method that will be used for'
    print '              training and tagging, possible values of'
    print '              <METHOD> are:'
    print '              1 = recurrent MLP network'
    print '              2 = feature-based perceptron'
    print '              3 = backoff tagger'
    print
    sys.exit(1)

if __name__ == '__main__':
    argv = sys.argv
    try:
        [train_size, test_size] = map(int, argv[-2:])
    except ValueError:
        usage()

    method = 0
    if '-m1' in argv:
        method = 1
    elif '-m2' in argv:
        method = 2
    elif '-m3' in argv:
        method = 3
    else:
        usage()

    load = '-l' in argv
    save = '-s' in argv
    output = '-o' in argv
    main(method, train_size, test_size, load, save, output)

