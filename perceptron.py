import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from collections import Counter, defaultdict as dd

class Perceptron:
    def __init__(self, in_size, eps=0.1, short=True, guess_weights=False):
        self.in_size = in_size
        self.guess_weights = guess_weights
        if not guess_weights:
            self.weights = [random.uniform(-0.1, 0.1) for _ in xrange(in_size + 1)]
        self.eps = eps

        # Short format vector contains the indices of positions to be set to 1,
        # the other are taken to be 0.
        self.short = short

    def activate(self, in_vect):
        if self.short:
            wsum = self.weights.get(0, 0.) + sum([self.weights.get(i+1, 0.) for i in in_vect])
        else:
            assert len(in_vect) == self.in_size, (len(in_vect), self.in_size)
            ext_in_vect = [1] + in_vect
            wsum = sum([w * x for w, x in zip(self.weights, ext_in_vect)])
        #print wsum
        return self.threshold(wsum)

    def threshold(self, wsum):
        return 1 if wsum >= 0 else 0

    def wsum(self, in_vect):
        if self.short:
            return self.weights.get(0, 0.) + sum([self.weights.get(i+1, 0.) for i in in_vect])
        else:
            assert len(in_vect) == self.in_size, (len(in_vect), self.in_size)
            ext_in_vect = [1] + in_vect
            return sum([w * x for w, x in zip(self.weights, ext_in_vect)])

    def learn(self, samples):
        self.batch_learn(samples)

    def online_learn(self, samples, epochs):
        for e in xrange(epochs):
            tp, tn, fp, fn = self.accuracy(samples)
            correct, incorrect = tp + tn, fp + fn
            #print 'Correct: {0}, incorrect: {1}'.format(correct, incorrect)
            print 'TP: {0}, TN: {1}, FP: {2}, FN: {3} (ACC = {4:.2f}%'.format(tp, tn, fp, fn, float(100*correct)/float(correct + incorrect))
            if incorrect == 0:
                break
            for in_vect, desired in samples:
                out = self.activate(in_vect)
                delta = out - desired
                if delta == 0:
                    continue
                self.weights = [w - self.eps * delta * x for w, x in zip(self.weights, [1] + in_vect)]

    def batch_learn(self, samples):
        time0 = time.time()

        # Guess initial weights using a Centroid classifier
        if self.guess_weights:
            print 'Computing initial weights...'
            inputs0 = [in_vect for in_vect, out in samples if out == 0]
            inputs1 = [in_vect for in_vect, out in samples if out == 1]

            center0 = self.find_center(inputs0)
            center1 = self.find_center(inputs1)

            weights = self.sub_vectors(center1, center0)
            weights = {k + 1: v for k, v in weights.items()}
            weights[0] = 0.

            weights[0] = (self.squared_norm(center0) - self.squared_norm(center1)) / 2
            self.weights = weights

            # Estimate bias using the Scut method
            """
            print 'Computing initial bias...'
            best_corr = 0
            best_bias = 0.
            incorr = 0
            num_samples = len(samples)
            for in_vect, out in samples:
                self.weights[0] = 0.
                bias = -self.wsum(in_vect)
                self.weights[0] = bias
                tp, tn, fp, fn = self.accuracy(samples)
                corr = tp + tn
                if corr > best_corr:
                    print 'Best accuracy: {0}/{1}'.format(corr, num_samples)
                    best_corr = corr
                    best_bias = bias
                    if corr == num_samples:
                        break
                    incorr = 0
                else:
                    incorr += 1
                if incorr > 30:
                    break

            self.weights[0] = best_bias
            #"""
        best_correct = 0
        last_correct = 0
        best_weights = self.weights
        last_weights = self.weights
        bad_corrections = 0
        e = 1
        div = 1
        while bad_corrections < 30:
            tp, tn, fp, fn = self.accuracy(samples)
            correct, incorrect = tp + tn, fp + fn
            acc = float(correct) / float(correct + incorrect)
            print 'Epoch {0}: TP {1}, TN {2}, FP {3}, FN {4} (acc = {5:.2f}%)'.format(\
                e, tp, tn, fp, fn, 100 * acc),
            if correct > best_correct:
                print '<-- best'
                best_correct = correct
                best_weights = self.weights
                bad_corrections = 0
            else:
                print
                bad_corrections += 1

            """
            if correct > last_correct:
                div = 1
            else:
                self.weights = last_weights
                div *= 2
            #"""

            #if acc > 0.95:
            #    break
            if incorrect <= 1:
                break

            """
            if div == 1:
                print 'Updating last correct & weights...'
                last_correct = correct
                last_weights = self.weights
            #"""

            weight_errors = dd(float)
            #if incorrect < 300:
            #    chosen_samples = samples
            #else:
            #    chosen_samples = choose_some(samples, 0.05)
            random.shuffle(samples)
            corrections = 0
            for in_vect, desired in samples:
                out = self.activate(in_vect)
                delta = out - desired
                if delta == 0:
                    continue
                if self.short:
                    ext_in_vect = dd(float, {k + 1: 1. for k in in_vect})
                    ext_in_vect[0] = 1.
                else:
                    ext_in_vect = [1.] + in_vect
                weight_errors = self.add_vectors(weight_errors, self.scalar_mult(delta, ext_in_vect))
                corrections += 1
                if corrections > 100:
                     break

            sqe = self.squared_norm(weight_errors)
            if sqe > 0:
                eps = -(self.dot(self.weights, weight_errors)) / (div * sqe)
                self.weights = self.add_vectors(self.weights, self.scalar_mult(eps, weight_errors))

            e += 1

        self.weights = best_weights

    def find_center(self, vectors):
        if self.short:
            feat_counts = Counter()
            for vect in vectors:
                feat_counts.update(vect)
            center = dd(float) #self.in_size * [0.]
            for feat_num in feat_counts:
                center[feat_num] = float(feat_counts[feat_num]) / len(vectors)
        else:
            center = [sum(x) / len(x) for x in zip(*vectors)]

        return center

    def accuracy(self, samples):
        tp = tn = fp = fn = 0
        for in_vect, desired in samples:
            out = self.activate(in_vect)
            if out == 1:
                 if desired == 1:
                     tp += 1
                 else:
                     fp += 1
            else:
                 if desired == 1:
                     fn += 1
                 else:
                     tn += 1

        return tp, tn, fp, fn

    def short2long(self, short_vect):
        long_vect = [0. for _ in xrange(self.in_size)]
        for k in short_vect:
            long_vect[k] = 1.
        return long_vect

    # Dot product
    def dot(self, vect1, vect2):
        assert isinstance(vect1, type(vect2)) or isinstance(vect2, type(vect1)), (type(vect1), type(vect2))
        if isinstance(vect1, dict):
            return sum([vect1.get(k,0.) * vect2.get(k,0.) for k in set(vect1).union(vect2)])
        else:
            assert len(vect1) == len(vect2), (len(vect1), len(vect2))
            return sum([x * y for x, y in zip(vect1, vect2)])

    # Euclidean norm of vector
    def squared_norm(self, vect):
        return self.dot(vect, vect)

    def add_vectors(self, vect1, vect2):
        assert isinstance(vect1, type(vect2)) or isinstance(vect2, type(vect1)), (type(vect1), type(vect2))
        if isinstance(vect1, dict):
            return {k: vect1.get(k,0.) + vect2.get(k,0.) for k in set(vect1).union(vect2)}
        else:
            assert len(vect1) == len(vect2), (len(vect1), len(vect2))
            return [x + y for x, y in zip(vect1, vect2)]

    def sub_vectors(self, vect1, vect2):
        assert isinstance(vect1, type(vect2)) or isinstance(vect2, type(vect1)), (type(vect1), type(vect2))
        if isinstance(vect1, dict):
            neg_vect2 = {k: -v for k, v in vect2.items()}
        else:
            neg_vect2 = [-x for x in vect2]
        return self.add_vectors(vect1, neg_vect2)

    def scalar_mult(self, scalar, vect):
        if isinstance(vect, dict):
            return {k: scalar * v for k, v in vect.items()}
        else:
            return [scalar * x for x in vect]

def choose_some(samples, portion):
    num = len(samples)
    chosen = []
    for i in xrange(num):
        if random.random() < portion:
            chosen.append(samples[i])
    return chosen

def choose_n_samples(samples, n):
    num = len(samples)
    if num <= n:
        return samples

    indices = range(num)
    chosen = []
    for i in xrange(n):
        r = random.randint(0, num-1)
        ind = indices[r]
        chosen.append(samples[ind])
        del indices[r]
        num -= 1

    return chosen

if __name__ == '__main__':

    #"""
    xs0 = [random.gauss(-1, .7) for _ in xrange(1000)]
    ys0 = [random.gauss(1, .7) for _ in xrange(1000)]
    #zs0 = [random.uniform(-1, 1) for _ in xrange(1000)]
    samples0 = [([x, y] + [0 for _ in xrange(1000)], 0) for x, y in zip(xs0, ys0)]

    xs1 = [random.gauss(1, .7) for _ in xrange(1000)]
    ys1 = [random.gauss(-1, .7) for _ in xrange(1000)]
    #zs1 = [random.uniform(-1, 1) for _ in xrange(1000)]
    samples1 = [([x, y] + [0 for _ in xrange(1000)], 1) for x, y in zip(xs1, ys1)]
    #"""

    """
    center0 = [sum(xs0) / len(xs0), sum(ys0) / len(ys0)]
    center1 = [sum(xs1) / len(xs1), sum(ys1) / len(ys1)]
    wgts = [c1 - c0 for c0, c1 in zip(center0, center1)]
    bias = (sum([c0 * c0 for c0 in center0]) - sum([c1 * c1 for c1 in center1])) / 2

    cx, cy = [[w/2] for w in wgts]
    wx, wy = [[w] for w in wgts]
    """
    #samples0 = [([0, 1], 0) for _ in xrange(1000)]
    #samples1 = [([0, 2], 1) for _ in xrange(1000)]

    samples = samples0 + samples1
    random.shuffle(samples)

    p = Perceptron(1002, short=False, guess_weights=True)
    p.learn(samples, 10)

    """
    w0, w1, w2 = p.weights[:3]

    linex = np.arange(-4, 4, 1)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    plt.plot(xs0, ys0, 'ro')
    plt.plot(xs1, ys1, 'bo')
    #linex, liney = np.meshgrid(linex, liney)
    #ax.plot_wireframe(linex, liney, -(w0/w3)-(w1/w3)*linex-(w2/w3)*liney, 'k')
    plt.plot(linex, -(w0/w2)-(w1/w2)*linex, 'k')

    plt.axis([-3,3,-3,3])
    plt.show()
    #"""
