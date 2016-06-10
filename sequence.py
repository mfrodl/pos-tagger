import re
import copy

class TaggedSequence:
    def __init__(self, seq):
        self.seq = first_word_to_lower(seq)
        self.length = len(seq)
        self.at = 0
        self.word = first_letter_to_lower(seq[0][0])

    def __repr__(self):
        if self.end_of_seq():
            return ' '.join(['{0}/{1}'.format(w,t) for w, t in self.seq]) + ' []'
        else:
            at = self.at
            left = ' '.join(['{0}/{1}'.format(w,t) for (w,t) in self.seq[:at]])
            current = '[' + str(self.seq[at][0]) + '/' + str(self.seq[at][1]) + ']'
            right = ' '.join(['{0}/{1}'.format(w,t) for (w,t) in self.seq[at+1:]])
            return ' '.join(filter(None, [left, current, right]))

    def local_repr(self):
        at = self.at
        start = max(at - 2, 0)
        end = min(at + 3, self.length)
        left = ' '.join(['{0}/{1}'.format(w,t) for (w,t) in self.seq[start:at]])
        current = '[' + self.seq[at][0] + '/' + self.seq[at][1] + ']'
        right = ' '.join(['{0}/{1}'.format(w,t) for (w,t) in self.seq[at+1:end]])
        return ' '.join(filter(None, [left, current, right]))

    def __eq__(self, other):
        if isinstance(other, TaggedSequence):
            return self.seq == other.seq
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())

    def end_of_seq(self):
        return self.at >= self.length

    def shift(self):
        self.at += 1
        self.word = first_letter_to_lower(self.seq[self.at][0]) if not self.end_of_seq() else None

    # Iterates over all fixed-size features and returns them. The variable-size
    # features must be handled separately.
    def all_features(self):
        feats = {}
        for label in feat_templates:
            func, args = feat_templates[label]
            feats[label] = func(self, *args)
        return feats

    def near_tags(self, start, stop):
        tags = []
        for i in range(self.at+start, self.at+stop+1):
            if 0 <= i and i < self.length:
                tags.append(self.seq[i][1])
            else:
                tags.append(None)
        return [tuple(tags)]

    def near_tag(self, pos):
        [(tag,)] = self.near_tags(pos, pos)
        return [tag]

    def tag_prefix(self, pos, length=1):
        [tag] = self.near_tag(pos)
        return [tag[:length] if tag else None]

    def near_words(self, start, stop):
        words = []
        for i in range(self.at+start, self.at+stop+1):
            if 0 <= i and i < self.length:
                words.append(self.seq[i][0])
            else:
                words.append(None)
        return [tuple(words)]

    def near_word(self, pos):
        [(word,)] = self.near_words(pos, pos)
        return [word]

    def word_prefixes(self, maxlength=9):
        prefs = []
        for i in xrange(1,min(maxlength,len(self.word))+1):
            prefs.append(self.word[:i])
        return prefs

    def word_suffixes(self, maxlength=9):
        sufs = []
        for i in xrange(1,min(maxlength,len(self.word))+1):
            sufs.append(self.word[-i:])
        return sufs

    # Returns True iff the word contains at least one hyphen
    def hyphenated(self):
        return [('-' in self.word)]

    # Returns True iff the current word contains at least one digit
    def has_digits(self):
        return [not not re.search(u'[0-9]', self.word)]

    # Returns True iff the current word contains at least one uppercase letter.
    # With default parameter settings, only non-accented letters A-Z are
    # considered uppercase. To include additional language-specific uppercase
    # letter, use the argument `extra_caps'.
    def has_caps(self, extra_caps=''):
        (word,) = self.near_word(0)
        return [not not re.search(u'[A-Z'+extra_caps+']', word)]

class SemiTaggedSequence(TaggedSequence):
    def __init__(self, tagged, untagged):
        self.seq = tagged + [(w,None) for w in untagged]
        self.length = len(self.seq)
        self.at = len(tagged)
        self.word = untagged[0]

    def tag(self, tag):
        self.seq[self.at] = (self.seq[self.at][0],tag)

    def tagged(self, tag):
        ts = copy.deepcopy(self)
        ts.seq[self.at] = (ts.seq[self.at][0],tag)
        return ts

    # NOTE: Does not generalize for different-length contexts.
    def precedes(self, other):
        return self.seq[1][1] == other.seq[0][1] and self.seq[2][1] == other.seq[1][1]

feat_templates = {
    'w0':     (TaggedSequence.near_word, [0]),
    'w-1':    (TaggedSequence.near_word, [-1]),
    'w-21':   (TaggedSequence.near_words, [-2,-1]),
    'w+1':    (TaggedSequence.near_word, [1]),
    'w+12':   (TaggedSequence.near_words, [1,2]),
    'w-2':    (TaggedSequence.near_word, [-2]),
    't-1':    (TaggedSequence.near_tag, [-1]),
    't-21':   (TaggedSequence.near_tags, [-2,-1]),
    't-2':    (TaggedSequence.near_tag, [-2]),
    #'t+1':    (TaggedSequence.near_tag, [1]),     NOT USED
    #'t+12':   (TaggedSequence.near_tags, [1,2]),  NOT USED
    #'t+2':    (TaggedSequence.near_tag, [2]),     NOT USED
    #'t-1[0]': (TaggedSequence.tag_prefix, [-1,1]),
    'hyp':    (TaggedSequence.hyphenated, []),
    'dig':    (TaggedSequence.has_digits, []),
    'cap':    (TaggedSequence.has_caps, []),
    #'pref':   (TaggedSequence.word_prefixes, [3]),
    'suf':    (TaggedSequence.word_suffixes, [6])
}

def first_word_to_lower(seq):
    w0, t0 = seq[0]
    return [(w0.lower(), t0)] + seq[1:]

def first_letter_to_lower(word):
    return word[0].lower() + word[1:]
