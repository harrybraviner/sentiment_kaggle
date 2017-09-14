#! /usr/bin/python3

import re, random
import numpy as np

class DataSet:

    def __init__(self, training_fraction = 0.75):

        f = open('data/train.tsv', 'r')
        f.readline()    # Throw away first line, it's a header
        
        self._phraseIds = []
        self._sentenceIds = []
        self._phrase = []
        self._sentiment = []

        regex = re.compile(r"(\d+)\s+(\d+)\s+(.*)\s+(\d+)")

        for l in f:
            result = regex.match(l)
            phraseId, sentenceId, phrase, sentiment = result.groups()
            self._phraseIds += [int(phraseId)]
            self._sentenceIds += [int(sentenceId)]
            self._phrase += [phrase]
            self._sentiment += [int(sentiment)]

        f.close()

        self.N_total = len(self._phraseIds)
        self.N_train = int(self.N_total * training_fraction)
        self.N_validation = self.N_total - self.N_train

        self._rng = random.Random()
        self._rng.seed(1234)

    def get_training_case(self):
        i = self._rng.randint(0, self.N_train - 1)

        return self._phrase[i], self._sentiment[i]

    def get_validation_set(self):
        return [ (self._phrase[i], self._sentiment[i])
                 for i in range(self.N_train, self.N_total) ]

class GloveDictionary:

    def __init__(self):

        f = open('data/glove.6B.50d.txt')

        self._word_vector_table = {}
        
        for l in f:
            l = l.split(' ')
            word = l[0]
            vector = np.array([float(x) for x in l[1:]], dtype = np.float32)
            self._word_vector_table[word] = vector

        f.close()

        self.dw = len([v for v in self._word_vector_table.values()][0])

    def try_get_word(self, word):
        try:
            return self._word_vector_table[word]
        except:
            return None
