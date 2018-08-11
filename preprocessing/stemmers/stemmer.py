from nltk.stem.porter import *

class PorterStemmerWithoutToLower(PorterStemmer):
    def stem(self, word):
        stem = word
        
        if self.mode == self.NLTK_EXTENSIONS and word in self.pool:
            return self.pool[word]

        if self.mode != self.ORIGINAL_ALGORITHM and len(word) <= 2:
            return word

        stem = self._step1a(stem)
        stem = self._step1b(stem)
        stem = self._step1c(stem)
        stem = self._step2(stem)
        stem = self._step3(stem)
        stem = self._step4(stem)
        stem = self._step5a(stem)
        stem = self._step5b(stem)
        
        return stem

class Stemmer():
    def __init__(self):
        self._stemmer = PorterStemmerWithoutToLower()
    def process(self,words):
        
        return [ self._stemmer.stem(word) for word in words]
        
