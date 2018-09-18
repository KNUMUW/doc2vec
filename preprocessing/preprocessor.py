from preprocessing.stemmers import Stemmer
from preprocessing.lemmatizers import Lemmatizer
from preprocessing.tokenizers import Tokenizer
from preprocessing.stop_words import StopWords
import pandas as pd
import logging
from functools import reduce
from collections import OrderedDict, Iterable

def get_all_implementations(name,cls):
	subclasses = {scls.__name__ :cls for scls in cls.__subclasses__()}
	subclasses[name] = cls
	return subclasses
	
class Preprocessor:
	def __init__(self, settings): 
		logging.basicConfig(level=logging.DEBUG)
		processors = OrderedDict()
		#Setting default implementations in order of execution
		processors["tokenizer"] = Tokenizer
		processors["lematization"] = Lemmatizer
		processors["stemming"] = Stemmer
		processors["stopwords_remove"]= StopWords
		
		self._processors_dict = OrderedDict([(name, get_all_implementations(name,cls)) for name,cls in processors.items()])
		self._processors = []
		
		for name, dic in self._processors_dict.items():
			if name in settings:
				processor = settings[name]
				if processor == True:
					logging.info('Using default '+str(name))
					self._processors.append(dic[name]())
				elif processor and processor in dic:
					logging.info('Using '+ processor+' as '+name)
					self._processors.append(dic[processor]())
					
		self.to_lower = "lowercase" in settings and settings["lowercase"]
		
	def _preprocess(self, text):
		if self.to_lower:
			text = text.lower()

		preprocessed = reduce(lambda v, preprocessor: preprocessor.process(v), self._processors, text)

		#if isinstance(preprocessed, Iterable) and not isinstance(preprocessed, str):
		#	preprocessed=" ".join(preprocessed)

		return preprocessed
		
	def preprocess(self, df):
		return df.document.apply(self._preprocess)
		
