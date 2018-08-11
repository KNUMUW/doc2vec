from nltk.stem.wordnet import WordNetLemmatizer

class Lemmatizer():
	
	def __init__(self): 
		self._lem = WordNetLemmatizer()

	def process(self,words):
		return  [self._lem.lemmatize(word) for word in words]
		