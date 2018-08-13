from nltk.tokenize import wordpunct_tokenize, word_tokenize

class Tokenizer():
	def process(self,text):
		return  wordpunct_tokenize(text)
		
class WordTokenizer(Tokenizer):
	def process(self,text):
		return word_tokenize(text)