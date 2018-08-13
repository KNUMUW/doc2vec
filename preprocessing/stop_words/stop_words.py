from nltk.corpus import stopwords

class StopWords():
	def process(self,words):
		return [word for word in words if word not in stopwords.words('english')]