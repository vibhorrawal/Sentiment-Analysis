import sys,re
import GetOldTweets3 as got
import spacy
nlp = spacy.load("en_core_web_sm")
debug = False
def processTweets(tweets):
	stopWords = set(spacy.lang.en.stop_words.STOP_WORDS)
	corpus = []
	for i in range(1):
		words = tweets.text.lower()
		words = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|\s[\w]+\.\w{2,3}[^\s]+)', '', words) # remove URLs
		words = re.sub('#([^\s]+)', '', words) # remove the # in #hashtag
		words = re.sub('@[^\s]+', '', words) # remove usernames
		words = re.sub('[^\w]', ' ', words) # remove non word characters
		words = re.sub('[\d]+', '' , words) # remove numbers
		corp = []
		if debug:
			words1 = words.split()
			for word in words1:
				if word not in stopWords:
					corp.append(word)
			
		doc = nlp(words)
		for token in doc:
			t = re.sub('[\s]+|(-PRON-)','',token.lemma_)
			if t not in stopWords and len(t) > 2:
				corp.append(t)		
		corpus.append(corp)
	
	if debug:
		print(corpus[-5:-1])

	return corpus	


class Test():
	"""docstring for test"""
	def __init__(self, text):
		self.text = text
		

if __name__ == '__main__':
	test = Test('I was walking down the road when it started raining')
	# print(test.text)
	print(processTweets(test))