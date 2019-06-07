import sys,re
import GetOldTweets3 as got
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

debug = True
MAX_TWEETS = 20
TOP_TWEETS = True
nlp = spacy.load("en_core_web_sm")
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

assert sys.version_info[0] >= 3, 'Please use Python 3.x'

def findTweets(query):
	
	tweetCriteria = got.manager.TweetCriteria()
	tweetCriteria.querySearch = query
	tweetCriteria.maxTweets = MAX_TWEETS
	tweetCriteria.topTweets = TOP_TWEETS
	# tweetCriteria.near = 'India'
	tweets = got.manager.TweetManager.getTweets(tweetCriteria,debug=debug)
	if debug:
		for tweet in tweets:
			print(tweet.permalink, ':',tweet.text)
			print('::')

	return processTweets(tweets)		

def processTweets(tweets):
	stopWords = set(spacy.lang.en.stop_words.STOP_WORDS)
	corpus = []
	for tweet in tweets:
		words = tweet.text.lower()
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

if __name__ == '__main__':
	
	if debug:
		text = 'zine'
	else:
		text = input()
	tweets = findTweets(text)
	print(tweets[0])