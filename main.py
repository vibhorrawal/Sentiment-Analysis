# This is the main file which does all the preproccessing on Twitter data and prints predictions
import sys, re
import argparse
import GetOldTweets3 as got
import spacy
from predictions import predict

debug = False
MAX_TWEETS = 20
TOP_TWEETS = True
nlp = spacy.load("en_core_web_sm")

assert sys.version_info[0] >= 3, 'Please use Python 3.x'

def findTweets(query):

	tweetCriteria = got.manager.TweetCriteria()
	tweetCriteria.querySearch = query
	tweetCriteria.maxTweets = MAX_TWEETS
	tweetCriteria.topTweets = TOP_TWEETS
	# tweetCriteria.near = 'India'
	tweets = got.manager.TweetManager.getTweets(tweetCriteria,debug=False)
	if debug:
		print('Tweets downloaded:')
		for tweet in tweets:
			print(tweet.permalink, ':',tweet.text)
			print('::')

	return processTweets(tweets)

def processTweets(tweets):
	stopWords = set(spacy.lang.en.stop_words.STOP_WORDS)
	corpus = []
	tweetText = []
	for tweet in tweets:
		words = tweet.text
		tweetText.append(words)
		words = words.lower()
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
		# corp.append('EOF')

		doc = nlp(words)
		for token in doc:
			t = re.sub('[\s]+|(-PRON-)','',token.lemma_)
			if t not in stopWords and len(t) >= 2:
				corp.append(t)
		corpus.append(corp)

	if debug:
		print(corpus[-5:-1])
	return [corpus,tweetText]

parser = argparse.ArgumentParser(description='Query for twitter mining, can be "search" or @username or #hashtag')
parser.add_argument('-q','--query', help='Add your query',required=False)
parser.add_argument('-d','--debug', help='Debug',required=False)
parser.add_argument('--model', help='Model Selection',required=False)
args = parser.parse_args()

if __name__ == '__main__':
	if args.query is None:
		text = input("Enter Query: ")
	else:
		text = args.query
	if args.debug:
		debug = True
	if args.model is None:
		accuracy = True
	else:
		accuracy = False

	tweets = findTweets(text)
	predictions = predict(tweets[0],debug=debug,accuracy=accuracy)
	for t,p in zip(tweets[1],predictions):
		print('-'*80)
		print("Tweet:")
		print('')
		print(t," \n### Predicted Sentiment[neg, pos]: ",p)
	# print(predictions)
