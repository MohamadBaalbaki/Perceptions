import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn import metrics
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import codecs
from time import time
from nltk.corpus import stopwords
from nltk import download
download('stopwords')  
import os
# from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import re
import requests
import sys
from emoji import UNICODE_EMOJI

#Step 3
def CleanEmojis(sentence):
	emojiList=list(UNICODE_EMOJI.keys())
	for emojiIndex in range(len(emojiList)):
		for sentenceIndex in range(len(sentence)):
			if emojiList[emojiIndex] in sentence[sentenceIndex]:
				sentence[sentenceIndex] = sentence[sentenceIndex].replace(emojiList[emojiIndex],'')
	for i in range(len(sentence)):
		if bool(re.search(r'\\u.{4}',sentence[i])):
			#sentence[i]=re.sub(r'\\u.{4}','',sentence[i])
			del sentence[i]
	sentence = list(filter(None, sentence)) #remove empty strings
	#print("After emoji cleaning: ",sentence)
	#print()
	return(sentence)

#Step 2
def CleanStopWords (sentence):
		stop_words = stopwords.words('arabic')  
		sentenceSplitted = sentence.split(" ")  
		sentence = [w for w in sentenceSplitted if w not in stop_words]
		sentence = CleanEmojis(sentence)
		return(sentence)

#Step 1
def ExtractVectors(sentence, removeStopwords):
		sentence = sentence.rstrip()
		#REMOVE UNNECESSARY STUFF HERE
		#print(sentence)
		sentence = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', sentence) #remove urls
		sentence = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z0-9_]+)','', sentence) #remove mentions
		bad_chars = ['!','$','%','^','&','*','~','«','»','”','“','•','❤','،','…','(',')','〰','#','+','=','[',']','{','}','\\','|','.','?',':',';','؟','"'] #remove punctuations
		for i in bad_chars : 
    			sentence = sentence.replace(i, '') 		
		sentence = re.sub(r'[0-9]+','',sentence) #remove english numbers
		arabic_numbers=['٠','١','٢','٣','٤','٥','٦','٧','٨','٩'] #remove arabic numbers
		for i in arabic_numbers: 
    			sentence = sentence.replace(i, '') 
		sentence = sentence.replace('_', ' ') #replace underscores by spaces
		sentence = sentence.replace('-', ' ') #replace underscores by spaces
		#print("Tweet after cleaning from things: ",sentence)
		if (removeStopwords):
			sentence = CleanStopWords(sentence)
			sentence = ' '.join(sentence)
		sv = model_ft.get_sentence_vector(sentence)
		#vector = sv.reshape(1, -1)
		#print("Extracted vector: ",sv)
		return(sv) #it was return vector


print("Loading FT model")
model_ft  = fasttext.load_model('/home/baalbaki/Desktop/FastText/cc.ar.300.bin')

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

dataset = pd.read_csv(r'dataset4.csv')
X, y = dataset.tweet, dataset.perception #X now holds the tweets and y holds the labels

##############
training_tweets, testing_tweets, training_labels, testing_labels = train_test_split(X, y, test_size=0.2, random_state=42)


#testing_tweets, validation_tweets, testing_labels, validation_labels = train_test_split(testing_tweets, testing_labels, test_size=0.5, random_state=42)
##############

print(len(training_tweets))
print(len(testing_tweets))
#print(len(validation_tweets))

#Transforming training tweets into vectors
training_tweets_vectors=[]
for t in training_tweets.iteritems():
	myId = t[0]	
	myTweet = t[1]
	myTweet= myTweet.replace('\r\n', ' ')
	myTweet= myTweet.replace('\n', ' ')
	sentenceVec = ExtractVectors(myTweet, True)
	training_tweets_vectors.append(sentenceVec)
training_tweets_vectors=np.array(training_tweets_vectors)

#Transforming training labels into an array
training_labels_array=[]
for t in training_labels.iteritems():
	myId = t[0]	
	myLabel = t[1]
	training_labels_array.append(myLabel)
training_labels_array=np.array(training_labels_array)

#Transforming testing tweets into vectors
testing_tweets_vectors=[]
for t in testing_tweets.iteritems():
	myId = t[0]	
	myTweet = t[1]
	myTweet= myTweet.replace('\r\n', ' ')
	myTweet= myTweet.replace('\n', ' ')
	sentenceVec = ExtractVectors(myTweet, True)
	testing_tweets_vectors.append(sentenceVec)
testing_tweets_vectors=np.array(testing_tweets_vectors)

#Transforming testing labels into an array
testing_labels_array=[]
for t in testing_labels.iteritems():
	myId = t[0]	
	myLabel = t[1]
	testing_labels_array.append(myLabel)
testing_labels_array=np.array(testing_labels_array)

#Transforming validation tweets into vectors
#validation_tweets_vectors=[]
#for t in validation_tweets.iteritems():
#	myId = t[0]	
#	myTweet = t[1]
#	myTweet= myTweet.replace('\r\n', ' ')
#	myTweet= myTweet.replace('\n', ' ')
#	sentenceVec = ExtractVectors(myTweet, True)
#	validation_tweets_vectors.append(sentenceVec)
#validation_tweets_vectors=np.array(validation_tweets_vectors)

#Transforming validation labels into an array
#validation_labels_array=[]
#for t in validation_labels.iteritems():
#	myId = t[0]	
#	myLabel = t[1]
#	validation_labels_array.append(myLabel)
#validation_labels_array=np.array(validation_labels_array)

try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV

possible_parameters = {
    'n_neighbors': [3,5,7],
    'weights': ['uniform','distance'],
    'metric': ['euclidean','manhattan']
}


from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = GridSearchCV(KNeighborsClassifier(), possible_parameters, n_jobs=4, cv=3) # n_jobs=4 means we parallelize the search over 4 threads
knn.fit(training_tweets_vectors, training_labels_array)
print("Best parameters: ")
print(knn.best_params_)

#Predict the response for test dataset
predicted_labels = knn.predict(testing_tweets_vectors)

report = metrics.classification_report(testing_labels_array, predicted_labels)

"""
inp = ""     
while inp != "exit":        
	inp = input() 

	myTweets_vectors = []
	myTweet = inp
	mySentenceVec = ExtractVectors(myTweet, True)
	myTweets_vectors.append(mySentenceVec)
	tweetToPredict_vector=np.array(myTweets_vectors)

	myTweetPrediction = clf.predict(tweetToPredict_vector)
	print(myTweet + " is classified as:")
	print(myTweetPrediction)
	print("-------------")
"""


print(report)

print("Accuracy of the model: ",metrics.accuracy_score(testing_labels_array, predicted_labels)) #This is the final accuracy



























