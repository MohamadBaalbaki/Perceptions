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

def CleanStopWords (sentence):
		stop_words = stopwords.words('arabic')  
		sentenceSplitted = sentence.split(" ")  
		sentence = [w for w in sentenceSplitted if w not in stop_words]
		return(sentence)


def ExtractVectors(sentence, removeStopwords):
		sentence = sentence.rstrip()
		if (removeStopwords):
			sentence = CleanStopWords(sentence)
			sentence = ' '.join(sentence)
		sv = model_ft.get_sentence_vector(sentence)
		#vector = sv.reshape(1, -1)
		return(sv) #it was return vector


print("Loading FT model")
model_ft  = fasttext.load_model('/home/baalbaki/Desktop/FastText/cc.ar.300.bin')

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

dataset = pd.read_csv(r'dataset2.csv')
X, y = dataset.tweet, dataset.perception #X now holds the tweets and y holds the labels

##############
training_tweets, testing_tweets, training_labels, testing_labels = train_test_split(X, y, test_size=0.2, random_state=42)


testing_tweets, validation_tweets, testing_labels, validation_labels = train_test_split(testing_tweets, testing_labels, test_size=0.5, random_state=42)
##############

print(len(training_tweets))
print(len(testing_tweets))
print(len(validation_tweets))

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
validation_tweets_vectors=[]
for t in validation_tweets.iteritems():
	myId = t[0]	
	myTweet = t[1]
	myTweet= myTweet.replace('\r\n', ' ')
	myTweet= myTweet.replace('\n', ' ')
	sentenceVec = ExtractVectors(myTweet, True)
	validation_tweets_vectors.append(sentenceVec)
validation_tweets_vectors=np.array(validation_tweets_vectors)

#Transforming validation labels into an array
validation_labels_array=[]
for t in validation_labels.iteritems():
	myId = t[0]	
	myLabel = t[1]
	validation_labels_array.append(myLabel)
validation_labels_array=np.array(validation_labels_array)

try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV
possible_parameters = {
    'C': [1e0, 1e1, 1e2, 1e3],
    'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
    'kernel': ['linear', 'rbf','poly']

}

#svc = SVC(kernel='rbf')

#Run gridsearch on validation, and once you find the best configuration run it on the test set
# The GridSearchCV is itself a classifier
# we fit the GridSearchCV with the training data
# and then we use it to predict on the test set
clf = GridSearchCV(SVC(), possible_parameters, n_jobs=4, cv=3) # n_jobs=4 means we parallelize the search over 4 threads
clf.fit(validation_tweets_vectors, validation_labels_array)

predicted_labels = clf.predict(validation_tweets_vectors)

report = metrics.classification_report(validation_labels_array, predicted_labels)

print(report)

print("Accuracy of the model after grid search *VALIDATION: ",metrics.accuracy_score(validation_labels_array, predicted_labels)) #This is the final accuracy

#THIS IS THE CORRECT WAY TO DO IT
predicted_labels = clf.predict(testing_tweets_vectors)

report = metrics.classification_report(testing_labels_array, predicted_labels)

print(report)

print("Accuracy of the model after grid search *TESTING: ",metrics.accuracy_score(testing_labels_array, predicted_labels)) #This is the final accuracy



























