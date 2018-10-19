#!/usr/bin/env python
#-*- encoding: utf8 -*-

import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import rospkg
import rospy
import re
import numpy as np
import time
stemmer = LancasterStemmer()

from LSTM_model import LSTMIntent


def main():
    utterance_file = os.path.join(rospkg.RosPack().get_path('intent_classifier'), 'data', 'utterance.txt')
    
    words=[]
    classes=[]
    documents=[]
    ignore_words=['?']

    with open(utterance_file) as f:
        for line in f:
            line_data = line.split('/')
            sentence_filtered = re.sub("[\'.,#!?:-]", '', line_data[1]).strip()

            w = nltk.word_tokenize(sentence_filtered)
            words.extend(w)
            documents.append((w, line_data[0]))

            if line_data[0] not in classes:
                classes.append(line_data[0])
    
    #stem and lower each word and remove duplicate
    words=[stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words=list(set(words))

    #remove duplicates
    classes=list(set(classes))

    # print(len(documents)," documents")
    print(documents)
    # print(len(classes), " classes", classes)
    # print(len(words)," unique stemmed words", words)

    training = []
    output = []

    output_empty = [0]*len(classes)

    #training set, bag of words for each sentence
    for doc in documents:
        #initialize our bag of words
        bag=[]
        #list of tokenized words for the pattern
        pattern_words=doc[0]
        #stem each word
        pattern_words=[stemmer.stem(word.lower()) for word in pattern_words]
        #create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        training.append(bag)
        #output is a 0 for each tag and 1 for current tag
        output_row=list(output_empty)
        output_row[classes.index(doc[1])] = 1
        output.append(output_row)

    return output, training, words, classes

if __name__ == '__main__':
    output, training, words, classes = main()
    l = LSTMIntent(output, training, words, classes)
    l.train_start()
