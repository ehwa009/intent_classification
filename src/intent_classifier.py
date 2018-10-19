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
from mind_msgs.msg import EntitiesIndex, Reply, ReplyAnalyzed

class IntentClassifier():

    def __init__(self, model):
        
        self.model = model
        self.model.load()

        rospy.Subscriber('reply', Reply, self.handle_domain_reply)
        self.pub_reply_analyzed = rospy.Publisher('reply_analyzed', ReplyAnalyzed, queue_size=10)
        
        rospy.loginfo("\033[93m[%s]\033[0m initialized." % rospy.get_name())

    def handle_domain_reply(self, msg):
        sent_text = msg.reply
        result = self.model.classify(sent_text, show_details=True)
        # result = result[0][0]

        msg = ReplyAnalyzed()
        # msg.header.stamp = rospy.Time.now()
        # msg.sents.append(sent_text)
        # msg.act_type.append(result + '/%d'%len(sent_text))

        self.pub_reply_analyzed.publish(msg)

if __name__ == '__main__':
    
    rospy.init_node('intent_classifier', anonymous=False)
    model = LSTMIntent()
    
    m = IntentClassifier(model)
    rospy.spin()




    
    