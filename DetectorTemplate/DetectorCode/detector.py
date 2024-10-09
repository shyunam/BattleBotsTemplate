from abc_classes import ADetector
from teams_classes import DetectionMark
import subprocess
import sys
'''
METHOD: Sentiment Analysis
'''

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ["transformers", "torch"]

for package in packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

from transformers import pipeline
import torch

classifier = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis", device=0)
THRESHOLD = 0.95 # Min threshold for sentiment score for bot

class Detector(ADetector):

    def detect_bot(self, session_data):
        marked_account = []
        user_scores = {} # Dictionary to track total sentiment scores per user

        for user in session_data.users:
            user_scores[user['id']] = 0

        # Sentiment analysis by post
        for post in session_data.posts:
            user_id = post['author_id']
            sentiment_result = classifier(post['text'])[0]
            score = sentiment_result['score']

            #print(user_id + ' ' + post['text'] + ' ' + str(score) + ' ' + sentiment_result['label'])

            if user_id in user_scores and sentiment_result['label']!='NEU':
                user_scores[user_id] += score

        # Detect bots
        for user in session_data.users:
            user_id = user['id']
            tweet_count = user['tweet_count']
            total_score = user_scores[user_id]

            average_score = 0
            confidence = 0

            if tweet_count == 0:
                average_score = THRESHOLD
                confidence = 100
            else:
                average_score = total_score/tweet_count
                if average_score >= THRESHOLD: # if bot
                    confidence = average_score
                else: # if not bot
                    confidence = (1-average_score)*100
            
            #print(user_id + ' ' + str(average_score))
            marked_account.append(DetectionMark(user_id=user_id, confidence=int(confidence), bot=(average_score>=THRESHOLD)))

        return marked_account