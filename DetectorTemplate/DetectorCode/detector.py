from abc_classes import ADetector
from teams_classes import DetectionMark
from transformers import pipeline
import torch

'''
METHOD: Sentiment Analysis
'''

classifier = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
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
            sentiment_result = classifier(post['text'], truncation=True)[0]
            score = sentiment_result['score']

            #print(user_id + ' ' + post['text'] + ' ' + str(score) + ' ' + sentiment_result['label'])

            if user_id in user_scores and sentiment_result['label']!='NEU':
                user_scores[user_id] += score

        # Detect bots
        for user in session_data.users:
            user_id = user['id']
            tweet_count = user['tweet_count']
            total_score = user_scores[user_id]
            z_score = user['z_score']

            average_score = 0
            confidence = 0
            is_bot = False

            if tweet_count == 0:
                average_score = THRESHOLD
                confidence = 100
                is_bot = True
            else:
                average_score = total_score/tweet_count
                if average_score >= THRESHOLD:
                    confidence = average_score*100
                    is_bot = True
                else: 
                    confidence = (1-average_score)*100
            
            # Case where tweet count attribute is wrong
            if confidence > 100:
                confidence = 100
                is_bot = True
            
            if z_score == 0:
                is_bot = True
            
            #print(user_id + ' ' + str(average_score))
            marked_account.append(DetectionMark(user_id=user_id, confidence=int(confidence), bot=is_bot))

        return marked_account
