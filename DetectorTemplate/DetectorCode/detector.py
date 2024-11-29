from abc_classes import ADetector
from teams_classes import DetectionMark
from transformers import pipeline
import torch
import enchant
import re

'''
METHOD: Sentiment Analysis
'''

classifier = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
THRESHOLD = 0.55 # Min threshold for sentiment score for bot
SPELLING_THRESHOLD = 0.01 # Min threshold for percentage of misspelled words

class Detector(ADetector):

    def detect_bot(self, session_data):
        marked_account = []
        user_scores = {} # Dictionary to track total sentiment scores per user
        tweet_counts = {} 

        d = enchant.Dict("en_US")
        user_spelling_scores = {} # Dictionary to track number of misspelled and total word count
        lowercase = re.compile(r'^[a-z]+$')

        for user in session_data.users:
            user_scores[user['id']] = 0
            tweet_counts[user['id']] = 0
            user_spelling_scores[user['id']] = {'misspelled': 0, 'total': 0}

        # Sentiment analysis by post
        for post in session_data.posts:
            user_id = post['author_id']
            sentiment_result = classifier(post['text'], truncation=True)[0]
            score = sentiment_result['score']

            #print(user_id + ' ' + post['text'] + ' ' + str(score) + ' ' + sentiment_result['label'])
            tweet_counts[user_id] += 1

            if user_id in user_scores and sentiment_result['label']!='NEU':
                user_scores[user_id] += score

            # Spelling error
            word_list = post['text'].split()
            for word in word_list:
                if lowercase.match(word):
                    user_spelling_scores[user_id]['total'] += 1
                    if not d.check(word):
                        user_spelling_scores[user_id]['misspelled'] += 1

            # If the first word of the post is not capitalized or is not an alphabet, consider it a misspell
            if post['text']:
                if not post['text'][0].isalpha() or post['text'][0].islower():
                    user_spelling_scores[user_id]['total'] += 1
                    user_spelling_scores[user_id]['misspelled'] += 1

        # Detect bots
        for user in session_data.users:
            user_id = user['id']
            tweet_count = tweet_counts[user_id]
            total_score = user_scores[user_id]
            z_score = user['z_score']

            misspelled = user_spelling_scores[user_id]['misspelled']
            total_words = user_spelling_scores[user_id]['total']
            misspelled_percentage = (misspelled / total_words) if total_words > 0 else 1

            average_sentiment_score = 0
            confidence = 0
            is_bot = False

            if tweet_count == 0:
                average_sentiment_score = THRESHOLD
                confidence = 100
                is_bot = True
            else:
                average_sentiment_score = total_score/tweet_count
                # is bot
                if average_sentiment_score >= THRESHOLD:
                    confidence = average_sentiment_score*100
                    is_bot = True

                    # Exclude users with certain num. spelling errors
                    if misspelled_percentage > SPELLING_THRESHOLD:
                        confidence = (1-average_sentiment_score)*100
                        is_bot = False
                else: # not bot
                    confidence = (1-average_sentiment_score)*100
            
            # Case where tweet count attribute is wrong
            if confidence > 100:
                confidence = 100
                is_bot = True
            
            if z_score == 0:
                is_bot = True
            
            #print(user_id + ' ' + str(average_score))
            marked_account.append(DetectionMark(user_id=user_id, confidence=int(confidence), bot=is_bot))

        return marked_account
