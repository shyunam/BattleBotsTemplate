from abc_classes import ADetector
from teams_classes import DetectionMark
import enchant
import re

'''
METHOD: Grammar accuracy check
'''

THRESHOLD = 0.01 # Min threshold for percentage of misspelled words 

class Detector(ADetector):

    def detect_bot(self, session_data):
        d = enchant.Dict("en_US")
        marked_account = []
        user_scores = {} # Dictionary to track number of misspelled and total word count
        misspelled_words_by_bot = {} # for testing

        # Check only lowercase alphabetic words
        lowercase = re.compile(r'^[a-z]+$')

        for user in session_data.users:
            user_scores[user['id']] = {'misspelled': 0, 'total': 0}
            misspelled_words_by_bot[user['id']] = []
        
        # Count misspelled words
        for post in session_data.posts:
            user_id = post['author_id']
            word_list = post['text'].split()

            for word in word_list:
                if lowercase.match(word):
                    user_scores[user_id]['total'] += 1
                    if not d.check(word):
                        user_scores[user_id]['misspelled'] += 1
                        misspelled_words_by_bot[user_id].append(word) # for testing

            # If the first word of the post is not capitalized or is not an alphabet, consider it a misspell
            if post['text']:
                if not post['text'][0].isalpha() or post['text'][0].islower():
                    user_scores[user_id]['total'] += 1
                    user_scores[user_id]['misspelled'] += 1

        # Detect bots
        for user in session_data.users:
            user_id = user['id']
            misspelled = user_scores[user_id]['misspelled']
            total_words = user_scores[user_id]['total']
            z_score = user['z_score']

            misspelled_percentage = (misspelled / total_words) if total_words > 0 else 1
            
            is_bot = misspelled_percentage <= THRESHOLD
            
            if is_bot:
                confidence = int((1-misspelled_percentage) * 100)
            else:
                if misspelled_percentage >= 0.1:
                    confidence = 100
                else:
                    confidence = 50 + misspelled_percentage*100

            
            if z_score == 0:
                is_bot = True
            
            marked_account.append(DetectionMark(user_id=user_id, confidence=int(confidence), bot=is_bot))


            # for testing
            '''
            if '-' in user_id:
                print(f"User ID: {user_id} Words: {misspelled_words_by_bot[user_id]}")
            '''

        return marked_account
