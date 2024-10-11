from abc_classes import ADetector
from teams_classes import DetectionMark

import subprocess
import sys
'''
METHOD: Bot cluster detection
'''

'''
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ["scikit-learn", "numpy"]

for package in packages:
    try:
        __import__(package)
    except ImportError:
        install(package)
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.4 # cosine similarity threshold for two posts to be considered simiilar
BOT_THRESHOLD = 0.8 # % similar posts threshold to be considered a bot

class Detector(ADetector):
    def detect_bot(self, session_data):

        posts = [post['text'] for post in session_data.posts]
        user_ids = [post['author_id'] for post in session_data.posts]
        found_similar_post = [False] * len(session_data.posts)

        vect = TfidfVectorizer().fit_transform(posts)
        similarity_matrix = cosine_similarity(vect)

        user_similarities = {user['id']: 0 for user in session_data.users}

        for i in range(len(posts)):
            for j in range(i+1, len(posts)):
                if user_ids[i] != user_ids[j] and similarity_matrix[i][j] >= SIMILARITY_THRESHOLD:
                    #print(f"{user_ids[i]}: {posts[i]}")
                    #print(f"{user_ids[j]}: {posts[j]}")
                    #print(f"Similarity: {similarity_matrix[i][j]}\n")

                    # Count once for each post, even if multiple similar posts are found
                    if not found_similar_post[i]:
                        user_similarities[user_ids[i]] += 1
                        found_similar_post[i] = True
                    if not found_similar_post[j]:
                        user_similarities[user_ids[j]] += 1
                        found_similar_post[j] = True

        marked_account = []

        for user in session_data.users:
            user_id = user['id']
            tweet_count = user['tweet_count']
            similarity_score = user_similarities[user_id]

            if tweet_count == 0:
                average_similarity = 1
            else:
                average_similarity = similarity_score / tweet_count
            
            #print(f"{user_id}: avg similarity {average_similarity}")
            
            is_bot = average_similarity >= BOT_THRESHOLD

            if is_bot:
                    confidence = int(average_similarity*100)
            else:
                confidence = int((1-average_similarity)*100)
            
            marked_account.append(DetectionMark(user_id=user_id, confidence=confidence, bot=is_bot))

        return marked_account
    