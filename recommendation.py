"""
Calculate scores based on:
1. user's goal + services' keywords
2. user's feature + past review dataset

Returns the top-k choices
"""
# ===============================================
# Import Libraries and Modules
# ===============================================
import numpy as np
import pandas as pd
from more_score import similarity
from review_score import evaluation
from sa_demo import SADemo

from importlib import reload
reload(evaluation)

# ===============================================
# Global Variables
# ===============================================
# review database csv
REVIEW_DATABASE = "review_score\\dummy.csv"
# folder containing extracted keywords, also csv format
SERVICE_KEYWORD_FOLDER = "more_score\\keywords"
# number of services
N_SERVICE = 10
# number of deterministic user features, e.g. language, gender, etc
N_FEATURE = 4
# top k choices!
TOP_K = 3


# ===============================================
# Main
# ===============================================
def main():
    # services
    services = ['PSC', 'TAO', 'Group', 'PhD', 'Tea', '7Cups',
                'Nightline', 'listening', 'SACOMSS', 'Buddy']
    assert (len(services) == N_SERVICE)

    # inupt from user's end:
    user_profile = {"ufeature1": "F", "ufeature2": "fr",
                    "ufeature3": "U3", "ufeaure4": "CA"}
    user_goal = [['time', 0.5], ['talk', 0.5],
                 ['friendly', 0.5], ['advice', 0.5]]
    assert (len(user_goal) == N_FEATURE), "wrong number of user features, expecting {}, get {} instead.".format(
        N_FEATURE, len(user_goal))

    # review dataset
    review_dataframe = pd.read_csv(REVIEW_DATABASE)
    # random weights for service and matchness
    service_weights = np.random.random_sample(N_FEATURE)
    match_weights = np.random.random_sample(N_FEATURE)

    s_score, m_score = evaluation.calculate_scores(
        user_profile.keys(), user_profile, review_dataframe, (service_weights, match_weights))

    print("service score: {}\nmatchness score: {}".format(s_score, m_score))

    # similarity between service description & user's goal
    service_keywords = similarity.fetch_data(SERVICE_KEYWORD_FOLDER)
    d_score = similarity.calculate_scores(user_goal, service_keywords)
    print("description score: {}".format(d_score))

    # total score!
    total_scores = np.array(s_score) + np.array(m_score) + d_score
    print(total_scores)

    # find the top-k services
    indices = np.argsort(total_scores)
    recommend = []
    for i in range(TOP_K):
        index = indices[i]
        recommend.append((services[index], total_scores[index]))
    print(recommend)


if __name__ == "__main__":
    main()
