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
# importancy cutoff
CUTOFF = similarity.IMPORTANCY_CUTOFF
# maximum number of keywords for each service
N_KEYWORD = similarity.N_KEYWORD
# verbose?
VERBOSE = True


# ===============================================
# Main
# ===============================================
def main():
    # services
    services = ['Peer Support Center', 'Therapist Assisted Online', 'Group Therapy', 'PhD Support Group', 'Vent Over Tea', '7 Cups',
                'McGill Student\'s Nightline', 'Project 10 Listening Line', 'SACOMSS Support Group', 'The Buddy Programm']
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
        user_profile.keys(), user_profile, review_dataframe, (service_weights, match_weights), verbose=VERBOSE, n_service=N_SERVICE, n_feature=N_FEATURE)

    # similarity between service description & user's goal
    service_keywords = similarity.fetch_data(
        SERVICE_KEYWORD_FOLDER, n_service=N_SERVICE)
    d_score = similarity.calculate_scores(
        user_goal, service_keywords, n_service=N_SERVICE, n_keyword=N_KEYWORD, cut_off=CUTOFF)
    if VERBOSE:
        print("service score: {}\nmatchness score: {}".format(s_score, m_score))
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
