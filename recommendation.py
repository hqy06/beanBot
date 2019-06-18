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
CUTOFF = 0.15
# maximum number of keywords for each service
N_KEYWORD = 6
# verbose?
VERBOSE = False


# ===============================================
# Updating the weight: a learning model
# ===============================================
def init_weights(n_feature=N_FEATURE):
    """randomly init weights for feature scores from review database"""
    return np.random.random_sample(n_feature), np.random.random_sample(n_feature)


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
    service_weights, match_weights = init_weights(N_FEATURE)

    s_score, m_score, _, _ = evaluation.calculate_scores(
        user_profile.keys(), user_profile, review_dataframe, (service_weights, match_weights), verbose=VERBOSE, n_service=N_SERVICE, n_feature=N_FEATURE)

    # similarity between service description & user's goal
    service_keywords = similarity.fetch_data(
        SERVICE_KEYWORD_FOLDER, n_service=N_SERVICE)
    d_score = similarity.calculate_scores(
        user_goal, service_keywords, n_service=N_SERVICE, n_keyword=N_KEYWORD, cut_off=CUTOFF)
    if VERBOSE:
        print("service score: {}\nmatchness score: {}".format(s_score, m_score))
        print("description score: {}".format(d_score))

    # total score: can do other fancy stuffs with it.
    total_scores = np.array(s_score) + np.array(m_score) + d_score
    print("\ntotal scores:\n\t{}".format(total_scores))

    # find the top-k services
    indices = np.argsort(total_scores)
    recommend = []
    for i in range(TOP_K):
        index = indices[i]
        recommend.append((services[index], total_scores[index]))

    print("\nrecommendation:\n\t{}".format(recommend))

    # print("\n {} of type{}".format(indices, type(indices)))

    # update_weights
    user_rating = {'Group Therapy': (3, 5),
                   'Vent Over Tea': (5, 5),
                   '7 Cups': (4, 5)}
    user_scores = evaluation.process_user_rating(user_rating, k=TOP_K)

    print("\nOld weights:\n\t{}\n\t{}".format(service_weights, match_weights))
    service_weights, match_weights = evaluation.update_weights(
        (service_weights, match_weights), user_profile, user_scores, indices[:TOP_K], review_dataframe, lr=0.01, verbose=False, n_service=N_SERVICE, n_feature=N_FEATURE)
    print("Given feedback\n\t{}\nupdate to new weights:\n\t{}\n\t{}".format(
        user_rating, service_weights, match_weights))


if __name__ == "__main__":
    main()
