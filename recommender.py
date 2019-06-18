"""
This is a wrapper for the recommendation.py
"""

# ===============================================
# import stuffs
# ===============================================
import numpy as np
import pandas as pd
from more_score import similarity
from review_score import evaluation
import recommendation

# ===============================================
# Preset stuffs for testing
# ===============================================
N_SERVICE = 11
N_FEATURE = 4
TOP_K = 3
CUTOFF = 0.15
SERVICE_KEYWORD_FOLDER = "more_score\\keywords"
REVIEW_DATABASE = "review_score\\dummy.csv"
N_KEYWORD = 7
SERVICES = ['Peer Counseling Services',
            'Office for Students with Disabilities',
            'Therapist Assisted Online',
            'Group Therapy',
            'PhD Support Group',
            'Vent Over Tea',
            '7 Cups',
            'McGill Student\'s Nightline',
            'Project 10 Listenting Line',
            'SACOMSS Support Groups',
            'The Buddy Program']


# ===============================================
# The recommender class
# ===============================================

class recommender:
    def __init__(self, n_service=N_SERVICE, service_list=SERVICES, n_fixed_feature=N_FEATURE, top_k=TOP_K, cut_off=CUTOFF, review_file=REVIEW_DATABASE, service_kw_folder=SERVICE_KEYWORD_FOLDER, n_keyword=N_KEYWORD, learning_rate=0.0015):
        self.n_service = n_service
        self.n_feature = n_fixed_feature
        self.top_k = top_k
        self.cut_off = cut_off
        self.services = service_list
        self.review_dataframe = pd.read_csv(review_file)
        self.service_keywords = similarity.fetch_data(
            service_kw_folder, n_service=n_service)
        self.n_keyword = n_keyword
        self.match_weights = None
        self.service_weights = None
        self.lr = learning_rate
        assert (len(service_list) == n_service), "wrong number of services in the service list, expecting {}, get {} instead".format(
            n_service, len(service_list))
        assert (isinstance(top_k, int) and top_k >
                0), "top_k should be an integer between 1 and n_service"

    def init_weights(self):
        self.service_weights, self.match_weights = recommendation.init_weights(
            self.n_feature)
        return self.service_weights, self.match_weights

    def update_weights(self, user_profile, user_scores, choices, verbose=False):
        assert (self.match_weights is not None), "please call init_weights first!"
        self.service_weights, self.match_weights = evaluation.update_weights(
            (self.service_weights, self.match_weights), user_profile, user_scores, choices, self.review_dataframe, lr=self.lr, verbose=verbose, n_service=self.n_service, n_feature=self.n_feature)
        return self.service_weights, self.match_weights

    def get_recommendation(self, user_profile, user_goal, verbose=False):
        assert (self.match_weights is not None), "please call init_weights first!"

        assert (len(user_profile) == self.n_feature), "wrong number of user features, expecting {}, get {} instead.".format(
            self.n_feature, len(user_profile))

        # service & matchness scores from review dataset
        s_score, m_score, _, _ = evaluation.calculate_scores(
            user_profile.keys(), user_profile, self.review_dataframe, (self.service_weights, self.match_weights), verbose=verbose, n_service=self.n_service, n_feature=self.n_feature)
        if verbose:
            print("\nservice score: {}\nmatchness score: {}".format(s_score, m_score))

        # similarity between service description and user's goal
        d_score = similarity.calculate_scores(
            user_goal, self.service_keywords, n_service=self.n_service, n_keyword=self.n_keyword, cut_off=self.cut_off)
        if verbose:
            print("\ndescription score: {}".format(d_score))

        # calculate total score
        total_scores = np.array(s_score) + np.array(m_score) + d_score
        if verbose:
            print("\nWith 1:1:1 weight for service score, matchness score and goal/description similairty, the total weight is {}".format(total_scores))

        # pick the top-k services
        indices = np.argsort(total_scores)
        if verbose:
            recommend = []
            for i in range(self.top_k):
                index = indices[i]
                recommend.append(
                    (self.services[index], total_scores[index]))
            print("\nrecommendation: {}".format(recommend))

        # return the indices of top_k choices.
        return indices[:self.top_k]

    def process_user_rating(self, user_rating):
        return evaluation.process_user_rating(user_rating, k=self.top_k)


# ===============================================
# Main
# ===============================================


def main():
    # because I don't have the keywords for'Office for Students with Disabilities'
    dummy_services = ['Peer Support Center', 'Therapist Assisted Online', 'Group Therapy', 'PhD Support Group', 'Vent Over Tea', '7 Cups',
                      'McGill Student\'s Nightline', 'Project 10 Listening Line', 'SACOMSS Support Group', 'The Buddy Programm']
    # shall be the review csv file, use the dummy one I created for testing
    review_database = REVIEW_DATABASE
    # points to the folder containing the keywords for each service
    service_folder = SERVICE_KEYWORD_FOLDER

    # init the recommender
    rr = recommender(n_service=10, service_list=dummy_services, n_fixed_feature=4, top_k=TOP_K, cut_off=CUTOFF,
                     review_file=review_database, service_kw_folder=service_folder, n_keyword=7, learning_rate=0.0015)
    # randomly init weights!
    rr.init_weights()

    # dummy inupt from user: fixed feature, keywords of goal and rating for service & matchness
    user_profile = {"ufeature1": "F", "ufeature2": "fr",
                    "ufeature3": "U3", "ufeaure4": "CA"}
    user_goal = [['time', 0.5], ['talk', 0.5],
                 ['friendly', 0.5], ['advice', 0.5]]
    user_rating = {'Group Therapy': (3, 5),
                   'Vent Over Tea': (5, 5),
                   '7 Cups': (4, 5)}

    # get the service index of top_k choices, start from 0
    choices = rr.get_recommendation(user_profile, user_goal, verbose=True)

    # update the weights (ML-ish learning)
    user_scores = rr.process_user_rating(user_rating)
    rr.update_weights(user_profile, user_scores, choices, verbose=True)


if __name__ == "__main__":
    main()
