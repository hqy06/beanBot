"""
A demo for our recommendation system!
"""

# ===============================================
# import the wrapper
# ===============================================
import recommender
import pandas as pd
import numpy as np

# ===============================================
# Preset stuffs for testing
# ===============================================
N_SERVICE = 11
N_FEATURE = 10
TOP_K = 3
CUTOFF = 0.15
SERVICE_KEYWORD_FOLDER = "dataset\\keywords"
REVIEW_DATABASE = "dataset\\c_review.csv"
N_KEYWORD = 7
SERVICES = ['Peer Counseling Services',
            'Therapist Assisted Online',
            'Group Therapy',
            'PhD Support Group',
            'Vent Over Tea',
            '7 Cups',
            'McGill Student\'s Nightline',
            'Project 10 Listenting Line',
            'SACOMSS Support Groups',
            'The Buddy Program',
            'McGill Counselling and Psychiatric Services']
U_FEATURES = ['faculty',
              'campus',
              'gender',
              'international',
              'language',
              'cultural_bg',
              'timeCommit',
              'urgency',
              'professional',
              'medium']

# ===============================================
# Main
# ===============================================


def main():
    print("This is the demo for the recommender!")

    # because I don't have the keywords for'Office for Students with Disabilities'
    dummy_services = SERVICES
    # shall be the review csv file, use the dummy one I created for testing
    review_database = REVIEW_DATABASE
    # points to the folder containing the keywords for each service
    service_folder = SERVICE_KEYWORD_FOLDER

    print("Services are: {}".format(dummy_services))

    # init the recommender
    rr = recommender.recommender(n_service=N_SERVICE, service_list=dummy_services, n_fixed_feature=10, top_k=TOP_K, cut_off=CUTOFF,
                                 review_file=review_database, service_kw_folder=service_folder, n_keyword=7, learning_rate=0.0015)
    # randomly init weights!
    rr.init_weights()

    print("recommender model inited!")

    while(True):
        print("Key in user profile:\n")
        user_profile = dict()
        for f in U_FEATURES:
            user_profile[f] = input("\t{}: ".format(f))

        print(user_profile)

        print("Preset user's goal: I want spend time talking iwth someone. I need advice.")
        user_goal = [['time', 0.5], ['talk', 0.5],
                     ['friendly', 0.5], ['advice', 0.5]]
        user_rating = {'Group Therapy': (3, 5),
                       'Vent Over Tea': (5, 5),
                       '7 Cups': (4, 5)}
        print("Preset user's rating: user_rating")

        # get the service index of top_k choices, start from 0
        choices, service_names = rr.get_recommendation(
            user_profile, user_goal, verbose=False)
        # print("returned indices (start from 0): {}".format(choices))
        choices_for_chatbot = [c + 1 for c in choices]
        print("service indicies (start from 1): {}".format(choices_for_chatbot))
        print("returned names:{}".format(service_names))

        # update the weights (ML-ish learning)
        user_scores = rr.process_user_rating(user_rating)
        rr.update_weights(user_profile, user_scores, choices, verbose=False)


if __name__ == "__main__":
    main()
