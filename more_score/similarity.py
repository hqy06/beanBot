"""
Calculate scores for services based on the similarity between service description and user's goal.
"""
# ===============================================
# Import Libraries
# ===============================================
import numpy as np
import pandas as pd


# ===============================================
# Global Variables
# ===============================================
DATASET = "service_keywords.csv"
N_FEATURE = 4               # number of user featues
N_SERVICE = 10              # number of services
N_KEYWORD = 10              # maximum number of keywords for each service
IMPORTANCY_CUTOFF = 0.01    # minimum "importancy" allowed for keywords.


# ===============================================
# Methods
# ===============================================

def calculate_scores(user_goal, service_keyword_df):
    # vectorization
    user_vector, service_vectors = vectorize(
        user_goal, service_keyword_df.values)

    # cosine similarity
    return


def _vectorizer(user_goal, weighted_keywords):
    return


# ===============================================
# Main
# ===============================================


def main():
    service_keywords = pd.read_csv(DATASET)
    usr_goal = []

    d_scores = calculate_scores(usr_goal, service_keywords)

    print(d_scores)


if __name__ == "__main__":
    main()
