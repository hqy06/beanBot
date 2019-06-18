"""
Calculate scores for services based on the similarity between service description and user's goal.
"""
# ===============================================
# Import Libraries
# ===============================================
import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics.pairwise import cosine_similarity


# ===============================================
# Global Variables
# ===============================================
DATASET_FOLDER = 'keywords'
# N_FEATURE = 4               # number of user featues
N_SERVICE = 10              # number of services
N_KEYWORD = 7               # maximum number of keywords for each service
IMPORTANCY_CUTOFF = 0.12    # minimum "importancy" allowed for keywords.

# ===============================================
# Load and clean the data
# ===============================================


def fetch_data(folder=DATASET_FOLDER, n_service=N_SERVICE):
    files = list_by_extension(folder)
    assert (n_service == len(files)), "wrong number of keyword csv files detected, expecting {}, get{}".format(
        n_service, len(files))
    cleaned_dfs = _read_and_clean_csv(files, folder)
    service_keywords = []
    for df in cleaned_dfs:
        s_keyword = df.values
        service_keywords.append(s_keyword)
    return service_keywords


def list_by_extension(path, extension=r".*(\.csv)"):
    """retrieve files with given extension in the given dir
    -------
    Copy and paste from vanillaRNN.py
    """
    dirlist = os.listdir(path)
    pattern = re.compile(extension)
    filtered = filter(pattern.match, dirlist)
    files = list(filtered)
    return files


def _read_and_clean_csv(file_names, folder_path):
    dataframes = []
    for f_name in file_names:
        f_path = os.path.join(folder_path, f_name)
        df = pd.read_csv(f_path)                 # read csv file
        df = df.drop_duplicates()                # remove duplicate entries
        # sort by "importancy"
        df = df.sort_values(by=(df.columns[1]), ascending=False)
        dataframes.append(df)
    return dataframes


# ===============================================
# Calculate Score
# ===============================================


def calculate_scores(user_goal, service_keywords, n_service=N_SERVICE, n_keyword=N_KEYWORD, cut_off=IMPORTANCY_CUTOFF):
    # vectorization
    vocab, user_vector, service_vectors = vectorize(
        user_goal, service_keywords, n_service, n_keyword, cut_off)

    # cosine similarity
    similarity = cosine_similarity(user_vector.reshape(
        1, user_vector.shape[0]), service_vectors)

    return similarity.reshape(similarity.shape[1])


def vectorize(user_goal, weighted_keywords, n_service=N_SERVICE, n_keyword=N_KEYWORD, cut_off=IMPORTANCY_CUTOFF):
    """user_goal is a 2-nested list, weighted_keywords a 3-nested list.
    user_vector of shape (n_vocab,); service_vectors of shape(N_SERVICE, n_vocab)
    """
    # vectorize keywords and create dictionary
    vocabulary_set = create_vocabulary(
        user_goal, weighted_keywords, n_keyword, cut_off)
    vocabulary = list(vocabulary_set)
    user_vector = create_single_vector(user_goal, vocabulary)
    service_vectors = create_bunch_vector(weighted_keywords, vocabulary)
    assert (user_vector.shape[0] == service_vectors.shape[1]
            ), "number of vocabulary should be equal!"
    assert (service_vectors.shape[0] == N_SERVICE)
    return vocabulary, user_vector, service_vectors


def create_vocabulary(user_goal, weighted_keywords, n_keyword=N_KEYWORD, cut_off=IMPORTANCY_CUTOFF):
    """findout the vocabulary given the maximum number of keyword and the minimum cutoff"""
    dict_set = set()
    for word, coeff in user_goal:
        dict_set.add(word)
    for s_keywords in weighted_keywords:
        for word, coeff in s_keywords:
            counter = 0
            if coeff > cut_off:
                dict_set.add(word)
                counter += 1
            if counter == n_keyword:
                break
    return dict_set


def create_single_vector(nested_list, vocabulary):
    word_num_dict = _to_dictionary(nested_list)
    vect = []
    for word in vocabulary:
        if word in word_num_dict:
            vect.append(word_num_dict[word])
        else:
            vect.append(0)
    return np.array(vect)


def create_bunch_vector(weighted_keywords, vocabulary):
    results = []
    for nested_list in weighted_keywords:
        vector = create_single_vector(nested_list, vocabulary)
        results.append(vector)
    return np.array(results)


def _to_dictionary(nested_list):
    dict = {}
    for word, coeff in nested_list:
        dict[word] = coeff
    return dict

# ===============================================
# Main
# ===============================================


def main():
    service_keywords = fetch_data(folder=DATASET_FOLDER)
    usr_goal = [['time', 0.5], ['talk', 0.5],
                ['friendly', 0.5], ['advice', 0.5]]
    # assert (len(usr_goal) == N_FEATURE), "wrong number of user features, expecting {}, get {} instead.".format(
    #     N_FEATURE, len(usr_goal))

    d_scores = calculate_scores(usr_goal, service_keywords)

    print(d_scores)


if __name__ == "__main__":
    main()
