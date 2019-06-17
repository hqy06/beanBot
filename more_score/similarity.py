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


# ===============================================
# Global Variables
# ===============================================
DATASET_FOLDER = 'keywords'
N_FEATURE = 4               # number of user featues
N_SERVICE = 10              # number of services
N_KEYWORD = 10              # maximum number of keywords for each service
IMPORTANCY_CUTOFF = 0.01    # minimum "importancy" allowed for keywords.

# ===============================================
# Load and clean the data
# ===============================================


def fetch_data(folder=DATASET_FOLDER):
    files = list_by_extension(folder)
    assert (N_SERVICE == len(files)), "wrong number of keyword csv files detected, expecting {}, get{}".format(
        N_SERVICE, len(files))
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
