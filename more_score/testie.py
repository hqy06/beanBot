"""As always, interactive script that explores stuffs for similarity.py"""

# %% import stuffs
from importlib import reload
import similarity
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# %% testing packages
import sys
if 'similarity' in sys.modules:
    print("me!")


# %% explore data
reload(similarity)
files = similarity.list_by_extension(similarity.DATASET_FOLDER)
f_name = files[0]
f_name

dataframes = []
f_path = os.path.join(similarity.DATASET_FOLDER, f_name)
df = pd.read_csv(f_path)
df.columns[1]
dataframes.append(df.drop_duplicates())
data = dataframes[0]
data

type(data.values)
service_keywords = similarity.fetch_data(similarity.DATASET_FOLDER)


# %% vectorization
usr_goal = [['time', 0.5], ['talk', 0.5],
            ['friendly', 0.5], ['advice', 0.5]]
reload(similarity)
vocabulary = list(similarity.create_vocabulary(usr_goal, service_keywords))
similarity.create_dictionary(usr_goal)
similarity.create_single_vector(usr_goal, vocabulary)
matrix = similarity.create_bunch_vector(service_keywords, vocabulary)
matrix.shape
_, u_vec, s_matrix = similarity.vectorize(usr_goal, service_keywords)


# %%
reload(similarity)
cosine_similarity(u_vec.reshape(1, u_vec.shape[0]), s_matrix)
sim = similarity.calculate_scores(usr_goal, service_keywords)
sim.shape
sim.reshape(sim.shape[1], 1)
