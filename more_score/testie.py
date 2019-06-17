"""As always, interactive script that explores stuffs for similarity.py"""

# %% import stuffs
from importlib import reload
import similarity
import pandas as pd
import numpy as np
import os

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

similarity.fetch_data(similarity.DATASET_FOLDER)
