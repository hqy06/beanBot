"""
sentimental analysis demo using SVM
------
Reference:
https://medium.com/@vasista/sentiment-analysis-using-svm-338d418e3ff1
"""

# ===============================================
# Import Libraries
# ===============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from datetime import datetime
import collections
import re           # regex
import nltk         # working with NLP
from nltk.corpus import stopwords   # english stopwords
from sklearn.cross_validation import train_test_split

# ===============================================
# Global Variables
# ===============================================
DEBUG = True
VERBOSE = True
DATASET = "Tweets.csv"
CONFIDENCE_CUTOFF = 0.5

# ===============================================
# Preprocess Data
# ===============================================


def tri_connotation(x):
    """quantifying sentiment: 0 for neutral, -1 for negative, 1 for positive"""
    if x == 'negative':
        return -1
    elif x == 'positive':
        return 1
    else:
        return 0


def bin_connotation(x):
    """quantify sentiment: 0 for negative, 1 for neutral and positive"""
    if x == 'positive' or x == 'neutral':
        return 1
    else:
        return 0


def tweet_word_check(raw_tweet):
    tweet = re.sub("[^a-zA-Z]", " ", raw_tweet)  # letter lonly
    tokens = tweet.lower().split()
    stoppers = stopwords.words("english")
    words = [t for t in tokens if not t in set(stoppers)]
    clean_tweet = " ".join(words)
    return clean_tweet


def tweet_length_clean(raw_tweet):
    pass

# ===============================================
# Visualize Data
# ===============================================


def _show_column_count(df_col, show_fig=True, save_fig=True):
    """Takes in one column of data_frame (should be a Seriese object) and output a bar chart that show the distribution"""
    # sanity check
    assert (type(df_col) is pd.core.series.Series)

    # generate names
    name = df_col.name
    f_name = "{} {}.png".format(
        datetime.now().strftime('%Y%m%d-%H%M'), name)

    # deal with NaN in 'negativereason' column
    if name == 'negativereason':
        for i in range(len(df_col)):
            if type(df_col[i]) is float:
                df_col.at[i] = "NA"

    # generate data for plotting
    counter = collections.Counter(df_col)
    keys = list(counter.keys())
    values = list(counter.values())
    # c = sorted(counter.items())
    # keys = [item[0] for item in c]
    # values = [item[1] for item in c]

    if VERBOSE:
        print("\n****** {} info".format(name))
        for key, value in zip(keys, values):
            print("{:>30} {:5}".format(key, value))

    # Prepare the canvas
    fig = plt.figure()
    ax = plt.gca()

    # Plot the bar graph and set title
    ax.bar(keys, values, color="plum", ec="orchid", lw=1)
    ax.set_title("review {} count".format(name))

    # Tightening convention
    plt.xticks(rotation=90)
    fig.tight_layout()

    # Save/Display Rountine
    if save_fig:
        plt.savefig(f_name)
    if show_fig:
        plt.show(block=False)
        plt.pause(3)
    plt.close()

    return df_col.unique()


def _show_column_count_by_class(df, count_name, class_name, show_fig=True, save_fig=True):
    f_name = "{} {} by {}.png".format(
        datetime.now().strftime('%Y%m%d-%H%M'), count_name, class_name)
    classes = df[class_name].unique()
    n_col = int(np.ceil(len(classes) ** 0.5))
    n_row = int(np.ceil(len(classes) / float(n_col)))

    # Setup canvas
    fig, axes = plt.subplots(n_row, n_col, figsize=(12, 12))

    # Flatten axes
    axis = axes.ravel()
    # Remove extra axes
    for i in range(len(classes), n_col * n_row):
        fig.delaxes(axes[i])

    # Draw on axis
    for cla, ax in zip(classes, axis):
        if VERBOSE:
            print("\t *** {} of class {}".format(count_name, cla))

        class_df = df[df[class_name] == cla]
        class_col = class_df[count_name]
        title = "{} count\n of {}".format(count_name, cla)
        _plot_count_bar_on_ax(ax, class_col, title)

    # Tightening convention
    fig.tight_layout()

    # Save/Display Rountine
    if save_fig:
        plt.savefig(f_name)
    if show_fig:
        plt.show(block=False)
        plt.pause(3)
    plt.close()

    return None


def _plot_count_bar_on_ax(axis, df_col, title):
    # generate data for plotting
    counter = collections.Counter(df_col)
    c = sorted(counter.items())
    keys = [item[0] for item in c]
    values = [item[1] for item in c]

    if VERBOSE:
        for key, value in zip(keys, values):
            print("{:>20} {:5}".format(key, value))

    # Plot the bar graph and set title
    axis.barh(keys, values, color="plum", ec="orchid", lw=1)
    axis.set_title(title, fontsize=9)

    return None

# ===============================================
# Main
# ===============================================


def main():
    data_frame = pd.read_csv(DATASET)
    if VERBOSE:
        print("\n********** {} dataset info".format(DATASET))
        print(data_frame.info())
        print("\n********** Visualize dataset")

    airlines = _show_column_count(data_frame['airline'])
    sentiments = _show_column_count(data_frame['airline_sentiment'])
    _show_column_count_by_class(data_frame, "airline_sentiment", "airline")

    neg_reasons = _show_column_count(data_frame['negativereason'])
    # neg reason for a specific airline
    # df = data_frame[data_frame['airline']==airline_name]
    # airline_neg = _show_column_count(df['negativereason'])
    neg_reason_confidence = _

    if VERBOSE:
        print("\n********** Prepare the dataset")

    clean_df = data_frame.copy()
    # only keep airline_sentiment larger than CONFIDENCE_CUTOFF
    clean_df = clean_df[clean_df['airline_sentiment'] > CONFIDENCE_CUTOFF]
    # quantify sentiment
    clean_df['sentiment'] = clean_df['airline_sentiment'].apply(
        bin_connotation)
    # tokenize
    clean_df['clean_text'].apply(tweet_word_check)
    # TODO: padding/truncationg

    train_df, test_df = train_test_split(
        clean_df, test_size=0.3, random_state=42)

    train_set = [tweet for tweet in train_df['clean_text']]
    test_set = [tweet for tweet in test_df['clean_text']]

    if VERBOSE:
        print("training set of size {}, test set of size {}".format(
            len(train_set), len(test_set)))
        print("\n********** declare classifiers")

    return 0


if __name__ == "__main__":
    main()
