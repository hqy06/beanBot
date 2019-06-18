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
from sklearn.model_selection import train_test_split   # train-test split
from sklearn.feature_extraction.text import CountVectorizer  # vectorizer

from sklearn.naive_bayes import GaussianNB  # naive bayes
# two classical ensemble methods
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score  # accuracy

from sklearn.externals import joblib    # for save/load

# ===============================================
# Global Variables
# ===============================================
DEBUG = False
VERBOSE = True
SAVE_MODELS = True
DATASET = "Tweets.csv"
CONFIDENCE_CUTOFF = 0.5
CONNONTATION = 3

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


def get_pred_results(pred, pred_prob, connotation_type=CONNONTATION):
    """
    --- dimensional math:
    pred is a 1-by-N matrix, where N is number of prediction made.
    pred_prob is a N-by-M matrix, where M is the size of support (==connotation_type)
    ---
    returns 2-tuple of lists. (confidence_for_each_prediction, result_of_prediction)
    """
    results = []
    confidences = []
    n_pred = pred.shape[0]
    for i in range(n_pred):
        y = pred[i]  # label of this prediction
        results.append(_decode_connotation(y, connotation_type))
        confidences.append(_get_confidence(y, pred_prob[i], connotation_type))
    assert (len(results) == len(confidences) == n_pred)
    return confidences, results


def _decode_connotation(y, connotation_type):
    assert (connotation_type == 2 or connotation_type == 3)
    if connotation_type == 2:
        result = "positive" if y == 1 else "negative/neutral"
    else:
        if y == 0:
            result = "neutral"
        else:
            result = "positive" if y == 1 else "negative"
    return result


def _get_confidence(y, y_prob, connotation_type):
    assert (connotation_type == 2 or connotation_type == 3)
    if connotation_type == 2:
        confidence = y_prob[y]
    else:
        confidence = y_prob[y + 1]
    return confidence


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
            print("*** {} of class {}".format(count_name, cla))

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
    if CONNONTATION == 2:
        connotation = bin_connotation
    elif CONNONTATION == 3:
        connotation = tri_connotation
    else:
        raise ValueError(
            'incorrect CONNONTATION number, except 2 or 3, get {} instead'.format(CONNONTATION))

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

    if VERBOSE:
        print("\n********** Prepare the dataset")

    clean_df = data_frame.copy()
    # only keep airline_sentiment larger than CONFIDENCE_CUTOFF
    clean_df = clean_df[clean_df['airline_sentiment_confidence']
                        > CONFIDENCE_CUTOFF]
    # quantify sentiment
    clean_df['sentiment'] = clean_df['airline_sentiment'].apply(
        connotation)
    # tokenize
    clean_df['clean_text'] = clean_df['text'].apply(tweet_word_check)

    train_df, test_df = train_test_split(
        clean_df, test_size=0.2, random_state=42)

    train_set = [tweet for tweet in train_df['clean_text']]
    test_set = [tweet for tweet in test_df['clean_text']]

    v = CountVectorizer(analyzer="word")
    train_features = v.fit_transform(train_set)
    test_features = v.transform(test_set)
    assert (test_features.shape[1] == train_features.shape[1])
    n_features = test_features.shape[1]

    if VERBOSE:
        print("training set of size {}, test set of size {}".format(
            len(train_set), len(test_set)))
        print("train features: {}, test features: {}".format(
            train_features.shape, test_features.shape))
        print("\n********** declare classifiers")

    # Random Forest
    rand_forest = RandomForestClassifier(n_estimators=200)
    # AdaBoost
    ada_boost = AdaBoostClassifier(n_estimators=100, random_state=42)
    # Naive Bayes with Gaussian
    naive_bayes = GaussianNB()
    # TODO: RNN LSTM or CNN

    classifiers = [rand_forest, ada_boost, naive_bayes]

    if VERBOSE:
        print("\n********** classification in progress")

    for clf in classifiers:
        if VERBOSE:
            print("Classifier: {}".format(clf.__class__.__name__))
        try:
            clf.fit(train_features, train_df['sentiment'])
            pred_test = clf.predict(test_features)
        except Exception:   # for Gaussian Naive Bayes
            if DEBUG and VERBOSE:
                print("\tException Caught!")
            model = clf.fit(
                train_features.toarray(), train_df['sentiment'])
            pred_test = clf.predict(test_features.toarray())

        # train_socre = accuracy_score(pred_train, test_df['sentiment'])
        test_score = accuracy_score(pred_test, test_df['sentiment'])

        if VERBOSE:
            print("\ttest score: {:.5f}".format(test_score))

    if not DEBUG:
        if CONNONTATION == 2:
            print("\n1 for positive, 0 for negative & neutral")
        if CONNONTATION == 3:
            print("\n1 for positive, 0 for neutral and -1 for negative")

        text = input("Key in a sentence: ")
        while text is not '':
            clean_text = tweet_word_check(text)
            xx = v.transform([clean_text])
            for clf in classifiers:
                clf_name = clf.__class__.__name__
                try:
                    pred = clf.predict(xx)
                    pred_prob = clf.predict_proba(xx)
                except Exception:
                    pred = clf.predict(xx.toarray())
                    pred_prob = clf.predict_proba(xx.toarray())
                confidences, results = get_pred_results(
                    pred, pred_prob, CONNONTATION)
                print("{:>30} | {} {:>15} {:.5f}" .format(
                    clf_name, pred[0], results[0], confidences[0]))
            text = input("Key in a tweet. Press enter to exit: ")
        print(">> Exit prediction model\n")

    if SAVE_MODELS:
        for clf in classifiers:
            f_name = "{} {}.pkl".format(
                datetime.now().strftime('%Y%m%d-%H%M'), clf.__class__.__name__)
            joblib.dump(clf, f_name)
            if VERBOSE:
                print("* {} saved to pkl file.".format(clf.__class__.__name__))
    if VERBOSE:
        print("************ ALL SET!")

    return 0


if __name__ == "__main__":
    main()
