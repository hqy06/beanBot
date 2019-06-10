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


# ===============================================
# Global Variables
# ===============================================
DEBUG = True
VERBOSE = True
DATASET = "Tweets.csv"

# ===============================================
# Preprocess Data
# ===============================================


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

    # generate data for plotting
    counter = collections.Counter(df_col)
    c = sorted(counter.items())
    keys = [item[0] for item in c]
    values = [item[1] for item in c]

    if VERBOSE:
        print("\n****** {} info".format(name))
        for key, value in zip(keys, values):
            print("{:>20} {:5}".format(key, value))

    # Prepare the canvas
    fig = plt.figure()
    ax = plt.gca()

    # Plot the bar graph and set title
    ax.bar(keys, values, color="plum", ec="orchid", lw=1)
    ax.set_title("review {} count".format(name))

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
        print("\n****** {} dataset info".format(DATASET))
        print(data_frame.info())

    _show_column_count(data_frame['airline'])
    _show_column_count(data_frame['airline_sentiment'])
    _show_column_count_by_class(data_frame, "airline_sentiment", "airline")

    return 0


if __name__ == "__main__":
    main()
