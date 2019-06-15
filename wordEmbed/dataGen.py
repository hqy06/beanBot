"""
Generate a mental health dataset.

It takes in a list of stem word, obtain their synonyms using WordNet, then use them to do web scraping on r/mentalhealth
"""
# ===============================================
# Import Libraries
# ===============================================
import synGen
import praw
import os
from praw.models import MoreComments
import pandas as pd
from pandas import DataFrame

# ===============================================
# Global Varaibles
# ===============================================
STEM = "stem.txt"
SUBREDDIT = 'mentalhealth'
VERBOSE = True
DEBUG = True


# ===============================================
# Functions
# ===============================================
def web_scraper(word, verbose=VERBOSE):
    """do web scrap on mentalhealth subreddit using given keyword
    adopted from the script by Jessica (@joopica)"""
    # server requests
    reddit = praw.Reddit(client_id='U6e7Akdybwf94w',
                         client_secret='CYjRrJkldCdgdm1N_G8Nywo4s1g',
                         password='72186996',
                         user_agent='joop',
                         username='joopica')

    # scraping with query
    posts = []

    sub_reddit = reddit.subreddit(SUBREDDIT)
    search = sub_reddit.search(word)

    for post in search:
        comments = []
        thread = reddit.submission(id=post.id).comments
        thread.replace_more(limit=None)
        # reddit.submission(id=post.id).comments.list():
        for top_comment in thread.list():
            comments.append(top_comment.body)
            # print(top_comment.body)
        posts.append([post.title, post.id, post.num_comments,
                      post.selftext, comments])

    if VERBOSE:
        print("****** scrape for {} | {}".format(word, len(posts)))

    df = pd.DataFrame(
        posts, columns=['title', 'id', 'num_comments', 'body', 'comments'])
    # print(df)

    # save to CSV, change your path before saving
    # path_name  "/Users/jessicachan/Desktop/AI/Bean/"
    file_name = "scrap-{}.csv".format(word)
    export_csv = df.to_csv(file_name, index=True, header=True)

    return df


# ===============================================
# Main
# ===============================================
def main(verbose=VERBOSE):
    stem_words = synGen.get_stems(STEM)
    synonyms = synGen.generate_synonyms(stem_words, remove_duplicate=True)
    if verbose:
        print(stem_words)
        print(synonyms)

    if DEBUG:
        synonyms = ["McGill"]
    dataframes = []

    for word in synonyms:
        df_w = web_scraper(word)
        dataframes.append(df_w)

    return 0


if __name__ == "__main__":
    main()
