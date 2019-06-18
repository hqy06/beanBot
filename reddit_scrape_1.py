import praw
import os 
from praw.models import MoreComments
import pandas as pd
from pandas import DataFrame

# server requests
reddit = praw.Reddit(client_id='U6e7Akdybwf94w',
                     client_secret='CYjRrJkldCdgdm1N_G8Nywo4s1g', 
                     password='72186996',
                     user_agent='joop', 
                     username='joopica')

# scraping with query 
posts = []

McGill_sr = reddit.subreddit('McGill')
search = McGill_sr.search('mental health')

for post in search:
	comments = []
	thread = reddit.submission(id=post.id).comments
	thread.replace_more(limit=None)
	for top_comment in thread.list(): #reddit.submission(id=post.id).comments.list():
		comments.append(top_comment.body)
		# print(top_comment.body)
	posts.append([post.title, post.id, post.num_comments, post.selftext, comments])

df = pd.DataFrame(posts, columns=['title','id','num_comments','body','comments'])
print(df)

# save to CSV, change your path before saving 
path_name "/Users/jessicachan/Desktop/AI/Bean/"
file_name = "reddit_scrape.csv"
export_csv = df.to_csv(‎⁨file_name, encoding='utf-8', index=True, header=True)

# THINGS TO DO WITH THE TEXT DATA 
# lemmatization 
# convert to lower case 
# getting rid of the and to etc. #stop words in nltk, create your own stop word list!!
# word embedding? sentence embedding? BERT