# ===============================================
# Import Libraries
# ===============================================
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from operator import itemgetter

# ===============================================
# Global Variables
# ===============================================

corpus = [[1, 'this is a old serivce'],
          [2, 'this is a cat service'],
          [3,  'this is a nice service']]

query = 'i need cat service'


def encoding(corpus):
    vectorizer = CountVectorizer()
    vectorizer.fit(map(itemgetter(1), corpus))
    return vectorizer


def text_simiar(query, description, vectorizer):
    query_vec, desc_vec = vectorizer.transform([query, description])
    return pairwise_distances(query_vec, desc_vec)


def main(query=query):
    vectorizer = encoding(corpus)
    for sid, svc_desciption in corpus:
        print sid, text_simiar(query, svc_desciption, vectorizer)


if __name__ == "__main__":
    main()
