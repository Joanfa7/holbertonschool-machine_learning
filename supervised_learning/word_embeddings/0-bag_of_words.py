#!/usr/bin/env python3
''' Module for creating masks to be used in the transformer '''
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    ''' creates a bag of words embedding matrix '''
    vectorizer = CountVectorizer(max_features=sentences)
    X = vectorizer.fit_transform(sentences)
    if vocabulary is None:
        vocabulary = []
    embeddings = vectorizer.fit_transform(sentences).toarray()
    vocab = list(vectorizer.get_feature_names())
    return embeddings, vocab