#!/usr/bin/env python3
''' Module for creating masks to be used in the transformer '''
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    ''' creates a bag of words embedding matrix '''
    if vocab is None:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names_out()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    return embeddings, vocab