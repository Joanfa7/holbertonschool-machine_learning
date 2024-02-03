#!/usr/bin/env python3
''' Module for creating masks to be used in the transformer '''
from sklearn.feature_extraction.text import TfidfVectorizer


def tf(sentences, vocab=None):
    ''' creates a bag of words embedding matrix '''
    if vocab is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names_out()
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    return embeddings, vocab
