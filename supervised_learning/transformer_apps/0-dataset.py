#!/usr/bin/env python3
''' Dataset class for transformer model '''

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    ''' Dataset class '''
    def __init__(self):
        ''' contructor '''
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)
        
        
    def tokenize_dataset(self, data):
        ''' creates sub-word tokenizers for our dataset '''

        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

dataset = Dataset()

# Check the shapes and types of the data in data_train
for pt, en in dataset.data_train.take(1):
    print("Portuguese data shape:", pt.shape)
    print("Portuguese data type:", pt.dtype)
    print("English data shape:", en.shape)
    print("English data type:", en.dtype)
    print()