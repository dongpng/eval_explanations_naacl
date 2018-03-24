import re

import numpy as np
import scipy

import explanation_util
import keras_networks


UNABLE_TO_SWITCH = -1

###### Word deletion / switch point
def get_switch_point_word_deletion(text, words_to_remove_all, old_prediction, pipeline, tokenize):
    """ How many words need to be removed before it is changed? """
    words_to_remove = []
    for i, word in enumerate(words_to_remove_all):
        words_to_remove.append(word)
        if has_prediction_changed(text, words_to_remove, old_prediction, pipeline, tokenize):
            return (i+1)

    return UNABLE_TO_SWITCH


def has_prediction_changed(text, words_to_remove, old_prediction, pipeline, tokenize):
    """ Return True if the prediction has changed after removing the words """
    new_text = remove_word(text, words_to_remove, tokenize)
    return old_prediction != pipeline([new_text])[0]


def remove_word(text, words, tokenize=None):
    """ Remove words from text using the tokenizer provided by the vectorizer"""
    # First, tokenize
    tokens = []
    if not tokenize:
        tokens = re.split(r'(%s)|$' % r'\W+', text) # this comes from LIME code
    else:
        tokens = tokenize(text)

    tokens_new = []
    for token in tokens:
        if token not in words and len(token.strip()) > 0:
            tokens_new.append(token.strip())
    return " ".join(tokens_new)

def compute_perturbation_curve(text, words_to_remove_all, old_prediction, pipeline_prob, tokenize, L=10):
    """ Compute AOPC https://arxiv.org/pdf/1509.06321.pdf"""
    values = []
    words_to_remove = []
    prob_orig = pipeline_prob([text])[0][old_prediction]
    for i, word in enumerate(words_to_remove_all):
        if i == L:
            break
        words_to_remove.append(word)
        new_text = remove_word(text, words_to_remove, tokenize)
        prob = pipeline_prob([new_text])[0][old_prediction]
        values.append(prob_orig - prob)

    return np.array(values).sum()/(L + 1)

