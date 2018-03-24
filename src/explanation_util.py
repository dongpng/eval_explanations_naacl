import re
import random

import numpy as np

#####################################
##### Local explanation approaches  ###############
#####################################
def random_baseline(bin_representation, dataset, max_num_words_to_remove, random_seed=2352398):
    """ Randomly select words """
    random.seed(random_seed)

    # first get all the words that are in the document
    selected_words = []
    for i, val in enumerate(bin_representation):
        if val > 0:
            selected_words.append(dataset.inv_vocab[i])

    # randomly select words
    random.shuffle(selected_words)
    random_to_remove_for = selected_words[:max_num_words_to_remove]
    random.shuffle(selected_words)
    random_to_remove_against = selected_words[:max_num_words_to_remove]
    return random_to_remove_for, random_to_remove_against



#### LIME ########
def get_words_to_remove_lime(lime_explainer, text, pipeline_function, num_samples, old_prediction, num_features=None, max_num_words_to_remove=None):
    """ Find words to remove for lime """
    if num_features is None:
        if max_num_words_to_remove is not None:
            num_features = max_num_words_to_remove
        else:
            num_features = 200

    # Get the explanation from LIME
    exp = lime_explainer.explain_instance(text, pipeline_function, num_features=num_features, num_samples=num_samples, labels=[old_prediction])


    # Find words to remove such that prediction changes. Assumes words are orderd by (abs) score
    words_to_remove_for = []
    for word, score in exp.as_list(label=old_prediction):
        if score > 0:
            words_to_remove_for.append(word)
        if max_num_words_to_remove and len(words_to_remove_for) == max_num_words_to_remove:
            break

    ## The important words against this prediction
    words_to_remove_against = [] 
    for word, score in exp.as_list(label=old_prediction):
        if score < 0:
            words_to_remove_against.append(word)
        if max_num_words_to_remove and len(words_to_remove_against) == max_num_words_to_remove:
            break


    return words_to_remove_for, words_to_remove_against, exp.score


def get_feature_weights_gradients(bin_representation, gradient_output, inv_vocab):
    """ Given the weights returned by the gradient, return map with weights and vocab """
    results = {}
    for i, val in enumerate(bin_representation):
        if val > 0:
            results[inv_vocab[i]] = [gradient_output[i]]  
    return results


def get_top_words(scores, num_words, score_index=0):
    """ Given a vocab with scores, return top weighted features """
    dict_values = {}

    # Read in the scores
    for word, values in scores.items():
        dict_values[word] = values[score_index]

    # Extract top words for and against
    selected_words_for = []
    for w in sorted(dict_values, key=dict_values.get, reverse=True)[:num_words]:
        if dict_values[w] < 0:
            break
        selected_words_for.append(w)

    selected_words_against = []
    for w in sorted(dict_values, key=dict_values.get, reverse=False)[:num_words]:
        if dict_values[w] > 0:
            break
        selected_words_against.append(w)       
  
    return selected_words_for, selected_words_against


def get_feature_weights(coefficients, invdict, predicted_class):
    """ Returns the feature weights for the tokens such as highest ranked features are 
    for the predicted class

    """
    feature_weights = {}

    # Multiclass
    if coefficients.shape[0] > 1:
        coefficients = coefficients[predicted_class]

        for i in range(len(coefficients)):
            feature_weights[invdict[i]] = coefficients[i]

    else:
        coefficients = coefficients[0]
        for i in range(len(coefficients)):
            feature_weights[invdict[i]] = coefficients[i] if predicted_class == 1 else -coefficients[i]


    return feature_weights


### Linear model
def get_top_words_based_on_weight_dict(transformed_text, invdict, coefficients, num_words, pred):
    """ Get words with the most weights in this model. Assume binary classification"""

    non_zero = transformed_text.nonzero()[1]


    # Saves the weights for the tokens presented in this text
    feature_weights = get_feature_weights(coefficients, invdict, pred)
    token_weights = {}

    # For each token, get the weight in the model
    for token_idx in non_zero:
        token = invdict[token_idx]
        if token not in token_weights:
            token_weights[token] = feature_weights[token] * transformed_text[0,token_idx] # multiply by feature value 
           
    # Get tokens with the most positive and negative weights in the model
    selected_words_for = []
    for w in sorted(token_weights, key=token_weights.get, reverse=True)[:num_words]:
        if token_weights[w] < 0:
            break
        selected_words_for.append(w)
    
    selected_words_against = []
    for w in sorted(token_weights, key=token_weights.get, reverse=False)[:num_words]:
        if token_weights[w] > 0:
            break
        selected_words_against.append(w)
        
  
    return selected_words_for, selected_words_against



