from __future__ import print_function
import time

import numpy as np
import scipy

from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import load_model
from scipy import spatial

import classifierwrapper


### Transformation
def transform_binary_matrix(text, vocab, tokenize):
    """ no embeddings"""
    result = np.zeros(len(vocab))
    tokens = tokenize(text)
    for token in tokens:
        if token in vocab:
            result[vocab[token]] = 1

    return result

def transform_batch_binary(texts, vocab, tokenize):
    """ Transform the list of texts """
    result = np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        result[i] = transform_binary_matrix(text, vocab, tokenize) 
    return result


### Gradients/saliency
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    return get_activations([X_batch, 0])


def compile_saliency_function(model):
    inp = model.layers[0].input
    outp = model.layers[-1].output
    max_output = K.max(outp, axis=1) 
    saliency = K.gradients(K.sum(max_output), inp)[0] 
    return K.function([inp, K.learning_phase()], [saliency,outp,max_output])



### Omission

def omission_summary_binary(text_transformed, model, inv_vocab):
    """ Omission scores based on binary representation """

    # init
    modifications = [text_transformed] # first document is the document with no words removed
    words = [""]

    # For all words, replace it with 0 (binary encoding)
    for i, w in enumerate(text_transformed):
        if w > 0:
            modified_text = list(text_transformed) #copy
            modified_text[i] = 0
            modifications.append(modified_text)
            words.append(inv_vocab[i])
            
    # Get the activations
    activations_text_idx = get_activations(model, len(model.layers) - 1, modifications)
    activation_base = activations_text_idx[0][0] #activations for the orig document
    
    # For each word, get the score by looking at distance with the original text.
    word_scores = {}
    for i in range(1, len(activations_text_idx[0])):
        activation = activations_text_idx[0][i]
         #doesn't make sense in current setup (layer -1). This score isn't used anywhere in the analysis (score index is 1 in the experiments file. We only use score diff.)
        score_cosine = scipy.spatial.distance.cosine(activation, activation_base)
        max_index = np.argmax(activation_base)
        score_diff = (activation_base-activation)[max_index]
        word_scores[words[i]] = (score_cosine, score_diff)

    return word_scores

class KerasNetwork(classifierwrapper.ClassifierWrapper):

    def _save_model(self, output_file):
        self.model.save(output_file)

    def _load_model(self, input_file):
        self.model = load_model(input_file)


class MLPSimple2(KerasNetwork):

    def fit_model(self):
        batch_size = 32
        epochs = 5
        num_classes = len(self.dataset.y_train_cat[0])

        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(len(self.dataset.vocab),)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes)) #number of classes
        self.model.add(Activation('softmax'))
                
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


        print('Train...')
        self.model.fit(self.dataset.x_train_trans_bin, self.dataset.y_train_cat,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(self.dataset.x_dev_trans_bin, 
                                             self.dataset.y_dev_cat))

    def predict_raw(self, text):
        """ Pipeline from raw text to prediction """
        texts_transformed = transform_batch_binary(text, self.dataset.vocab, self.dataset.tokenize)
        return self.model.predict_proba(texts_transformed, verbose=0)
        
    def predict_raw_class(self, text):
        texts_transformed = transform_batch_binary(text, self.dataset.vocab, self.dataset.tokenize)
        return self.model.predict_classes(texts_transformed, verbose=0)


    def get_predictions_test(self):
        return self.model.predict_classes(self.dataset.x_test_trans_bin, verbose=0)
   
    def get_pipeline(self):
        return self.predict_raw

    def get_pipeline_class(self):
        return self.predict_raw_class


    def evaluate_model(self):
        pred = self.model.predict_classes(self.dataset.x_dev_trans_bin, verbose=0)
        self.report_results(self.dataset.dev_y, pred, "Dev")

        pred = self.model.predict_classes(self.dataset.x_test_trans_bin, verbose=0)
        self.report_results(self.dataset.test_y, pred, "Test")


