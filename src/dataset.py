import json
import os 
import pickle
import random

import keras

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

import keras_networks


class Dataset:

    def __init__(self):
        self.tokenize_func = None
        self.name = None

    def get_categories(self):
        return []

    def get_name(self):
        return ""

    def get_config_str(self):
        result = "Categories: %s" % str(self.get_categories())
        result += "\nName: %s " % self.get_name()
        return result

    def prepare_data(self):
        """Preprocessing steps"""
        self.read_dataset()
        self.fit_vectorizer()
        self.transform_binary_encoding()
        self.transform_tfidf_encoding()

    def split_data(self, texts, labels, seed=79234):
        """ Divide into a training, development and test set """
        random.seed(seed)

        self.train_x = []
        self.train_y = []

        self.dev_x = []
        self.dev_y = []   

        self.test_x = []
        self.test_y = []

        self.splits = []

        for i in range(len(texts)):
            r = random.random()
            if r < 0.6:
                self.train_x.append(texts[i])
                self.train_y.append(labels[i])
                self.splits.append((i, "train"))
            elif r < 0.8:
                self.dev_x.append(texts[i])
                self.dev_y.append(labels[i])
                self.splits.append((i, "dev"))
            else:
                self.test_x.append(texts[i])
                self.test_y.append(labels[i])
                self.splits.append((i, "test"))
        
    def fit_vectorizer(self, train_x=None, min_df=10, max_df=0.25, tokenizer=None):
        """Fit a vectorizer"""
        if not train_x:
            train_x = self.train_x

        if tokenizer:
            self.vectorizer = TfidfVectorizer(lowercase=True, min_df=min_df, max_df=max_df, norm=False, tokenizer=tokenizer)
        else:
            self.vectorizer = TfidfVectorizer(lowercase=True, min_df=min_df, max_df=max_df, norm=False)

        self.vectorizer.fit(train_x)
        self.tokenize_func = self.vectorizer.build_analyzer()
        self.vocab = self.vectorizer.vocabulary_
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        """ Tokenize a text"""
        if self.tokenize_func is None:
            self.tokenize_func = self.vectorizer.build_analyzer()
        return self.tokenize_func(text)

    def transform_binary_encoding(self):
        self.x_train_trans_bin = keras_networks.transform_batch_binary(self.train_x, 
                                                                       self.vocab, self.tokenize)

        self.x_dev_trans_bin = keras_networks.transform_batch_binary(self.dev_x, 
                                                                       self.vocab, self.tokenize)        

        self.x_test_trans_bin = keras_networks.transform_batch_binary(self.test_x, 
                                                                       self.vocab, self.tokenize)

        self.y_train_cat = keras.utils.to_categorical(self.train_y, len(self.get_categories()))
        self.y_dev_cat = keras.utils.to_categorical(self.dev_y, len(self.get_categories()))
        self.y_test_cat = keras.utils.to_categorical(self.test_y, len(self.get_categories()))

    def transform_tfidf_encoding(self):
        self.x_train_trans_tfidf = self.vectorizer.transform(self.train_x)
        self.x_dev_trans_tfidf = self.vectorizer.transform(self.dev_x)
        self.x_test_trans_tfidf = self.vectorizer.transform(self.test_x)


    def save_dataset(self, output_dir):
        """Save the data to an output file"""
        with open(output_dir + 'train_x.pickle', 'wb') as handle:
            pickle.dump(self.train_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_dir + 'train_y.pickle', 'wb') as handle:
            pickle.dump(self.train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_dir + 'dev_x.pickle', 'wb') as handle:
            pickle.dump(self.dev_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_dir + 'dev_y.pickle', 'wb') as handle:
            pickle.dump(self.dev_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_dir + 'test_x.pickle', 'wb') as handle:
            pickle.dump(self.test_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_dir + 'test_y.pickle', 'wb') as handle:
            pickle.dump(self.test_y, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        with open(output_dir + 'vocab.pickle', 'wb') as handle:
            pickle.dump(self.vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        with open(output_dir + 'vectorizer.pickle', 'wb') as handle:
            pickle.dump(self.vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.x_train_trans_bin is not None:
            with open(output_dir + 'x_train_trans_bin.pickle', 'wb') as handle:
                pickle.dump(self.x_train_trans_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(output_dir + 'x_dev_trans_bin.pickle', 'wb') as handle:
                pickle.dump(self.x_dev_trans_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)  
            with open(output_dir + 'x_test_trans_bin.pickle', 'wb') as handle:
                pickle.dump(self.x_test_trans_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(output_dir + 'y_train_cat.pickle', 'wb') as handle:
                pickle.dump(self.y_train_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(output_dir + 'y_dev_cat.pickle', 'wb') as handle:
                pickle.dump(self.y_dev_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)    
            with open(output_dir + 'y_test_cat.pickle', 'wb') as handle:
                pickle.dump(self.y_test_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_dir + 'x_train_trans_tfidf.pickle', 'wb') as handle:
            pickle.dump(self.x_train_trans_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_dir + 'x_dev_trans_tfidf.pickle', 'wb') as handle:
            pickle.dump(self.x_dev_trans_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_dir + 'x_test_trans_tfidf.pickle', 'wb') as handle:
            pickle.dump(self.x_test_trans_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)

        config_str = self.get_config_str()
        with open(output_dir + 'config.txt', 'w') as handle:
            handle.write(config_str)

        with open(output_dir + 'splits.txt', 'w') as handle:
            for filename, split in self.splits:
                handle.write("%s\t%s\n" % (filename, split))



    def load_dataset(self, input_dir):
        with open(input_dir + 'train_x.pickle', 'rb') as handle:
            self.train_x = pickle.load(handle)
        with open(input_dir + 'train_y.pickle', 'rb') as handle:
            self.train_y = pickle.load(handle)      
        with open(input_dir + 'dev_x.pickle', 'rb') as handle:
            self.dev_x = pickle.load(handle)
        with open(input_dir + 'dev_y.pickle', 'rb') as handle:
            self.dev_y = pickle.load(handle)
        with open(input_dir + 'test_x.pickle', 'rb') as handle:
            self.test_x = pickle.load(handle)
        with open(input_dir + 'test_y.pickle', 'rb') as handle:
            self.test_y = pickle.load(handle)
        with open(input_dir + 'vocab.pickle', 'rb') as handle:
            self.vocab = pickle.load(handle)                     
            self.inv_vocab = {v: k for k, v in self.vocab.items()}  
        with open(input_dir + 'vectorizer.pickle', 'rb') as handle:
            self.vectorizer = pickle.load(handle)    

        if os.path.isfile(input_dir + 'x_train_trans_bin.pickle'):
            with open(input_dir + 'x_train_trans_bin.pickle', 'rb') as handle:
                self.x_train_trans_bin = pickle.load(handle)    
            with open(input_dir + 'x_dev_trans_bin.pickle', 'rb') as handle:
                self.x_dev_trans_bin = pickle.load(handle)    
            with open(input_dir + 'x_test_trans_bin.pickle', 'rb') as handle:
                self.x_test_trans_bin = pickle.load(handle)    
            with open(input_dir + 'y_train_cat.pickle', 'rb') as handle:
                self.y_train_cat = pickle.load(handle)    
            with open(input_dir + 'y_dev_cat.pickle', 'rb') as handle:
                self.y_dev_cat = pickle.load(handle)    
            with open(input_dir + 'y_test_cat.pickle', 'rb') as handle:
                self.y_test_cat = pickle.load(handle)                                        

        with open(input_dir + 'x_train_trans_tfidf.pickle', 'rb') as handle:
            self.x_train_trans_tfidf = pickle.load(handle)    
        with open(input_dir + 'x_dev_trans_tfidf.pickle', 'rb') as handle:
            self.x_dev_trans_tfidf = pickle.load(handle)               
        with open(input_dir + 'x_test_trans_tfidf.pickle', 'rb') as handle:
            self.x_test_trans_tfidf = pickle.load(handle)    

    def print_stats(self):
        print("Length train %s" % len(self.train_x))
        print("Length dev %s" % len(self.dev_x))
        print("Length test %s" % len(self.test_x))

        print("Train/dev/test label distributions")
        print(Counter(self.train_y))
        print(Counter(self.dev_y))
        print(Counter(self.test_y))

        print("Vocab size")
        print(len(self.vocab))

        total_length = 0
        total_docs = 0
        for d in self.train_x + self.dev_x + self.test_x:
            total_length += len(d)
            total_docs += 1


        print("Avg length in characters %.2f" % (float(total_length)/total_docs))
        print("Total number of docs %s" % total_docs)


    def read_splits(self, input_dir):
        """ read in the file with split information and save file names for each doc id"""
        self.train_files = []
        self.dev_files = []
        self.test_files = []
        with open(input_dir + 'splits.txt', 'r') as handle:
            for line in handle.readlines():
                name, s = line.strip().split("\t")
                if s == "train":
                    self.train_files.append(name)
                elif s == "dev":
                    self.dev_files.append(name)
                elif s == "test":
                    self.test_files.append(name)