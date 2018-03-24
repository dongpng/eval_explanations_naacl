import glob
import random

from sklearn.feature_extraction.text import TfidfVectorizer

import dataset

### Code to process the Twenty Newsgroups data """

ATHEIST_TRAIN_DIR = "../data/20news-bydate/20news-bydate-train/alt.atheism"
CHRISTIAN_TRAIN_DIR = "../data/20news-bydate/20news-bydate-train/soc.religion.christian"

ATHEIST_TEST_DIR = "../data/20news-bydate/20news-bydate-test/alt.atheism"
CHRISTIAN_TEST_DIR = "../data/20news-bydate/20news-bydate-test/soc.religion.christian"


class TwentyNewssGroupsData(dataset.Dataset):

    def get_name(self):
        return 'newsgroups'

    def get_categories(self):
        return ['christian', 'atheist']

    def get_config_str(self):
        result = "Categories: %s" % str(self.get_categories())
        result += "\nName: %s " % self.get_name()
        result += ATHEIST_TRAIN_DIR + "\n"
        result += CHRISTIAN_TRAIN_DIR + "\n"
        result += ATHEIST_TEST_DIR + "\n"
        result += CHRISTIAN_TEST_DIR
        return result

    def read_dataset(self):
        """ Returns data split into train and test"""
        random.seed(58454)

        # The data
        self.train_x = []
        self.train_y = []

        self.dev_x = []
        self.dev_y = []   

        self.test_x = []
        self.test_y = []

        self.splits = []

        ## Divide training into train + dev
        for filename in sorted(glob.iglob(ATHEIST_TRAIN_DIR + '/*')):
            with open(filename, 'r', encoding='ISO-8859-1') as input_file:
                text = input_file.read()
                if random.random() < 0.2:
                    self.dev_x.append(text)
                    self.dev_y.append(1)
                    self.splits.append((filename, "dev"))
                else:
                    self.train_x.append(text)
                    self.train_y.append(1)
                    self.splits.append((filename, "train"))

        for filename in sorted(glob.iglob(CHRISTIAN_TRAIN_DIR + '/*')):
            with open(filename, 'r', encoding='ISO-8859-1') as input_file:
                text = input_file.read()
                if random.random() < 0.2:
                    self.dev_x.append(text)
                    self.dev_y.append(0)
                    self.splits.append((filename, "dev"))
                else:
                    self.train_x.append(text)
                    self.train_y.append(0)
                    self.splits.append((filename, "train"))
       
        # Test set
        for filename in sorted(glob.iglob(ATHEIST_TEST_DIR + '/*')):
            with open(filename, 'r', encoding='ISO-8859-1') as input_file:
                text = input_file.read()
                self.test_x.append(text)
                self.test_y.append(1)
                self.splits.append((filename, "test"))

        for filename in sorted(glob.iglob(CHRISTIAN_TEST_DIR + '/*')):
            with open(filename, 'r', encoding='ISO-8859-1') as input_file:
                text = input_file.read()
                self.test_x.append(text)
                self.test_y.append(0)    
                self.splits.append((filename, "test"))

    def fit_vectorizer(self):
        super().fit_vectorizer(min_df=5)
