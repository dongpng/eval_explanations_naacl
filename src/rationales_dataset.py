import glob
import random
import dataset

from sklearn.feature_extraction.text import TfidfVectorizer

### Code to process the rationales dataset
### Data comes from Using "Annotator Rationales" to Improve Machine Learning for Text Categorization
### Omar F. Zaidan, Jason Eisner, and Christine Piatko, NAACL 2007


POSITIVE_DIR = "../data/review_polarity_rationales/withRats_pos"
NEGATIVE_DIR = "../data/review_polarity_rationales/withRats_neg"

class RationalesData(dataset.Dataset):

    def get_categories(self):
        return ['neg', 'pos']

    def get_name(self):
        return 'rationales'

    def get_config_str(self):
        result = "Categories: %s" % str(self.get_categories())
        result += "\nName: %s " % self.get_name()
        result += POSITIVE_DIR + "\n"
        result += NEGATIVE_DIR + "\n"
        return result

    def read_dataset(self):
        """ Returns data split into train and test"""
        random.seed(9373456352)

        # The data
        texts = []
        texts_with_rationales = []
        labels = []
        filenames = []
        # Read in the the positve and negative data
        for file in sorted(glob.glob(POSITIVE_DIR + '/*.txt')):
            text =  open(file, 'r').read()
            texts_with_rationales.append(text)
            text = text.replace('<POS>', '').replace('</POS>', '')
            text = text.replace('<NEG>', '').replace('</NEG>', '')
            texts.append(text)
            labels.append(1)
            filenames.append(file)
          
        for file in sorted(glob.glob(NEGATIVE_DIR + '/*.txt')):
            text =  open(file, 'r').read()
            texts_with_rationales.append(text)
            text = text.replace('<POS>', '').replace('</POS>', '')
            text = text.replace('<NEG>', '').replace('</NEG>', '')
            texts.append(text)
            labels.append(0)
            filenames.append(file)
      
        # Divide into a training and test set
        self.train_x = []
        self.train_y = []
        self.train_rationales = []

        self.dev_x = []
        self.dev_y = []
        self.dev_rationales = []      

        self.test_x = []
        self.test_y = []
        self.test_rationales = []

        self.splits = []

        for i in range(len(texts)):
            r = random.random()
            if r < 0.6:
                self.train_x.append(texts[i])
                self.train_y.append(labels[i])
                self.train_rationales.append(texts_with_rationales[i])
                self.splits.append((filenames[i], "train"))
            elif r < 0.8:
                self.dev_x.append(texts[i])
                self.dev_y.append(labels[i])
                self.dev_rationales.append(texts_with_rationales[i])
                self.splits.append((filenames[i], "dev"))
            else:
                self.test_x.append(texts[i])
                self.test_y.append(labels[i])
                self.test_rationales.append(texts_with_rationales[i])
                self.splits.append((filenames[i], "test"))

        
    