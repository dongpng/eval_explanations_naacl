import time
import pickle

import sklearn.linear_model
import sklearn.ensemble

from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline

class ClassifierWrapper:
    """ Wraps both sklearn and keras"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.pipeline = None
        self.model = None


    def set_properties(self, library_type, model_name, model_dir):
        self.library_type = library_type
        self.model_name = model_name
        print(self.model_name)
        print(model_dir)
        self.model_dir = model_dir

    def predict_raw(self, text):
        """ From unprocessed text to prediction """
        return None

    def save_model(self):
        """ Save the model and configuration data to file """
        self.output_file = self.model_dir + self.model_name + str(int(time.time())) + (".plk" if self.library_type=="sklearn" else ".keras")
        print("saving model %s" % self.output_file)
        self._save_model(self.output_file)

        config_file = self.model_dir + self.model_name + str(int(time.time())) + ".config"
        config_data = {'library_type': self.library_type,
                        'dataset_name': self.dataset.get_name(),
                        'model_name': self.model_name,
                        'model_dir': self.model_dir,
                        'model_file': self.output_file,
                       }

        with open(config_file, 'wb') as handle:
            pickle.dump(config_data, handle)

    def _save_model(self, output_file):
        pass

    def _load_model(self, input_file):
        pass

    def load_model(self, config_file):
        with open(config_file, 'rb') as fp:
            config_data = pickle.load(fp)
            self.library_type = config_data['library_type']
            self.model_name = config_data['model_name']
            self.model_dir = config_data['model_dir']

            self._load_model(config_data['model_file'])

    def get_predictions_test(self):
        """ Return predictions on the test set """
        pass

    def get_pipeline(self):
        """ Return the pipeline to predict a probability"""
        pass

    def get_pipeline_class(self):
        """ Return the pipeline to predict a label"""
        pass

    def report_results(self, gold, pred, name_set):
        print("%s macro: %s %s " % (name_set, self.model_name, sklearn.metrics.f1_score(gold, pred, average='macro')))
        print("%s micro: %s %s " % (name_set, self.model_name, sklearn.metrics.f1_score(gold, pred, average='micro')))
        print("%s Accuracy: %s  %s" % (name_set, self.model_name, sklearn.metrics.accuracy_score(gold, pred)))

class SklearnClassifier(ClassifierWrapper):
    """ Wrapper for sklearn (random forest and logistic regression)"""

    def _save_model(self, output_file):
        joblib.dump(self.model, output_file) 

    def _load_model(self, input_file):
        self.model = joblib.load(input_file)

    def fit_model(self):
        if self.model_name == "rf":
            self.model = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
        elif self.model_name == "lr":
            self.model = sklearn.linear_model.LogisticRegression()
        else:
            print("Error %s" % self.model_name)


        self.model.fit(self.dataset.x_train_trans_tfidf, self.dataset.train_y)
        self.evaluate_model()
        
    def evaluate_model(self):
        """ Evaluate the model on the development and test set """
        pred = self.model.predict(self.dataset.x_dev_trans_tfidf)
        self.report_results(self.dataset.dev_y, pred, "Dev")

        pred = self.model.predict(self.dataset.x_test_trans_tfidf)
        self.report_results(self.dataset.test_y, pred, "Test")

    def get_pipeline(self):
        return make_pipeline(self.dataset.vectorizer, self.model).predict_proba

    def get_pipeline_class(self):
        return make_pipeline(self.dataset.vectorizer, self.model).predict

    def get_predictions_test(self):
        return self.model.predict(self.dataset.x_test_trans_tfidf)
