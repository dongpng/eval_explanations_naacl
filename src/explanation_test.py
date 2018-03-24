import unittest

import numpy as np
import numpy.testing as npt

import explanation_util
import explanation_evaluation
import experiments
import keras_networks
import rationales_dataset
import classifierwrapper

class ExplanationTest(unittest.TestCase):


    def mock_pipeline(self, text):
        if '125' not in text[0]:
            return [0]
        else:
            return [1]

    def mock_pipeline_prob(self, text):
        if 'test' in text[0]:
            return [[0, 1]]
        if 'a' in text[0]:
            return [[0.2, 0.8]]
        if '125' in text[0]:
            return [[0.3, 0.7]]
        else:
            return [[0.5, 0.5]]

    def tokenize(self, text):
        return text.split(" ")

    def test_top_words(self):
        a = {'a': [-1],
             'b': [-2],
             'c': [0.5],
             'd': [1],
             'e': [-1]}

        expected2 = (['d', 'c'], ['b', 'a'])
        result = explanation_util.get_top_words(a, 2, score_index=0)
        self.assertListEqual(expected2[0], result[0])
        self.assertListEqual(expected2[1], result[1])

        expected2 = (['d'], ['b'])
        result = explanation_util.get_top_words(a, 1, score_index=0)
        self.assertListEqual(expected2[0], result[0])
        self.assertListEqual(expected2[1], result[1])

    def test_switch_point(self):
        document = "this is a test hello 125"
        selected_words = ["test", "a", "125", "this"]

        self.assertEqual(explanation_evaluation.get_switch_point_word_deletion(document, selected_words, 
                                                                                1, self.mock_pipeline, None), 3)
    def test_perturbation_curve(self):
        document = "this is a test hello 125"
        selected_words = ["test", "a", "125", "this"]
        # 0 
        # 1: remove test: 1 - 0.8 = 0.2
        # 2: remove a: 1 - 0.7 = 0.3
        # (0.2 + 0.3)/(2 + 1)
        self.assertEqual(explanation_evaluation.compute_perturbation_curve(document, selected_words, 1, self.mock_pipeline_prob, self.tokenize, L=2), 0.5/3)



    def test_omission(self):
        rationales_data = rationales_dataset.RationalesData()
        rationales_data.load_dataset(experiments.OUTPUT_DATA_DIR + experiments.RATIONALES_DATA_DIR)
    
        
        MLP_rationales = keras_networks.MLPSimple2(rationales_data)
        MLP_rationales.load_model(experiments.OUTPUT_DATA_DIR  + experiments.RATIONALES_DATA_DIR + experiments.RATIONALES_MLP)
        MLP_rationales.evaluate_model()

        doc_id = 0

        base_pred = MLP_rationales.model.predict_proba(np.array([rationales_data.x_test_trans_bin[doc_id]]), verbose=0)[0]
        max_index = np.argmax(base_pred) 

        omission_summary = keras_networks.omission_summary_binary(rationales_data.x_test_trans_bin[doc_id], MLP_rationales.model, rationales_data.inv_vocab)

        for word, score in omission_summary.items():
            s = score[1]
            modified_text = list(rationales_data.x_test_trans_bin[doc_id]) #copy
            modified_text[rationales_data.vocab[word]] = 0
            new_pred = MLP_rationales.model.predict_proba(np.array([modified_text]), verbose=0)[0]

            npt.assert_almost_equal(base_pred[max_index] - new_pred[max_index], s, decimal=7)


    def test_feature_weight_extraction(self):
        coefficients = np.array([[0.7, 0.1, 0.2, -0.3]])
        print(coefficients.shape)
        invdict = {0 : 'a', 1: 'b', 2:'c', 3:'d'}
        expected1 = {'a': 0.7, 'b': 0.1, 'c': 0.2, 'd': -0.3}
        expected2 = {'a': -0.7, 'b': -0.1, 'c': -0.2, 'd': 0.3}
        npt.assert_equal(explanation_util.get_feature_weights(coefficients, invdict, 1), expected1)
        npt.assert_equal(explanation_util.get_feature_weights(coefficients, invdict, 0), expected2)


    def test_lr_explanation(self):
        coefficients = np.array([[0.7, 0.1, 0.2, -0.3]])
        invdict = {0 : 'a', 1: 'b', 2:'c', 3:'d'}
        text = np.array([[0, 1, 1, 1]])
        expected1 = (['c', 'b'], ['d'])
        npt.assert_equal(explanation_util.get_top_words_based_on_weight_dict(text, invdict, coefficients, 2, 1), expected1)
        expected2 = (['d'], ['c', 'b'])
        npt.assert_equal(explanation_util.get_top_words_based_on_weight_dict(text, invdict, coefficients, 2, 0), expected2)
        expected3 = (['d'], ['c'])
        npt.assert_equal(explanation_util.get_top_words_based_on_weight_dict(text, invdict, coefficients, 1, 0), expected3)

        coefficients = np.array([[0.7, 0.1, 0.2, -0.3]])
        invdict = {0 : 'a', 1: 'b', 2:'c', 3:'d'}
        text = np.array([[0, 3, 1, 1]])
        expected4 = (['b', 'c'], ['d'])
        npt.assert_equal(explanation_util.get_top_words_based_on_weight_dict(text, invdict, coefficients, 2, 1), expected4)
        

    def test_omission_lr(self):
        rationales_data = rationales_dataset.RationalesData()
        rationales_data.load_dataset(experiments.OUTPUT_DATA_DIR + experiments.RATIONALES_DATA_DIR)
        doc_id = 63
        model_wrapper = classifierwrapper.SklearnClassifier(rationales_data)
        model_wrapper.load_model(experiments.OUTPUT_DATA_DIR  + experiments.RATIONALES_DATA_DIR + experiments.RATIONALES_LR)
        tmp = rationales_data.x_test_trans_tfidf[doc_id]
        _, num_features = tmp.shape

         # Iterate over all features
        score = 0
        score_without_wisc = 0
        for idx in range(num_features):
            if tmp[0,idx] > 0:
                #print("%s\t%s\t%s\t%s" % (idx, tmp[0, idx], dataset.inv_vocab[idx], tmp[0, idx] * model_wrapper.model.coef_[0][idx]))
                score += tmp[0, idx] * model_wrapper.model.coef_[0][idx]
                if rationales_data.inv_vocab[idx] != "wisc":
                    score_without_wisc += tmp[0, idx] * model_wrapper.model.coef_[0][idx]

        # Add intercept
        score += model_wrapper.model.intercept_
        score_without_wisc += model_wrapper.model.intercept_
        npt.assert_almost_equal(score, model_wrapper.model.decision_function(rationales_data.x_test_trans_tfidf[doc_id]))
       
        # Check without a word     
        to_remove = ["wisc"]
        new_text = explanation_evaluation.remove_word(rationales_data.test_x[doc_id], to_remove, rationales_data.tokenize)
        npt.assert_almost_equal(score_without_wisc, model_wrapper.model.decision_function(rationales_data.vectorizer.transform([new_text])))
        

    

if __name__ == '__main__':
    unittest.main()

