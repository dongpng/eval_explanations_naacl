import ast
import csv
import os
import random
import re
import time

import keras
import numpy as np
import scipy.stats
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics


from lime import lime_text
from lime.lime_text import LimeTextExplainer

from keras.models import load_model

from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline

import explanation_evaluation
import explanation_util
import keras_networks
import rationales_dataset
import twentynewsgroups_dataset
import classifierwrapper



EXPERIMENTS_DIR = "../experiments/"

RATIONALES_DATA_DIR = "rationales/"
NEWS_DATA_DIR = "newsgroups/"
OUTPUT_DATA_DIR = "../output_data/"

NEWS_MLP = "mlp1498635980.config"
NEWS_LR = "lr1498635976.config"

RATIONALES_MLP = "mlp1498635976.config"
RATIONALES_LR = "lr1498635970.config"

#####################################
##### PREPROCESSING  ###############
#####################################

def prepare_data(datasets):
    """ Read and parse the data, save it to directory"""
    if "newsgroups" in datasets:
        newsgroups_data = twentynewsgroups_dataset.TwentyNewssGroupsData()
        newsgroups_data.prepare_data()
        newsgroups_data.save_dataset(OUTPUT_DATA_DIR + NEWS_DATA_DIR)

    if "rationales" in datasets:
        rationales_data = rationales_dataset.RationalesData()
        rationales_data.prepare_data()
        rationales_data.save_dataset(OUTPUT_DATA_DIR + RATIONALES_DATA_DIR)


def train_models(dataset, models):
    """ Train the models for the dataset """
    output_dir = OUTPUT_DATA_DIR + dataset.get_name() + "/"
    
    result = {}

    if "rf" in models:
        rf = classifierwrapper.SklearnClassifier(dataset)
        rf.set_properties("sklearn", "rf", output_dir)
        rf.fit_model()
        print(dir(rf))
        rf.save_model()
        result["rf"] = rf

    if "lr" in models:
        lr = classifierwrapper.SklearnClassifier(dataset)
        lr.set_properties("sklearn", "lr", output_dir)
        lr.fit_model()
        lr.save_model()
        lr.evaluate_model()
        result["lr"] = lr

    if "MLP" in models:
        # Train models
        MLP = keras_networks.MLPSimple2(dataset)
        MLP.set_properties("keras", "mlp", output_dir)
        MLP.fit_model()
        MLP.save_model()
        MLP.evaluate_model()
        result["MLP"] = MLP

    return result

def prepare_and_train():
    """ To load an train the models """
    # First, read the data
    prepare_data(["rationales", "newsgroups"]) 
    
    # Read in the parse data. Print some statistics
    rationales_data = rationales_dataset.RationalesData()
    rationales_data.load_dataset(OUTPUT_DATA_DIR + RATIONALES_DATA_DIR)
    rationales_data.print_stats()
    train_models(rationales_data, ["lr", "MLP"])

    news_data = twentynewsgroups_dataset.TwentyNewssGroupsData()
    news_data.load_dataset(OUTPUT_DATA_DIR + NEWS_DATA_DIR)
    news_data.print_stats()
    train_models(news_data, ["lr", "MLP"])


def print_stats_and_evaluate_models_rationales():
    """Load the movie (rationales) dataset, print stats and evaluate the models """
    print("Load rationales")
    rationales_data = rationales_dataset.RationalesData()
    rationales_data.load_dataset(OUTPUT_DATA_DIR + RATIONALES_DATA_DIR)
    rationales_data.print_stats()

    print("Evaluate models")
    lr_rationales = classifierwrapper.SklearnClassifier(rationales_data)
    lr_rationales.load_model(OUTPUT_DATA_DIR  + RATIONALES_DATA_DIR + RATIONALES_LR)
    lr_rationales.evaluate_model()

    MLP_rationales = keras_networks.MLPSimple2(rationales_data)
    MLP_rationales.load_model(OUTPUT_DATA_DIR  + RATIONALES_DATA_DIR + RATIONALES_MLP)
    MLP_rationales.evaluate_model()

def print_stats_and_evaluate_models_news():
    """Load the 20news dataset, print stats and evaluate the models """
    news_data =  twentynewsgroups_dataset.TwentyNewssGroupsData()   
    news_data.load_dataset(OUTPUT_DATA_DIR + NEWS_DATA_DIR)
    news_data.print_stats()

    print("Evaluate models")
    lr_news = classifierwrapper.SklearnClassifier(news_data)
    lr_news.load_model(OUTPUT_DATA_DIR  + NEWS_DATA_DIR + NEWS_LR)
    lr_news.evaluate_model()

    MLP_news = keras_networks.MLPSimple2(news_data)
    MLP_news.load_model(OUTPUT_DATA_DIR + NEWS_DATA_DIR + NEWS_MLP)
    MLP_news.evaluate_model()


#####################################
##### GENERATE EXPLANATIONS  ###############
#####################################


def get_rationales_data_and_models():
    """ Read in the data and saved models for the rationales dataset """
    rationales_data = rationales_dataset.RationalesData()
    rationales_data.load_dataset(OUTPUT_DATA_DIR + RATIONALES_DATA_DIR)
    MLP_rationales = keras_networks.MLPSimple2(rationales_data)
    MLP_rationales.load_model(OUTPUT_DATA_DIR  + RATIONALES_DATA_DIR + RATIONALES_MLP)
    lr_rationales = classifierwrapper.SklearnClassifier(rationales_data)
    lr_rationales.load_model(OUTPUT_DATA_DIR  + RATIONALES_DATA_DIR + RATIONALES_LR)
    return rationales_data, MLP_rationales, lr_rationales


def get_news_data_and_models():
    """ Read in the data and saved models for the movie dataset"""
    news_data =  twentynewsgroups_dataset.TwentyNewssGroupsData()   
    news_data.load_dataset(OUTPUT_DATA_DIR + NEWS_DATA_DIR)
    MLP_news = keras_networks.MLPSimple2(news_data)
    MLP_news.load_model(OUTPUT_DATA_DIR + NEWS_DATA_DIR + NEWS_MLP)
    lr_news = classifierwrapper.SklearnClassifier(news_data)
    lr_news.load_model(OUTPUT_DATA_DIR  + NEWS_DATA_DIR + NEWS_LR)
    return news_data, MLP_news, lr_news


def explanations_rationales():
    """ Generate explanations for the movie dataset """
    print("Rationales: Load models and data")
    rationales_data, MLP_rationales, lr_rationales = get_rationales_data_and_models()

    print("Write explanations")
    write_explanations_to_file(rationales_data, lr_rationales, "rationales_lr_explanations" + str(int(time.time())) + ".csv", methods=["omission", "lime", "random"])
    write_explanations_to_file(rationales_data, MLP_rationales, "rationales_MLP_explanations" + str(int(time.time())) + ".csv", methods=["omission", "lime", "random", "gradients"])
    

def explanations_news():
    """ Generate explanations for 20news dataset """
    print("News: Load models and data")
    news_data, MLP_news, lr_news = get_news_data_and_models()
    
    write_explanations_to_file(news_data, lr_news, "news_lr_explanations" + str(int(time.time())) + ".csv", methods=["omission", "lime", "random"]) 
    write_explanations_to_file(news_data, MLP_news, "news_MLP_explanations" + str(int(time.time())) + ".csv", methods=["omission", "lime", "random", "gradients"])
    


def get_explanation_map(doc_id, old_prediction, model_wrapper, dataset, methods=["omission", "lime", "random"]):
    """ Return a map with explanations for the docid
    doc_id: the document id
    old_prediction: the prediction of the model for the document
    model_wrapper: wrapper for the classifier/model
    dataset: dataset instance
    methods: which methods to print explanations for
    """
    result = {}
    model = model_wrapper.model
        
    explainer = LimeTextExplainer(class_names=dataset.get_categories())

    result['doc_length_tokens'] = len(np.nonzero(dataset.x_test_trans_bin[doc_id])[0]) #non zero indices
    max_num_words = result['doc_length_tokens']


    # random
    if "random" in methods:
        random_selected_words_for, random_selected_words_against = explanation_util.random_baseline(dataset.x_test_trans_bin[doc_id], 
                                                                                                    dataset, max_num_words, 
                                                                                                    random_seed=896918 + doc_id)

        random_switch_point = explanation_evaluation.get_switch_point_word_deletion(dataset.test_x[doc_id], random_selected_words_for, 
                                                                                   old_prediction, 
                                                                                   model_wrapper.get_pipeline_class(), 
                                                                                   dataset.tokenize)

        random_pert_curve_for = explanation_evaluation.compute_perturbation_curve(dataset.test_x[doc_id], 
                                                                                random_selected_words_for, 
                                                                                old_prediction, 
                                                                                model_wrapper.get_pipeline(), 
                                                                                dataset.tokenize, L=10)

        result['random_pert_curve_for'] = random_pert_curve_for

        random_pert_curve_against = explanation_evaluation.compute_perturbation_curve(dataset.test_x[doc_id], 
                                                                                      random_selected_words_against, 
                                                                                      old_prediction, 
                                                                                      model_wrapper.get_pipeline(), 
                                                                                      dataset.tokenize, L=10)
        
        result['random_pert_curve_against'] = random_pert_curve_against
        result['random_selected_words_for'] = random_selected_words_for
        result['random_selected_words_against'] = random_selected_words_against
        result['random_switch_point'] = random_switch_point
        result['random_normalized_switch_point'] = normalize_switchpoint(random_switch_point, result['doc_length_tokens'])



    if "lime" in methods:
        # seed for lime
        np.random.seed(9863502+doc_id)
        lime_samples = [500, 1000, 1500, 2000, 5000]
        for num_samples in lime_samples:

            lime_selected_words_for,lime_selected_words_against,_ = explanation_util.get_words_to_remove_lime(explainer, dataset.test_x[doc_id], 
                                                                                model_wrapper.get_pipeline(), num_samples, 
                                                                                old_prediction, num_features = max_num_words,
                                                                                max_num_words_to_remove=max_num_words)

            lime_switch_point = explanation_evaluation.get_switch_point_word_deletion(dataset.test_x[doc_id], 
                                                                                      lime_selected_words_for, 
                                                                                      old_prediction, 
                                                                                      model_wrapper.get_pipeline_class(), None) 

            lime_pert_curve_for = explanation_evaluation.compute_perturbation_curve(dataset.test_x[doc_id], 
                                                                                    lime_selected_words_for, 
                                                                                    old_prediction, 
                                                                                    model_wrapper.get_pipeline(), None, L=10)

            result['lime' + str(num_samples) + '_pert_curve_for'] = lime_pert_curve_for

            lime_pert_curve_against = explanation_evaluation.compute_perturbation_curve(dataset.test_x[doc_id], 
                                                                                        lime_selected_words_against, 
                                                                                        old_prediction, 
                                                                                        model_wrapper.get_pipeline(), 
                                                                                        None, L=10)

            result['lime' + str(num_samples) + '_pert_curve_against'] = lime_pert_curve_against
            result['lime' + str(num_samples) + '_selected_words_for'] = lime_selected_words_for
            result['lime' + str(num_samples) + '_selected_words_against'] = lime_selected_words_against
            result['lime' + str(num_samples) + '_switch_point'] = lime_switch_point
            result['lime' + str(num_samples) + '_normalized_switch_point'] = normalize_switchpoint(lime_switch_point, result['doc_length_tokens'])




    if "gradients" in methods and model_wrapper.library_type == "keras":
        gradient_function_mlp = keras_networks.compile_saliency_function(model_wrapper.model)
        gradient_output = gradient_function_mlp([ [dataset.x_test_trans_bin[doc_id]], 0])

        gradients_weights = explanation_util.get_feature_weights_gradients(dataset.x_test_trans_bin[doc_id], 
                                                                           gradient_output[0][0],
                                                                           dataset.inv_vocab
                                                                           )

        gradients_selected_words_for, gradients_selected_words_against = explanation_util.get_top_words(gradients_weights, max_num_words, score_index=0)
        

        gradients_switch_point = explanation_evaluation.get_switch_point_word_deletion(dataset.test_x[doc_id], 
                                                                                        gradients_selected_words_for, 
                                                                                        old_prediction, 
                                                                                        model_wrapper.get_pipeline_class(), dataset.tokenize)

        gradients_pert_curve_for = explanation_evaluation.compute_perturbation_curve(dataset.test_x[doc_id], 
                                                                                     gradients_selected_words_for, 
                                                                                     old_prediction, 
                                                                                     model_wrapper.get_pipeline(),
                                                                                     dataset.tokenize, L=10)

        result['gradients_pert_curve_for'] = gradients_pert_curve_for

        gradients_pert_curve_against = explanation_evaluation.compute_perturbation_curve(dataset.test_x[doc_id], 
                                                                                         gradients_selected_words_against, 
                                                                                         old_prediction, 
                                                                                        model_wrapper.get_pipeline(), 
                                                                                        dataset.tokenize, L=10)

        result['gradients_pert_curve_against'] = gradients_pert_curve_against
        result['gradients_switch_point'] =  gradients_switch_point
        result['gradients_normalized_switch_point'] = normalize_switchpoint(gradients_switch_point, result['doc_length_tokens'])
        result['gradients_selected_words_for'] = gradients_selected_words_for
        result['gradients_selected_words_against'] = gradients_selected_words_against



    # Words based on ommission
    if "omission" in methods:

        omission_selected_words_for,  omission_selected_words_against = None,None

        # Logistic regression
        if model_wrapper.model_name == "lr":
            omission_selected_words_for,  omission_selected_words_against = explanation_util.get_top_words_based_on_weight_dict(
                                                                dataset.x_test_trans_tfidf[doc_id], dataset.inv_vocab, 
                                                                model_wrapper.model.coef_, max_num_words, old_prediction)

            
        # Keras
        else:
            # Omission not necessairly perfect for word wise deletion. Individual words vs multiple words removed
            omission_summary = keras_networks.omission_summary_binary(dataset.x_test_trans_bin[doc_id], model, dataset.inv_vocab)
            omission_selected_words_for,  omission_selected_words_against = explanation_util.get_top_words(omission_summary, max_num_words, 
                                                                            score_index=1)

            

        omission_switch_point = explanation_evaluation.get_switch_point_word_deletion(dataset.test_x[doc_id], omission_selected_words_for, 
                                                                                       old_prediction, 
                                                                                       model_wrapper.get_pipeline_class(), 
                                                                                       dataset.tokenize)
        
        omission_pert_curve_for = explanation_evaluation.compute_perturbation_curve(dataset.test_x[doc_id], 
                                                                                    omission_selected_words_for, old_prediction, 
                                                                                    model_wrapper.get_pipeline(), dataset.tokenize, L=10)

        omission_pert_curve_against = explanation_evaluation.compute_perturbation_curve(dataset.test_x[doc_id], 
                                                                                        omission_selected_words_against, old_prediction, 
                                                                                        model_wrapper.get_pipeline(), dataset.tokenize, L=10)
        
        result['omission_pert_curve_for'] = omission_pert_curve_for
        result['omission_pert_curve_against'] = omission_pert_curve_against
        result['omission_switch_point'] =  omission_switch_point
        result['omission_normalized_switch_point'] = normalize_switchpoint(omission_switch_point, result['doc_length_tokens'])
        result['omission_selected_words_for'] = omission_selected_words_for
        result['omission_selected_words_against'] = omission_selected_words_against

        
    prediction_type = None
    if (old_prediction == dataset.test_y[doc_id]):
        prediction_type = "tp" if old_prediction == 1 else "tn"
    else:
        prediction_type = "fn" if dataset.test_y[doc_id] == 1 else "fp"

    result['doc_id'] =  doc_id
    result['prediction_type'] =  prediction_type
    result['doc_length'] = len(dataset.test_x[doc_id])
    result['classifier'] = model_wrapper.model_name

    return result


def write_explanations_to_file(dataset, model_wrapper, output_file="explanations.csv", methods=["omission", "lime", "random", "gradients"]):
    """ Iterate over the documents and write output to file """
    old_predictions = model_wrapper.get_predictions_test()

    start_doc_id = 0
    explanation = get_explanation_map(start_doc_id, old_predictions[start_doc_id], model_wrapper, dataset, methods)


    with open(output_file, 'w', encoding="utf-8") as f:
        w = csv.DictWriter(f, explanation.keys())
        w.writeheader()

        for doc_id in range(len(dataset.test_x)):
            print("%s from %s" % (doc_id, len(dataset.test_x)))
            explanation = get_explanation_map(doc_id, old_predictions[doc_id], model_wrapper, dataset, methods)
            w.writerow(explanation)
            f.flush()


def normalize_switchpoint(switchpoint, token_length):
    """ Normalize switch point. If the value is -1, take the token length """
    if switchpoint == -1:
        return 1
    else:
        return switchpoint / float(token_length)


#####################################
##### Evaluate the local explanation methods using automatic methods  ###############
#####################################

def output_word_deletion_analysis(input_file, evaluation_type="_pert_curve_for", 
                                  methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission", "gradients"]):
    """ Calculate and print out mean for different evaluation metrics.
    Possible types: _pert_curve_against, _pert_curve_for, _normalized_switch_point """
    input_file = csv.DictReader(open(input_file, 'r', encoding="utf-8"))
    values = {}
    for explanation in input_file:  
        for method in methods:
            if method not in values:
                values[method] = []

            values[method].append(float(explanation[method + evaluation_type]))

    # Print
    for method, v in values.items():
        print("%s\t%.4f\t%.4f" % (method, np.array(v).mean(), np.array(v).std()))




def correlation_analysis(input_file, dataset, model_wrapper, analysis_type, 
                                   methods= ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission"] ):
    """ Check for correlations between evaluation values and other characteristics.
    analysis type: doc_length or confidence
     """
    pipeline = model_wrapper.get_pipeline()
    print(input_file)
    print(analysis_type)
    input_file = csv.DictReader(open(input_file, 'r', encoding="utf-8"))


    # Init
    results = {}
    for method in methods:
        results[method] = {'pbcurves': [], 'switchpoints': [], 'confidences': [], 'doc_lengths': []}


    # Read in data
    for explanation in input_file:

        prediction = explanation['prediction_type']
        token_length = int(explanation['doc_length_tokens'])
        doc_id = int(explanation['doc_id'])

        for method in methods:

            # Get switch point
            results[method]['switchpoints'].append(float(explanation[method + "_normalized_switch_point"]))
            results[method]['pbcurves'].append(float(explanation[method + "_pert_curve_for"]))

            # Get confidence value (just probability)
            confidence = pipeline(
                            [dataset.test_x[doc_id]]
                         )[0]
            confidence = np.max(confidence)
            results[method]['confidences'].append(confidence)
            results[method]['doc_lengths'].append(token_length)
           
    # Print data
    for method in methods:
        if analysis_type == "doc_length":
            print("%s&%.3f&%.3f\\\\" % (method, 
                                    scipy.stats.spearmanr(results[method]['switchpoints'], results[method]['doc_lengths'])[0],
                                    scipy.stats.spearmanr(results[method]['pbcurves'], results[method]['doc_lengths'])[0],
                                    ))
        else:
            print("%s&%.3f&%.3f\\\\" % (method, 
                                    scipy.stats.spearmanr(results[method]['switchpoints'], results[method]['confidences'])[0],
                                    scipy.stats.spearmanr(results[method]['pbcurves'], results[method]['confidences'])[0],
                                    ))

    # Write to file so we can analyze this futher in R (e.g. LM with two independent variables)
    with open('confidence_' + model_wrapper.model_name + "_" + dataset.get_name() + '.txt', 'w') as output_file:
        output_file.write("%s\t%s\t%s\t%s\t%s\n" % ('doc_length', 'method', 'confidence', 'switchpoint', 'pbcurve'))
        for method in methods:
            for i in range(len(results[method]['doc_lengths'])):
                output_file.write("%s\t%s\t%s\t%s\t%s\n" % (results[method]['doc_lengths'][i],
                                                 method,
                                                 results[method]['confidences'][i],
                                                 results[method]['switchpoints'][i],
                                                 results[method]['pbcurves'][i]
                                                ))


def print_correlation_analysis(analysis_type="confidence"):
    """ Print relation between confidence and evaluation metrics.
    Analysis type: doc_length or confidence """
    print("Rationales")
    rationales_data, MLP_rationales, lr_rationales = get_rationales_data_and_models()
    print("LR")
    correlation_analysis(EXPERIMENTS_DIR + "rationales/rationales_lr_explanations.csv",
                                    rationales_data, lr_rationales, analysis_type,
                                    methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission"] 
                                    )
    print("MLP")
    correlation_analysis(EXPERIMENTS_DIR + "rationales/rationales_mlp_explanations.csv",
                                    rationales_data, MLP_rationales, analysis_type,
                                    methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission", "gradients"] 
                                    )
    print("News")
    news_data, MLP_news, lr_news = get_news_data_and_models()
    print("LR")
    correlation_analysis(EXPERIMENTS_DIR + "news/news_lr_explanations.csv",
                                    news_data, lr_news,  analysis_type,
                                    methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission"] 
                                    )
    print("MLP")
    correlation_analysis(EXPERIMENTS_DIR + "news/news_mlp_explanations.csv",
                                    news_data, MLP_news,  analysis_type,
                                    methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission", "gradients"] 
                                    )

def print_evaluations():
    # Movie dataset: Normalized switchpoint 
    output_word_deletion_analysis(EXPERIMENTS_DIR + "rationales/rationales_MLP_explanations.csv", "_normalized_switch_point", 
                                  methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission", "gradients"]) 
    
    output_word_deletion_analysis(EXPERIMENTS_DIR + "rationales/rationales_lr_explanations.csv", "_normalized_switch_point", 
                                  methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission"])
    
    # News dataset: Normalized switchpoint
    output_word_deletion_analysis(EXPERIMENTS_DIR + "news/news_MLP_explanations.csv", "_normalized_switch_point", 
                                  methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission", "gradients"])
    
    output_word_deletion_analysis(EXPERIMENTS_DIR + "news/news_lr_explanations.csv", "_normalized_switch_point", 
                                 methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission"])
    

    # News dataset: AOPC
    output_word_deletion_analysis(EXPERIMENTS_DIR + "news/news_lr_explanations.csv", "_pert_curve_for", 
                                  methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission"])
    
    output_word_deletion_analysis(EXPERIMENTS_DIR + "news/news_lr_explanations.csv", "_pert_curve_against", 
                                 methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission"])
    
    output_word_deletion_analysis(EXPERIMENTS_DIR + "news/news_MLP_explanations.csv", "_pert_curve_for", 
                                  methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission", "gradients"])
    
    output_word_deletion_analysis(EXPERIMENTS_DIR + "news/news_MLP_explanations.csv", "_pert_curve_against", 
                                  methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission", "gradients"])
    
    # Movie dataset: AOPC
    output_word_deletion_analysis(EXPERIMENTS_DIR + "rationales/rationales_lr_explanations.csv", "_pert_curve_for", 
                                  methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission"])
    
    output_word_deletion_analysis(EXPERIMENTS_DIR + "rationales/rationales_lr_explanations.csv", "_pert_curve_against", 
                                   methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission"])

    output_word_deletion_analysis(EXPERIMENTS_DIR + "rationales/rationales_MLP_explanations.csv", "_pert_curve_for", 
                                  methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission", "gradients"])
    
    output_word_deletion_analysis(EXPERIMENTS_DIR + "rationales/rationales_MLP_explanations.csv", "_pert_curve_against", 
                                 methods = ["random", "lime500", "lime1000", "lime1500", "lime2000", "lime5000", "omission", "gradients"])
    

if __name__ == "__main__":
    print("*")
    ##### Preprocess the data and train ML models
    #prepare_and_train()
    
    ##### Read in the processed data and trained models and print statistics
    #print("** Rationales")
    #print_stats_and_evaluate_models_rationales()
    #print("** News")
    #print_stats_and_evaluate_models_news()

    ##### Generate the explanations
    #explanations_rationales()
    #explanations_news()

    #### Evaluate the explanations
    #print_evaluations()
    
    # Correlation analysis
    #print_correlation_analysis()