import csv
import datetime
import numpy as np

import scipy
from scipy.stats import spearmanr
from dateutil.parser import parse
import experiments

def cor_intrinsic_extrinsic(input_file, num_words, conf_file=""):
    """ Compute the correlations between the automatic measures and crowd accuracy. 
    input_file: the file with the responses by the crowdworkers
    num_words: if we want to filter on the number of words (or -1 to include everything)
    conf_file: the file with the classifier confidence for each instance
    """

    # If there is a file with classifier confidences, read it and save the values.
    instance_conf = {}
    if conf_file != "":
        with open(conf_file, 'r') as input_file_conf:
            reader = csv.DictReader(input_file_conf)
            for row in reader:
                instance_conf[row['instance_id']] = row['conf']
   


    # for each instance id, store the intrinsic evaluation metrics and the crowd accuracy 
    instance_data = {} # <prediction_type, <instance_id, <intrinsic, [correct count]>>

    with open(input_file) as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            
            # Prediction type
            response = row['choose_the_system_output']
            prediction_type = row['prediction_type']

            # in case we want to filter based on the number of words
            if num_words != -1 and int(row['num_words']) != num_words:
                continue

            if prediction_type not in instance_data:
                instance_data[prediction_type] = {}            

            # init stats with intrinsic measures
            if row['instance_id'] not in instance_data[prediction_type]:
                switch_point = int(row['switch_point'])
                norm_switchpoint = experiments.normalize_switchpoint(switch_point, int(row['doc_length_tokens']))

                instance_data[prediction_type][row['instance_id']] = {'intrinsic': 
                                                                        {'pert_curve_for': float(row['pert_curve_for']), 
                                                                        'norm_switchpoint': norm_switchpoint},
                                                                     'stats': [0,0], # correct, incorrect
                                                                     'confidence': [] # save scores of crowd regarding confidence
                                                                    }

            if int(row['correct']) == 1:
                instance_data[prediction_type][row['instance_id']]['stats'][0] += 1 
            else:
                instance_data[prediction_type][row['instance_id']]['stats'][1] += 1
            
            instance_data[prediction_type][row['instance_id']]['confidence'].append(int(row['i_am_confident_in_my_answer']))

    
    # print results
    for intrinsic_measure in ["norm_switchpoint", "pert_curve_for"]:
        with open('overall_accuracy_' + intrinsic_measure + ".txt", "w") as output_file:
            header = ["instance_id", "prediction_type", intrinsic_measure, "acc"]
            if conf_file != "":
                header += ["conf"]

            w = csv.DictWriter(output_file, header)
            w.writeheader()
            print(intrinsic_measure)

            for prediction_type, pt_data in instance_data.items():
                a = []
                b = []
                c = []
                for instance_id, data in pt_data.items():
                    a.append(pt_data[instance_id]['intrinsic'][intrinsic_measure])
                    b.append(pt_data[instance_id]['stats'][0]/float(pt_data[instance_id]['stats'][0] + pt_data[instance_id]['stats'][1]))
                    c.append(np.array(pt_data[instance_id]['confidence']).mean())

                    to_write = {'instance_id': instance_id,
                                'prediction_type': prediction_type,
                                intrinsic_measure: a[-1],
                                'acc': b[-1]}

                    if conf_file != "":
                        to_write['conf'] = instance_conf[instance_id]

                    w.writerow(to_write)
                    

                cor, pval = spearmanr(a,b)
                pval_print = ""
                if pval < 0.001:
                    pval_print = "***"
                elif pval < 0.01:
                    pval_print = "**"
                elif pval < 0.05:
                    pval_print = "*"
                print("Accuracy %s\t%.3f\t%s" % (prediction_type, cor, pval_print))
                
           

if __name__ == "__main__":
    print("Analyze correlations between automatic and human evaluations")
    # Correlation extrinsic and intrinsic measures
    cor_intrinsic_extrinsic("../experiments/rationales/rationales_cf_responses.csv", -1, 
                            conf_file="../experiments/rationales/rationales_MLP_cf_classifier_conf.csv")

    cor_intrinsic_extrinsic("../experiments/news/news_cf_responses.csv", -1,
                            conf_file="../experiments/news/news_MLP_cf_classifier_conf.csv")


