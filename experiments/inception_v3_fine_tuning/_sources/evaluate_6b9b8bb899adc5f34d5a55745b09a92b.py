
from utils import get_sub_dirs, pad_string, get_kvasir_labels
from metrics import *
from writers import *

import tensorflow as tf
import numpy as np
import os

missclassified = {}

def evaluate(model, config, validation_directory, experiment, file_identifier):

    # get file names
    results_file_name = f'{file_identifier}_{config["summary_files"]["split_evaluation_summary"]}'
    missclassification_file_name = f'{file_identifier}_{config["summary_files"]["missclassification_summary"]}'

    # get temp path
    temp_path = config['misc']['temp_path']

    # get file paths
    results_path = os.path.join(temp_path, results_file_name)
    missclassification_path = os.path.join(temp_path, missclassification_file_name)

    # get number of classes in model
    number_of_classes = config['dataset']['number_of_classes']
    table_size = config['misc']['table_size']

    # get keras labels in label-index format
    label_index = get_kvasir_labels(order='label_index')

    # praper confusion table
    confusion = np.zeros((number_of_classes, number_of_classes))
    
    # get class directory names from validation directory
    class_names = get_sub_dirs(validation_directory)

    # iterate over each class name
    for class_name in class_names:
        print(f'Starting {class_name}')

        # set path to class directory
        class_dir = os.path.join(validation_directory, class_name)

        # iterate over each image in class directory
        for file_name in os.listdir(class_dir):
            
            # use model to classify image
            prediction = model.predict_from_path(os.path.join(class_dir, file_name))

            # check prediction against ground truth, i.e, if it equals
            # the class directory name
            if (prediction != label_index[class_name]):

                # initialize empty list of fist missclassified of class
                if class_name not in missclassified:
                    missclassified[class_name] = []
                
                # append filename to missclassified list
                missclassified[class_name].append(file_name)

            # update confusion table
            confusion[prediction][label_index[class_name]] += 1

    # calculate FP, FN, TP and TN based on confusion table
    FP = confusion.sum(axis=0) - np.diag(confusion)  
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)

    # calculate metrics based on FP, FN, TP and TN
    f1 = f1score(TP, TN, FP, FN)
    rec = recall(TP, TN, FP, FN)
    acc = accuracy(TP, TN, FP, FN)
    prec = precision(TP, TN, FP, FN)
    spec = specificity(TP, TN, FP, FN)
    mcc = matthews_correlation_coefficient(TP, TN, FP, FN)

    # bundle metrics into dictionary
    metrics = { 'FP': FP, 'FN': FN, 'TP': TP, 'TN': TN, 'f1': f1, 'rec': rec, 'acc': acc, 'prec': prec, 'spec': spec, 'mcc': mcc }

    # save missclassified images to file together with class
    for class_name in missclassified:
        write_class_missclassification_files(missclassification_path, missclassified[class_name], class_name, table_size)

    # write kvasir legend to results file
    write_kvasir_legend(results_path, table_size)

    # write confusion table to results file
    write_confusion_table(results_path, confusion, table_size)

    # write model summary to results file
    write_model_summary(results_path, metrics, table_size)

    # write summaries for each class
    for class_name in class_names:

        # class index
        class_index = label_index[class_name]

        # write class summary to results file
        write_class_summary(results_path, metrics, class_name, class_index, table_size)

    # add evaluation files to experiment
    experiment.add_artifact(results_path)
    experiment.add_artifact(missclassification_path)

    # return evaluation metrics
    return {
        'f1': np.mean(f1),
        'rec': np.mean(rec),
        'acc': np.mean(acc),
        'prec': np.mean(prec),
        'spec': np.mean(spec),
        'mcc': np.mean(mcc)
    }

            