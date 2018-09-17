
from utils.util import get_sub_dirs, pad_string
from utils.metrics import *
from utils.logging import *
from utils.constants import TEMP_PATH

import tensorflow as tf
import numpy as np
import os

# file paths
kfold_split_file_path = ''


def evaluate(model, config, validation_directory, experiment, file_identifier):

    missclassified = {}

    # get number of classes in model
    number_of_classes = config['dataset']['number_of_classes']

    # get class directory names from validation directory
    class_names = get_sub_dirs(validation_directory)

    # get keras labels in label-index format
    label_index = { class_name: index for index, class_name in enumerate(class_names) }

    # prepare confusion table
    confusion = np.zeros((number_of_classes, number_of_classes))

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
        log_misclassifications( f'{ class_name }_misclassififed.txt', missclassified[class_name], class_name )
        experiment.add_artifact( f'../tmp/{ class_name }_misclassififed.txt' )

    # write kvasir legend to results file
    log_class_legend('split_evaluation_summary.txt', class_names)

    # write confusion table to results file
    log_confusion_table('split_evaluation_summary.txt', confusion)

    # write model summary to results file
    log_model_results('split_evaluation_summary.txt', metrics, file_identifier)

    # write summaries for each class
    for class_name in class_names:

        # class index
        class_index = label_index[class_name]

        # write class summary to results file
        log_class_results(f'{ class_name }_results.txt', metrics, class_name, class_index)

        # add results to experiment
        experiment.add_artifact(f'../tmp/{ class_name }_results.txt')

    # add evaluation files to experiment
    experiment.add_artifact('../tmp/split_evaluation_summary.txt')

    # return evaluation metrics
    return {
        'f1': np.mean(f1),
        'rec': np.mean(rec),
        'acc': np.mean(acc),
        'prec': np.mean(prec),
        'spec': np.mean(spec),
        'mcc': np.mean(mcc)
    }

            