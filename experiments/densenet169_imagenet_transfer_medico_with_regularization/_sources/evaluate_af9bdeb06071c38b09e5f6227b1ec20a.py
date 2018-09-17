from utils.util import get_sub_dirs, pad_string
from utils.metrics import *
from utils.logging import *

from scipy.misc import imread, imsave, imresize

import tensorflow as tf
import numpy as np
import os

# file paths
kfold_split_file_path = ''


def evaluate(model, config, experiment, validation_directory, file_identifier=''):

    missclassified = {}

    # get number of classes in model
    number_of_classes = config['dataset']['number_of_classes']

    # image dimensions
    image_width = config['image_processing']['image_width']
    image_height = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    # get class directory names from validation directory
    class_names = get_sub_dirs(validation_directory)
    class_names.sort()

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

            # models class prediction for image
            prediction = None

            # process image before passing it through the network
            image = imread(os.path.join(class_dir, file_name), mode='RGB')
            image = imresize(image, (image_width, image_height, image_channels))
            image = image.reshape(1, image_width, image_height, image_channels)
            image = np.true_divide(image, 255.)

            with tf.get_default_graph().as_default():
                prediction = np.argmax(model.predict(image)[0])

            # check prediction against ground truth, i.e, if it equals the class directory name
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

    print ( f"True Positives: { TP }" )
    print ( f"True Negatives: { TN }" )
    print ( f"False Positives: { FP }" )
    print ( f"False Positives: { FN }" )

    # calculate metrics based on FP, FN, TP and TN
    f1 = np.nan_to_num(f1score(TP, TN, FP, FN))
    rec = np.nan_to_num(recall(TP, TN, FP, FN))
    acc = np.nan_to_num(accuracy(TP, TN, FP, FN))
    prec = np.nan_to_num(precision(TP, TN, FP, FN))
    spec = np.nan_to_num(specificity(TP, TN, FP, FN))
    mcc = np.nan_to_num(matthews_correlation_coefficient(TP, TN, FP, FN))

    # bundle metrics into dictionary
    metrics = { 'FP': FP, 'FN': FN, 'TP': TP, 'TN': TN, 'f1': f1, 'rec': rec, 'acc': acc, 'prec': prec, 'spec': spec, 'mcc': mcc }

    # save missclassified images to file together with class
    for class_name in missclassified:
        log_misclassifications( missclassified[class_name], class_name )

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

        class_metrics = { key: value[class_index] for key, value in metrics.items() }

        # write class summary to results file
        log_class_results( class_metrics, class_name, class_index)

    # add evaluation files to experiment
    experiment.add_artifact( '../tmp/split_evaluation_summary.txt' )
    experiment.add_artifact( '../tmp/class_misclassifications.txt' )
    experiment.add_artifact( '../tmp/class_results.txt' )

    # return evaluation metrics
    return {
        'f1': np.mean(f1),
        'rec': np.mean(rec),
        'acc': np.mean(acc),
        'prec': np.mean(prec),
        'spec': np.mean(spec),
        'mcc': np.mean(mcc)
    }

            