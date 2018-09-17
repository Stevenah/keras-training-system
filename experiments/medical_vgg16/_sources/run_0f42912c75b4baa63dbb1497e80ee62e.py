from keras import backend as K
from keras.models import load_model

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

from utils.util import prepare_dataset, split_data, ModelHelper
from utils.logging import *

from train import train
from evaluate import evaluate

import tensorflow as tf

import os
import json
import shutil
import importlib

# initialize globals
config = None
config_path = None
experiment_name = None
experiment_path = None

# file paths
full_kfold_summary_file_path = 'kfold_summary.txt'
all_results_file_path = 'all_results.txt'

# reset tensorflow graph
tf.reset_default_graph()

# remove unnecessary tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set dimension ordering to tensorflow
K.set_image_dim_ordering('tf')

# main run function
def run():
    
    # declare global variables
    global config
    global config_path

    # add config file to experiment
    experiment.add_artifact(config_path)

    if config['dataset'].get('link', True):
        dataset_config_path = f'../configs/datasets/{config["dataset"]["link"]}'
        experiment.add_artifact(dataset_config_path)
        config['dataset'].update( json.load( open( dataset_config_path ) ) )

    # import model builder
    model_builder_path = config['model']['build_file']
    model_builder = importlib.import_module(f'models.{model_builder_path}')

    # image dimensions for training and validation 
    image_width = config['image_processing']['image_width']
    image_height = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']
    
    # numer of classes in dataset
    number_of_classes = config['dataset']['number_of_classes']

    # dataset specific variables
    folds = config['dataset']['split']
    data_directory = config['dataset']['path']
        
    # split dataset into k folds
    split_dirs = split_data(folds, data_directory)

    # total results dictionary
    results = {
        'f1': [],
        'rec': [],
        'acc': [],
        'mcc': [],
        'prec': [],
        'spec': []
    }

    # iterate over each dataset split
    for split_index in range(len(split_dirs)):
        
        # print current validation split index
        print(f'start validating on split {split_index}')
        
        # restart keras session
        K.clear_session() 

        # prepare dataset by distributing the k splits
        # into training and validation sets
        training_directory, validation_directory = prepare_dataset(split_dirs, split_index, len(split_dirs))

        if config['dataset'].get('validation_extension', False):

            extension_path = config['dataset']['validation_extension_path']

            for class_extension in os.listdir(extension_path):

                class_path = os.path.join(extension_path, class_extension)
                target_path = os.path.join(validation_directory, class_extension)

                for filename in os.listdir(class_path):
                    shutil.copy(os.path.join(class_path, filename), os.path.join(target_path, filename))

        # print training directories for sanity
        print(f'training on {training_directory}')
        print(f'validation on {validation_directory}')
        
        # load model from model file or build it using a build file.
        if config['model'].get('load_model', False):
            model = load_model(config['model']['model_splits'][split_index])
        else: 
            model = model_builder.build(config)

        # train model and get last weigths
        if config['model'].get('train', True):
            print("Start training...")
            model = train(model, config, experiment, training_directory, validation_directory, f'split_{split_index}')

        # if fine tune, train model again on config link found in config
        if config['fine_tuning']['enabled'] and config['model'].get('train', True): 
            
            print("Start fine tuning...")

            # load config link from config
            fine_tuning_config_name = config['fine_tuning']['link']
            fine_tuning_config_path = f'../configs/links/{fine_tuning_config_name}'
            fine_tuning_config = json.load(open(fine_tuning_config_path))

            if fine_tuning_config['dataset'].get('link', True):
                dataset_config_path = f'../configs/datasets/{fine_tuning_config["dataset"]["link"]}'
                experiment.add_artifact( dataset_config_path )
                fine_tuning_config['dataset'].update( json.load(open( dataset_config_path ) ) )

            # add link config to experiment
            experiment.add_artifact(fine_tuning_config_path)

            # train using new config
            model = train(model, fine_tuning_config, experiment, training_directory, validation_directory, f'fine_split_{split_index}') 

        # wrap model in model helper before evaluation for ease of image preprocessing
        model_helper = ModelHelper(model, number_of_classes, image_width, image_height, image_channels)

        # evaluate train model and get metrics
        print("Start evaluation...")
        split_results = evaluate(model_helper, config, validation_directory, experiment, f'split_{split_index}') 

        # merge split results with total results
        for key in split_results:
            results[key].append(split_results[key])
            print(key, results[key])

    # log results
    log_cross_validation_results(full_kfold_summary_file_path, results, experiment_name, folds)
    log_to_results_comparison( results, experiment_name, folds)

    experiment.add_artifact(full_kfold_summary_file_path)
    experiment.add_artifact(all_results_file_path)

if __name__ == '__main__':
    
    # list configs in active directory
    configs = os.listdir( '../configs/active' )

    # iterate over each config and perform experiment
    for config_file in configs:

        # set config path
        config_path = f'../configs/active/{config_file}'

        # load config file
        config = json.load(open(config_path))

        # get experiment path
        experiment_name = config['experiment']['name']
        experiment_path = f'../experiments/{experiment_name}'

        # initialize experiment
        experiment = Experiment(experiment_name)
        experiment.captured_out_filter = apply_backspaces_and_linefeeds
        experiment.observers.append(FileStorageObserver.create(experiment_path))

        # run experiment
        experiment.automain(run)
