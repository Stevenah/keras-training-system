from keras import backend as K

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

from utils import prepare_dataset, merge_dict_of_lists
from utils import split_data, save_artifact, ModelHelper

from train import train
from evaluate import evaluate

import tensorflow as tf

import os
import json
import importlib

# initialize globals
config = None
experiment_name = None
experiment_path = None

# reset tensorflow graph
tf.reset_default_graph()

# remove unnecessary tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set dimension ordering to tensorflow
K.set_image_dim_ordering('tf')

# main run function
def run():
    
    global config

    # import model builder
    model_builder = importlib.import_module(f'models.{config["model"]["build_file"]}')

    # table size for print table
    table_size = config['misc']['table_size']

    # get temp path for temporary files
    temp_path = config['misc']['temp_path']

    # kfolds summary filename
    model_summary_file_name = config['summary_files']['model_evaluation_summary']
    model_summary_path = os.path.join(temp_path, model_summary_file_name)

    image_width = config['image_processing']['image_width']
    image_height = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']
    number_of_classes = config['dataset']['number_of_classes']

    # dataset specific variables
    folds = config['dataset']['split']
    data_directory = config['dataset']['path']
    temp_directory = config['dataset']['temp']

    # split dataset into k folds
    split_dirs = split_data(folds, data_directory, temp_directory)

    results = { }

    # total results dictionary
    results['f1']   = [] 
    results['rec']  = []
    results['acc']  = []
    results['mcc']  = []
    results['prec'] = []
    results['spec'] = []

    # iterate over each dataset split
    for split_index in range(len(split_dirs)):
        
        # print current validation split index
        print(f'start validating on split {split_index}')
        
        # restart keras session
        K.clear_session() 

        # prepare dataset by distributing the k splits
        # into training and validation sets
        training_directory, validation_directory = prepare_dataset(split_dirs, temp_directory, split_index, len(split_dirs))

        # print training directories for sanity
        print(f'training on {training_directory}')
        print(f'validation on {validation_directory}')
        
        # build model based on config name
        model = model_builder.build(config)

        # train model and get last weigths
        print("Start training...")
        last_weights = train(model, config, experiment, training_directory, validation_directory, f'split_{split_index}')

        # load best weights based on validation accuracy

        if os.path.isfile(last_weights):
            model.load_weights(last_weights)

        # if fine tune, train model again on config link found 
        # in config
        if config['fine_tuning']['mode'] == 'enabled': 
            
            print("Start fine tuning...")

            # load config link from config
            fine_tuning_config = json.load(open(f'./configs/links/{config["fine_tuning"]["link"]}'))

            # train using new config
            train(model, fine_tuning_config, experiment, training_directory, validation_directory, f'split_{split_index}') 

        model = ModelHelper(model, number_of_classes, image_width, image_height, image_channels)

        # evaluate train model and get metrics
        print("Start evaluation...")
        split_results = evaluate(model, config, validation_directory, experiment, f'split_{split_index}') 

        # merge split results with total results
        merge_dict_of_lists(results, split_results) 

    # write summary file 
    write_model_summary(model_summary_path, results, experiment_name, folds, table_size)
    write_model_summary('all-results', results, experiment_name, folds, table_size, 'a')

    # add kfold summary to experiment
    experiment.add_artifact(model_summary_path)

if __name__ == '__main__':
    
    # list configs in active directory
    configs = os.listdir( './configs/active' )

    # iterate over each config and perform experiment
    for config_file in configs:

        # load config file
        config = json.load(open(f'./configs/active/{config_file}'))

        # get experiment path
        experiment_name = config['experiment']['name']
        experiment_path = f'./experiments/{experiment_name}'

        # initialize experiment
        experiment = Experiment(experiment_name)
        experiment.captured_out_filter = apply_backspaces_and_linefeeds
        experiment.observers.append(FileStorageObserver.create(experiment_path))

        # run experiment
        experiment.automain(run)