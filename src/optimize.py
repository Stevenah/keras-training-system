import keras.backend as K

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

from utils.util import prepare_dataset, split_data

from train import train
from evaluate import evaluate

import tensorflow as tf

import GPy
import GPyOpt
import functools
import json
import os
import importlib

# reset tensorflow graph
tf.reset_default_graph()

# remove unnecessary tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set dimension ordering to tensorflow
K.set_image_dim_ordering('tf')

# set bounds for hyperparameter optimization
bounds = [
    { 'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0, 0.0006) },
    { 'name': 'freeze_layers', 'type': 'discrete', 'domain': range(0, 594) }
]

def optimize( config, config_path=None ):

    if config_path is not None:
        experiment.add_artifact( config_path )

    # optimize hyperparameters
    op_function = functools.partial(run_model, config=config, experiment=experiment)
    hyperparameter_optimizer = GPyOpt.methods.BayesianOptimization( f=op_function, domain=bounds )
    hyperparameter_optimizer.run_optimization( max_iter=10 )

    # print best hyperparameters and achieved score
    for index, param in enumerate(bounds):
        print(f'{ bounds[index]["name"] }: { hyperparameter_optimizer.x_opt[index] }')

    print(f'optimized score: { hyperparameter_optimizer.fx_opt }')
    
def run_model( params, config, experiment ):

    file_identifier = []

    for index, param in enumerate(params[0]):
        print(f'{ bounds[index]["name"] } => { param }')
        file_identifier.extend((bounds[index]['name'], str(param)))
    
    file_identifier = '_'.join(file_identifier)

    # load dataset configuration
    if config['dataset'].get('link', True):
        dataset_config_path = f'../configs/datasets/{ config["dataset"]["link"] }'
        experiment.add_artifact(dataset_config_path)
        config['dataset'].update( json.load( open( dataset_config_path ) ) )

    # split dataset into k split
    split_dirs = split_data(config['dataset']['split'], config['dataset']['path'])
    
    # build training and validation directory
    training_directory, validation_directory = prepare_dataset(split_dirs, 0, len(split_dirs))

    # optionally load pre-trainied model
    if config['model'].get('load_model', False):
        model = load_model(config['model']['load_model'])
    else: 
        model_builder = importlib.import_module(f'models.{ config["model"]["build_file"] }')
        model = model_builder.build(config)

    # modify config to use optimization parameters
    config['hyper_parameters']['learning_rate'] = float(params[:, 0])
    config['hyper_parameters']['freeze_layers'] = int(params[:, 1])

    # train model using optimized hyper parameters
    model = train(model, config, experiment, training_directory, validation_directory, file_identifier)

    # evalaute model using standard metrics
    evaluation = evaluate(model, config, experiment, validation_directory, file_identifier)

    return evaluation['f1']

if __name__ == '__main__':

    # list configs in active directory
    configs = os.listdir( '../configs/active' )

    # iterate over each config and perform experiment
    for config_file in configs:

        # set config path
        config_path = f'../configs/active/{ config_file }'

        # load config file
        config = json.load( open( config_path ) )

        # get experiment path
        experiment_name = config['experiment']['name']
        experiment_path = f'../experiments/{ experiment_name }'

        # initialize experiment
        experiment = Experiment(experiment_name)
        experiment.captured_out_filter = apply_backspaces_and_linefeeds
        experiment.observers.append(FileStorageObserver.create(experiment_path))

        def wrapper():
            return optimize(config, config_path)

        # run experiment
        experiment.automain(wrapper)
