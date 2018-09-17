from keras import backend as K 
from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras.optimizers import Adam

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

from utils.util import prepare_dataset, split_data, ModelHelper

import GPy
import GPyOpt
import functools
import json
import os

import tensorflow as tf
import numpy as np

# reset tensorflow graph
tf.reset_default_graph()

# remove unnecessary tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set dimension ordering to tensorflow
K.set_image_dim_ordering('tf')

def optimize( config ):

    # set bounds for hyperparameter optimization
    bounds = [
        { 'name': 'freeze_layers', 'type': 'discrete', 'domain': range(0, 594) },
        { 'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0, 0.3) }
    ]

    op_function = functools.partial(run_model, config=config, experiment=experiment)

    # build optimizer object
    hyperparameter_optimizer = GPyOpt.methods.BayesianOptimization( f=op_function, domain=bounds )

    # optimize hyperparameters
    hyperparameter_optimizer.run_optimization( max_iter=10 )

def run_model( params, config, experiment ):

    if config['dataset'].get('link', True):
        dataset_config_path = f'../configs/datasets/{config["dataset"]["link"]}'
        experiment.add_artifact(dataset_config_path)
        config['dataset'].update( json.load( open( dataset_config_path ) ) )

    # prepare dataset
    folds = config['dataset']['split']
    data_directory = config['dataset']['path']
    split_dirs = split_data(folds, data_directory)
    training_directory, validation_directory = prepare_dataset(split_dirs, 0, len(split_dirs))

    config['dataset']['training_directory'] = training_directory
    config['dataset']['validation_directory'] = validation_directory

    if config['model'].get('load_model', False):
        model = load_model(config['model']['load_model'])
    else: 
        model_builder_path = config['model']['build_file']
        model_builder = importlib.import_module(f'models.{model_builder_path}')
        model = model_builder.build(config)

    model = train(params, model, config, experiment)
    evaluation = evaluate(model, config, experiment)

    return evaluation['f1']

def train( params, model, config, experiment ):

    validation_directory = config['dataset']['validation_directory']
    training_directory = config['dataset']['training_directory']

    # number of splits and samples in dataset
    validation_split = config['dataset']['split']
    dataset_samples = config['dataset']['samples']

    # dynamic hyperparameters
    freeze_layers = float(params[:, 0])
    learning_rate = float(params[:, 1])

    # static hyperparameters
    epochs = config['hyper_parameters']['epochs']
    patience = config['hyper_parameters']['patience']
    batch_size = config['hyper_parameters']['batch_size']

    # metrics to measure under training
    metrics = [ 'accuracy' ]

    # callbacks to be called after every epoch
    callbacks = [
        ModelCheckpoint('../tmp/weights.h5', monitor='val_acc', verbose=1, save_best_only=True),
        TensorBoard(log_dir=os.path.join('../tmp', 'logs'), batch_size=batch_size)
    ]

    # set image dimensions
    number_of_classes = config['dataset']['number_of_classes']
    image_width = config['image_processing']['image_width']
    image_height = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    # set number of  training and validation samples
    training_samples = dataset_samples - (dataset_samples // validation_split)
    validation_samples = dataset_samples // validation_split

    optimizer = Nadam(
        lr=learning_rate)

    # build data generators
    training_generator_file = config['image_processing']['training_data_generator']
    validation_generator_file = config['image_processing']['validation_data_generator']
    training_data_generator = importlib.import_module(f'generators.{training_generator_file}').train_data_generator
    validation_data_generator = importlib.import_module(f'generators.{validation_generator_file}').validation_data_generator

    # freeze layers based on freeze_layers parameter
    for layer in model.layers[:freeze_layers]:
        layer.trainable = False
    for layer in model.layers[freeze_layers:]:
        layer.trainable = True

    # initialize training generator
    training_generator = training_data_generator.flow_from_directory(
        training_directory, target_size=(image_width, image_height),
        batch_size=batch_size, class_mode='categorical', follow_links=True)

    # initialize validation generator
    validation_generator = validation_data_generator.flow_from_directory(
        validation_directory, target_size=(image_width, image_height),
        batch_size=batch_siaze, class_mode='categorical', follow_links=True)

    # only set early stoppiang if patience is more than 0
    if patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor='val_acc',
                patience=patience,
                verbose=1))

    # print out the class indicies for sanity
    print('train indicies: ', training_generator.class_indices)
    print('validation indicies: ', validation_generator.class_indices)

    # compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=metrics)

    # train model and get training metrics
    history = model.fit_generator(training_generator,
        steps_per_epoch=training_steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks)

    model.load_weights('../tmp/weights.h5')

    return model


def evaluate( model, config ):

    validation_directory = config['dataset']['validation_directory']
    number_of_classes = config['dataset']['number_of_classes']
    
    class_names = get_sub_dirs(validation_directory)
    label_index = { class_name: index for index, class_name in enumerate(class_names) }
    confusion = np.zeros((number_of_classes, number_of_classes))

    for class_name in class_names:
        print(f'Starting {class_name}')
        class_dir = os.path.join(validation_directory, class_name)
        for file_name in os.listdir(class_dir):
            prediction = model.predict_from_path(os.path.join(class_dir, file_name))
            confusion[prediction][label_index[class_name]] += 1

    FP = confusion.sum(axis=0) - np.diag(confusion)  
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)

    f1 = f1score(TP, TN, FP, FN)
    rec = recall(TP, TN, FP, FN)
    acc = accuracy(TP, TN, FP, FN)
    prec = precision(TP, TN, FP, FN)
    spec = specificity(TP, TN, FP, FN)
    mcc = matthews_correlation_coefficient(TP, TN, FP, FN)

    metrics = { 'FP': FP, 'FN': FN, 'TP': TP, 'TN': TN, 'f1': f1, 'rec': rec, 'acc': acc, 'prec': prec, 'spec': spec, 'mcc': mcc }

    return metrics

if __name__ == '__main__':

    # list configs in active directory
    configs = os.listdir( '../configs/optimization' )

    # iterate over each config and perform experiment
    for config_file in configs:

        # set config path
        config_path = f'../configs/optimization/{ config_file }'

        # load config file
        config = json.load(open(config_path))

        # get experiment path
        experiment_name = config['experiment']['name']
        experiment_path = f'../experiments/{ experiment_name }'

        # initialize experiment
        experiment = Experiment(experiment_name)
        experiment.captured_out_filter = apply_backspaces_and_linefeeds
        experiment.observers.append(FileStorageObserver.create(experiment_path))

        def wrapper():
            return optimize(config)

        # run experiment
        experiment.automain(wrapper)
