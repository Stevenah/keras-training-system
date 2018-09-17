from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard

from utils.util import save_artifact
from utils.builders import build_optimizer
from utils.plotters import plot_accuracy, plot_loss

import tensorflow as tf
import importlib
import shutil
import os

# function for training a model given a configuration
def train(model, config, experiment, training_directory, validation_directory, file_identifier):
    
    # extract model and weight file names
    weights_file_name = config['model']['weights_file']
    model_file_name = config['model']['model_file']

    # get file path
    weights_file = f'{file_identifier}_{weights_file_name}'
    model_file = f'{file_identifier}_{model_file_name}'

    # get plot file names
    accuracy_plot_file = f'{file_identifier}_accuracy_plot.png'
    loss_plot_file = f'{file_identifier}_loss_plot.png'

    # plot file paths
    accuracy_plot_path = os.path.join('../tmp', accuracy_plot_file)
    loss_plot_path = os.path.join('../tmp', loss_plot_file)

    # weights/model file paths
    model_path = os.path.join('../tmp', model_file)
    weights_path = os.path.join('../tmp', weights_file)

    # callbacks to be called after every epoch
    callbacks = [
        ModelCheckpoint(weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        TensorBoard(log_dir=os.path.join('../tmp', 'logs'), batch_size=8)
    ]

    # number of splits in dataset
    split = config['dataset']['split']

    # number of samples in dataset
    samples = config['dataset']['samples']

    # metrics to measure under training
    metrics = [ 'accuracy' ] 

    # set number of  training and validation samples
    training_samples   = samples - (samples // split) # number of training samples in dataset
    validation_samples = samples // split # number of validation samples in dataset

    # set number of epochs
    epochs = config['hyper_parameters']['epochs']

    # set batch size
    batch_size = config['hyper_parameters']['batch_size']

    # set number of layers to freeze (not train)
    freeze_layers = config['hyper_parameters']['freeze_layers']

    # set patience (number of epochs where val_acc does not improve before stopping)
    patience = config['hyper_parameters']['patience']

    # number of classes in dataset
    number_of_classes = config['dataset']['number_of_classes']

    # set image dimensions
    image_width = config['image_processing']['image_width'] # change based on the shape/structure of your images
    image_height = config['image_processing']['image_height'] # change based on the shape/structure of your images
    image_channels = config['image_processing']['image_channels'] # number of image channels

    # set training steps based on training sampels and batch size
    training_steps = training_samples // batch_size   # number of training batches in one epoch
    validation_steps = validation_samples // batch_size # number of validation batches in one epoch

    # build optimizer
    optimizer = build_optimizer(config['optimizer'])

    training_generator_file = config['image_processing']['training_data_generator']
    validation_generator_file = config['image_processing']['validation_data_generator']

    # build data generators
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
        batch_size=batch_size, class_mode='categorical', follow_links=True)

    # only set early stopping if patience is more than 0
    if patience > 0:
        
        # append early stopping to callbacks
        callbacks.append(EarlyStopping(monitor='val_acc', patience=patience, verbose=0))

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

    # plot loss
    plot_loss(history, loss_plot_path)

    # plot accuracy
    plot_accuracy(history, accuracy_plot_path)

    # add plots to experiment
    experiment.add_artifact(accuracy_plot_path)
    experiment.add_artifact(loss_plot_path)

    # load best weights
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)

    # save keras model file
    model.save(model_path)

    # add keras model file to experiment
    experiment.add_artifact(model_path)

    # return training history metrics
    return model
