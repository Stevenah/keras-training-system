from keras.applications.nasnet import NASNetLarge
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import importlib

def build(config):

    image_width = config['image_processing']['image_width']
    image_height = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    number_of_classes = config['dataset']['number_of_classes']

    model_file = config['model'].get('model_file', None)
    regularization = config['hyper_parameters'].get('activity_regularizer', None)

    weights = config['model'].get('weights', None)
    print("weights:", weights)
    
    if weights == 'imagenet':
        base_model = NASNetLarge(
            input_shape=(image_width, image_height, image_channels),
            weights='imagenet',
            include_top=False)
    else:
        base_model = NASNetLarge(
            input_shape=(image_width, image_height, image_channels),
            weights=None,
            include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if regularization is not None:
        regularization = getattr(importlib.import_module(f'keras.regularizers'), regularization['name'])
        regularization = regularization(**config['hyper_parameters']['activity_regularizer']['params'])
    
    predictions = Dense(
        activity_regularizer=regularization,
        units=number_of_classes,
        activation='softmax',
        name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if weights != 'imagenet' and weights is not None:
        print(weights, weights is not 'imagenet')
        model.load_weights(weights)

    return model