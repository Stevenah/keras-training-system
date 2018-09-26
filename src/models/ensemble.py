from keras.applications.densenet import DenseNet169
from keras.layers import Dense, GlobalAveragePooling2D, Average, Input
from keras.models import Model, load_model

def build(config):

    image_width = config['image_processing']['image_width']
    image_height = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    model_input = Input((image_width, image_height, image_channels))

    models = config['model']['models']
    models = [ load_model(model) for model in models ]

    for model_index, model in enumerate(models):
        model.name = f'model_{ model_index }'

    models = [ model(model_input) for model in models ]

    averaged = Average()(models)

    return Model(inputs=model_input, outputs=averaged, name='ensemble')