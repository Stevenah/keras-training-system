from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16

def build(config):

    base_model = VGG16(
        weights=None,
        include_top=False,
        input_shape=(config['image_processing']['image_width'], config['image_processing']['image_height'], config['image_processing']['image_channels']))

    x = base_model.output
    x = Flatten()(x)

    predictions = Dense(
        units=config['dataset']['number_of_classes'],
        activation='softmax',
        name='predictions')(x)

    return Model(inputs=base_model.input, outputs=predictions)

    
