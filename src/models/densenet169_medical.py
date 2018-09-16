from keras.applications.densenet import DenseNet169
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model

def build(config):

    image_width = config['image_processing']['image_width']
    image_height = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    number_of_classes = config['dataset']['number_of_classes']
    model_file = config['model']['model_file']

    base_model = load_model(model_file)

    for index in range(5):
        print(base_model.layers[-index].name)

    x = base_model.layers[-3].output
    x = GlobalAveragePooling2D()(x)
    
    predictions = Dense(
        units=number_of_classes,
        activation='softmax',
        name='predictions')(x)

    return Model(inputs=base_model.input, outputs=predictions)