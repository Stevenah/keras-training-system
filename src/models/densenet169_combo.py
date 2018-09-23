from keras.applications.densenet import DenseNet169
from keras.layers import Dense, GlobalAveragePooling2D, Average, Input
from keras.models import Model, load_model

def build(config):

    model_input = Input((224, 224, 3))

    models = [
        load_model(model) for model in config['model']['models']
    ]    
    
    averaged = Average()([ model(model_input) for model in models ])

    return Model(inputs=model_input, outputs=averaged, name='ensemble')