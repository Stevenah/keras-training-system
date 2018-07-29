from keras.optimizers import * 

# build keras optimizer from given params config
def build_optimizer(config):

    # name of optimizer, e.g., nadam, sgd, etc.
    name = config['name']
    
    # params for Keras optimizers
    params = config['params']

    if name == 'nadam':
        return Nadam(**params)

    if name == 'sgd':
        return SGD(**params)

    if name == 'adam':
        return Adam(**params)
