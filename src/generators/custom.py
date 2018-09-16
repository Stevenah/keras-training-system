from keras.preprocessing.image import ImageDataGenerator

train_data_generator = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)
    
validation_data_generator = ImageDataGenerator(
    rescale=1./255)