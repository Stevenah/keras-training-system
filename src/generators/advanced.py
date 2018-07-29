from keras.preprocessing.image import ImageDataGenerator

train_data_generator = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False, 
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
    
validation_data_generator = ImageDataGenerator(
    rescale=1./255)