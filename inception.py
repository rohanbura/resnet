from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras as keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

conv_base = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten(input_shape = (150,150,3)))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(6, activation =  'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics =  ['accuracy'])

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, validation_split=0.1)

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_set = train_datagen.flow_from_directory(r'E:\rohan\rethink\augmented_data',
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 subset = 'training',
                                                 class_mode = 'categorical')

val_set = train_datagen.flow_from_directory(r'E:\rohan\rethink\augmented_data',
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 subset = 'validation',
                                                 class_mode = 'categorical')

test_ds = test_datagen.flow_from_directory(directory = r'E:\rohan\rethink\.',
                                            batch_size = 7,
                                            seed = 42,
                                            shuffle = False,
                                            classes = ['test'],
                                            target_size = (150,150))

model.fit_generator(train_set,
                    validation_data=val_set,
                    epochs=100,
                    steps_per_epoch=2000,
                    validation_steps=400)

model.save("inception.h5")
print("Saved model to disk")

