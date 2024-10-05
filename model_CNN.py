import keras
import pandas as pd
import numpy as np

from keras import Sequential, layers, models, callbacks, losses, optimizers, metrics

input_size = (64, 64, 1)

model = keras.Sequential(
    [
    keras.layers.Input(shape = input_size),
    keras.layers.Conv2D(32, kernel_size = (3,3), activation = "relu"),
    keras.layers.Conv2D(32, kernel_size = (3,3), activation = "relu"),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(16, kernel_size = (3,3), activation = "relu"),
    keras.layers.Conv2D(16, kernel_size = (3,3), activation = "relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation = "softmax")
]
)

model.summary()

model.compile(
    loss = keras.losses.CategoricalCrossentropy(),
    optimizer= keras.optimizers.Adam(),
    metrics = [
                  keras.metrics.CategoricalAccuracy(name = 'categ'), 'accuracy'
              ]
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_new = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_new = ImageDataGenerator(rescale=1./255)

training = train_new.flow_from_directory('C:/Users/JYOTI/Downloads/DEEP_LEARNING/sign_language/data/train',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

testing = test_new.flow_from_directory('C:/Users/JYOTI/Downloads/DEEP_LEARNING/sign_language/data/test',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical')

model.fit(
        training,
        steps_per_epoch=210, #number of images in train
        epochs=10, 
        validation_data=testing,
        validation_steps=90 #number of images in test
        )

#SAVE THE MODEL

#to save architecture
model_json = model.to_json()
with open("model_json","w") as json_file:
    json_file.write(model_json)

#to save weights
model.save_weights("model_json.weights.h5")