import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import random

random.seed(50)


from keras.callbacks import ReduceLROnPlateau,EarlyStopping, History

hist = History()

#train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        #rescale=1./255)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split = 0.2)

train_generator = test_datagen.flow_from_directory(
        'data/images_original',
        target_size=(200,200),
        batch_size=20,
        seed=50,
        subset = "training")

validation_generator = test_datagen.flow_from_directory(
        'data/images_original',
        target_size=(200,200),
        seed=50,
        batch_size=20,
        subset = "validation")

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(8, (5, 5), activation='relu', input_shape=(200,200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(8, (5, 5), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

lr=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001,
                           patience = 20, mode = 'min', verbose = 1,
                           restore_best_weights = True)
from tensorflow.keras.optimizers import RMSprop



model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'
),
              metrics=['accuracy'])






model.fit_generator(
      train_generator,
      steps_per_epoch=8,
      epochs=160,
      validation_data = validation_generator,
      validation_steps = 8,
      verbose=1,
      callbacks = [hist,lr,early_stop])

plt.plot(hist.history['accuracy'], color = 'red', label="train accuracy")
plt.plot(hist.history['val_accuracy'], color = 'blue', label="validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('CNN2D.png')

