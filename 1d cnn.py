#%%
import os
import pickle
import pathlib
import librosa
import keras
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import random

random.seed(50)

######################################################################
# This python code is inspired from:
#https://github.com/krishmhatre/Music-Genre-Recognition/blob/main/genre_recognition.ipynb
######################################################################

data_dir = 'genres'
genres = np.array(tf.io.gfile.listdir(str(data_dir)))
print(genres)


filenames = tf.io.gfile.glob(str(data_dir) + "/*/*")
np.random.shuffle(filenames)


def get_max_length(arr):
  max = 0
  for i in arr:
    if len(i) > max:
      max = len(i)
  return max

train_data_size = 1000
INPUT_SHAPE = (train_data_size, 1025, 1320)
MAX_LENGTH = 675808

def get_features_and_target(file_path):
  target = str(file_path).split(os.path.sep)[-2]
  audio_binary = tf.io.read_file(file_path)
  audio, _ = tf.audio.decode_wav(audio_binary)
  features = np.array(audio).reshape(len(audio))
  return features, target

def equal_length(waveform):
  zero_padding = np.zeros(MAX_LENGTH - np.shape(waveform)[0], dtype=np.float32)
  waveform = np.asarray(waveform, np.float32)
  equal_length = np.concatenate((waveform, zero_padding), 0)
  return equal_length

def get_data(filenames):
  X = np.zeros(INPUT_SHAPE)
  Y = []
  i = 0
  for f_name in filenames:
    features, target = get_features_and_target(f_name)
    X[i] = librosa.amplitude_to_db(np.abs(librosa.stft( np.array(equal_length(features), dtype=np.float32))))
    Y.append(target)
    i += 1
    print(i)
  return X, np.array(Y)


X, Y = get_data(filenames[:train_data_size])

y_encoder = LabelEncoder()
y_encoder.fit(Y)

Y = y_encoder.transform(Y)

X = X / np.max(np.abs(X))


model = keras.models.Sequential()
model.add(keras.layers.Input(shape=INPUT_SHAPE[1:]))

model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=512, activation='relu'))
model.add(keras.layers.Dense(units=512, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax'))



model.summary()

lr=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001,
                           patience = 20, mode = 'min', verbose = 1,
                           restore_best_weights = True)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, Y, epochs=400, validation_split=0.2, callbacks = [lr,early_stop])



plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.title('1D-CNN')
plt.savefig('1D-CNN.png')
plt.show()
