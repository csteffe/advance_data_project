import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


######################################################################
# This python code is inspired from:
# https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/13-%20Implementing%20a%20neural%20network%20for%20music%20genre%20classification/code/mlp_genre_classifier.py
#
######################################################################

tf.random.set_seed(123)

df1 = pd.read_csv('Data/features_30_sec.csv')
df1.head()

genre_list = df1.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
df1 = df1.drop(labels='filename',axis=1)
#print(y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(np.array(df1.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from keras.models import Sequential


from keras.callbacks import ReduceLROnPlateau,EarlyStopping, History

# Neural network

hist = History()

model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

lr=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,verbose=1)
es=EarlyStopping(monitor='val_loss',patience=20,verbose=1)

classifier = model.fit(X_train,
                    y_train,
                    epochs=400,
                    batch_size=128,
                    validation_split= 0.2,
                       callbacks = [hist,lr])


plt.plot(hist.history['accuracy'], color = 'red', label="train accuracy")
plt.plot(hist.history['val_accuracy'], color = 'blue', label="validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('neuralnet.png')


test_loss, test_acc  = model.evaluate(X_test, y_test, batch_size=128)


#y_predicted = model.predict(X_test,batch_size=128)

#print(confusion_matrix(y_true=y_test, y_pred=y_predicted))