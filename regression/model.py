
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

def build_model():
  model = keras.models.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]
# print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
# print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features

# column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                # 'TAX', 'PTRATIO', 'B', 'LSTAT']

# df = pd.DataFrame(train_data, columns=column_names)
# print(df.head())

# print(train_labels[0:10])  # Display first 10 entries

mean = train_data.mean(axis=0)
# print("mean: ",mean)
std = train_data.std(axis=0)
# print("std: ",std)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# print(train_data[0])  # First training sample, normalized

model = build_model()
print(model.summary())
