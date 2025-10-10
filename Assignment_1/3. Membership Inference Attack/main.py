from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Remove TF XLA warnings
import tensorflow as tf

MODEL_PATH = '../cifar10_cnn_initial_model.keras'
N = 10000  # members / non-members
NUM_CLASSES = 10

model = tf.keras.models.load_model(MODEL_PATH)

from sklearn.metrics import accuracy_score, precision_score, recall_score
loss_object = tf.keras.losses.CategoricalCrossentropy()

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Loading data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_test = x_test.astype("float32") / 255.0
x_train = x_train.astype("float32") / 255.0

# Split the data
x_nonmem, x_member =x_test[:N],  x_train[:N]

# Predictions (model confidence)
p_non_member ,p_member =  model.predict(x_nonmem), model.predict(x_member)

# Get the highest probability score (max softmax output)
s_member, s_nonmem = np.max(p_member, 1), np.max(p_non_member, 1)

# Combine scores and labels
labels = np.concatenate([np.ones(N), np.zeros(N)])
scores = np.concatenate([s_member, s_nonmem])
threshold = np.mean(scores)  # Use mean score to calculate threshold

preds = (scores > threshold).astype(int)

# Evaluation metrics
print("Accuracy:", accuracy_score(labels, preds))
print("Precision:", precision_score(labels, preds, zero_division=0))
print("Recall:", recall_score(labels, preds, zero_division=0))
