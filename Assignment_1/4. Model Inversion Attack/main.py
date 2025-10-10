from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Remove TF XLA warnings
import tensorflow as tf

MODEL_PATH = '../cifar10_cnn_initial_model.keras'
NUM_CLASSES = 10

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
loss_object = tf.keras.losses.CategoricalCrossentropy()
model = tf.keras.models.load_model(MODEL_PATH)

# --- Evaluate accuracy on real test set ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32")/255.0
y_test = tf.keras.utils.to_categorical(y_test, 10)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Original test accuracy: {acc*100:.2f}%")

from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.inference.model_inversion import MIFace

classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=NUM_CLASSES,
    input_shape=x_test.shape[1:],
    loss_object=loss_object,
    optimizer=adam_optimizer,
    clip_values=(0.0, 1.0),
)

import numpy as np

# --- Model inversion attack using MIFace ---
attack = MIFace(classifier, max_iter=150, batch_size=1)
reconstructed = attack.infer(x=None, y=np.arange(NUM_CLASSES))  # x=None -> start from zeros/random

# --- Predict on reconstructed images ---
preds = classifier.predict(reconstructed)
pred_labels = np.argmax(preds, axis=1)
print("Predicted labels on reconstructed images:", pred_labels)
inv_acc = (pred_labels == np.arange(NUM_CLASSES)).mean()
print(f"Accuracy on reconstructed images: {inv_acc*100:.2f}%")

# --- Display reconstructed images ---
import matplotlib.pyplot as plt
plt.figure(figsize=(20,3))
for i in range(NUM_CLASSES):
    plt.subplot(1,NUM_CLASSES,i+1)
    plt.imshow(reconstructed[i])
    plt.title(f"class {i} pred:{pred_labels[i]}")
    plt.axis('off')
plt.suptitle("ART MIFace Model Inversion Reconstructions (classes 0-9)")
plt.show()
