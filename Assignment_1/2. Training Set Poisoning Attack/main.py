from os import environ, makedirs
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Remove TF XLA warnings
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10

MODEL_PATH = '../cifar10_cnn_initial_model.keras'
NUM_CLASSES = 10
POISON_FRAC=0.20

model = tf.keras.models.load_model(MODEL_PATH)

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0


# simple label-flip poisoning
n=int(len(y_train)*POISON_FRAC)
idx=np.random.choice(len(y_train),n,replace=False)
y_poison=y_train.copy()
for i in idx:
    choices=list(range(NUM_CLASSES))
    choices.remove(int(np.argmax(y_train[i])))
    y_poison[i] = np.zeros(NUM_CLASSES)
    y_poison[i][np.random.choice(choices)] = 1

model.fit(x_train, y_poison, epochs=125, batch_size=64, verbose=1)

pred = model.predict(x_test, verbose=0)

from numpy import argmax

y_pred = argmax(pred, axis=1)
y_true = argmax(y_test, axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

# eval before/after not shown (keeps code tiny) — save some poisoned images
SAVE_DIR = "training_set_poison_results"
makedirs(SAVE_DIR, exist_ok=True)

saved=0
for i in idx[:50]:
    img=np.clip((x_train[i]*255).astype("uint8"),0,255)
    original_label = int(np.argmax(y_train[i]).item())
    poisoned_label = int(np.argmax(y_poison[i]).item())

    Image.fromarray(img).save(f"{SAVE_DIR}/idx{i}_o{original_label}_p{poisoned_label}.png")
    saved+=1
