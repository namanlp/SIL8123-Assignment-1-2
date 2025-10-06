import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Remove TF XLA warnings
import tensorflow as tf

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
num_classes = 10

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D

train_y = tf.keras.utils.to_categorical(train_y, num_classes)
test_y = tf.keras.utils.to_categorical(test_y, num_classes)

test_x = test_x.astype('float32')
test_x = test_x / 255.0
train_x = train_x.astype('float32')
train_x = train_x / 255.0

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

train_x, X_valid, train_y, y_valid = train_test_split(train_x, train_y, test_size=0.2)

def cifar_10_model():
    cifar10_model = tf.keras.models.Sequential()

    # First Conv layer
    cifar10_model.add(Conv2D(filters=128, activation='relu',
                             kernel_regularizer=regularizers.l2(1e-4), kernel_size=(3, 3),
                             padding='same',
                             input_shape=(32, 32, 3)))
    cifar10_model.add(MaxPooling2D(pool_size=(2, 2)))
    cifar10_model.add(Dropout(0.3))

    # Second Conv layer
    cifar10_model.add(Conv2D(filters=256, activation='relu',
                             kernel_size=(3, 3), kernel_regularizer=regularizers.l2(1e-4),
                             padding='same'))
    cifar10_model.add(MaxPooling2D(pool_size=(2, 2)))
    cifar10_model.add(Dropout(0.3))

    # Third, fourth, fifth convolution layer
    cifar10_model.add(Conv2D(filters=512,  activation='relu',
                             kernel_size=(3, 3), kernel_regularizer=regularizers.l2(1e-4),
                             padding='same'))
    cifar10_model.add(Conv2D(filters=512,  activation='relu',
                             kernel_size=(3, 3), kernel_regularizer=regularizers.l2(1e-4),
                             padding='same'))
    cifar10_model.add(Conv2D(filters=256,  activation='relu',
                             kernel_size=(3, 3), kernel_regularizer=regularizers.l2(1e-4),
                             padding='same'))
    cifar10_model.add(MaxPooling2D(pool_size=(2, 2)))
    cifar10_model.add(Dropout(0.3))

    # Fully Connected layers
    cifar10_model.add(Flatten())

    cifar10_model.add(Dense(512, activation='relu'))
    cifar10_model.add(Dropout(0.5))
    cifar10_model.add(Dense(256, activation='relu'))
    cifar10_model.add(Dropout(0.5))
    cifar10_model.add(Dense(128, activation='relu'))
    cifar10_model.add(Dropout(0.5))
    cifar10_model.add(Dense(10, activation='softmax'))

    return cifar10_model

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=False,
    samplewise_center=False,
    samplewise_std_normalization=False,

    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1)

datagen.fit(train_x)

model = cifar_10_model()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              metrics=['accuracy'])

history = model.fit(datagen.flow(train_x, train_y, batch_size = 64),
                    steps_per_epoch = len(train_x) // 64,
                    epochs = 50,
                    validation_data= (X_valid, y_valid),
                    verbose=1)

pred = model.predict(test_x, verbose=0)

from numpy import argmax

y_pred = argmax(pred, axis=1)
y_true = argmax(test_y, axis=1)

print(classification_report(y_true, y_pred))

model.save('cifar10_cnn.keras')
