import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

"""Odmah po importu se izvrsi ovo sve sto je u telu"""

# da je np.random.seed(0) uvek bi random davao iste brojeve...
np.random.seed(1)
numberEpochs = 10
numberOutput = 10
n_classes = 10


def data_preparation(X_train, Y_train, X_test, Y_test):
        width = X_train[0].shape[0]
        height = X_train[0].shape[1]
                                  # br. ulaznih pod.
        X_train = X_train.reshape(X_train.shape[0], height, width, 1)       # mozda ce se menjati nacin reshape-a
        X_test = X_test.reshape(X_test.shape[0], height, width, 1)

        # X_train = X_train.reshape(X_train.shape[0], 784)  # mozda ce se menjati nacin reshape-a
        # X_test = X_test.reshape(X_test.shape[0], 784)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255
        X_test = X_test / 255

        Y_train = np_utils.to_categorical(Y_train, n_classes)  # binarna matricna reprezentacija
        Y_test = np_utils.to_categorical(Y_test, n_classes)

        return (X_train, Y_train), (X_test, Y_test)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

width = x_train[0].shape[0]
height = x_train[0].shape[1]

(x_train, y_train), (x_test, y_test) = data_preparation(x_train, y_train, x_test, y_test)


def add_layers(model, shape):
    # prvi sloj
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # drugi sloj
    model.add(Conv2D(filters=16, kernel_size=(3, 3)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))        # Dropout consists in randomly setting a fraction rate of input units to 0 at each
                                    # update during training time, which helps prevent overfitting
    model.add(Flatten())            # "zaravniti", Flattens the input. Does not affect the batch size.

    # model.add(Dense(output_dim=128))
    # novije
    model.add(Dense(units=128))
    model.add(Activation('relu'))
    # model.add(Dense(output_dim=64))
    model.add(Dense(units=64))
    model.add(Activation('relu'))
    model.add(Dense(numberOutput))
    model.add(Activation('softmax'))

    return model


def init_method():
    # "channels_last"(default) assumes (rows, cols, channels) while "channels_first" assumes (channels, rows, cols)
    # if K.image_data_format() == 'channels_first':
    #     shape = (1, sirina, visina)
    # else:
    #     shape = (sirina, visina, 1)

    # shape = (1, width, height)
    shape = (width, height, 1)  # default channels_last

    model = Sequential()
    model = add_layers(model, shape)
    print(model.summary())              # toString modela....
    return model


def get_model():
    model = init_method()
    model.load_weights("neural_network.h5")
    return model


def train_model():
    model = init_method()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=128, epochs=numberEpochs,
                        validation_data=(x_test, y_test))
    rez = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (rez[1] * 100))
    model.save_weights("neural_network.h5")

    return model

# laksa nm
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# # normalizing the data to help with the training
# X_train /= 255
# X_test /= 255
#
# n_classes = 10
# Y_train = np_utils.to_categorical(y_train, n_classes)
# Y_test = np_utils.to_categorical(y_test, n_classes)


# building a linear stack of layers with the sequential model
# model = Sequential()
# model.add(Dense(512, input_shape=(784,), activation="relu"))
# # model.add(Activation('relu'))
# model.add(Dropout(0.2))
#
# model.add(Dense(512, activation="relu"))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
#
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
#
# model.add(Dense(10))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#
# # training the model and saving metrics in history
# history = model.fit(x_train, x_train,
#                     batch_size=128, epochs=10,
#                     verbose=2,
#                     validation_data=(x_test, y_test))
#
# model.save('model.h5')
