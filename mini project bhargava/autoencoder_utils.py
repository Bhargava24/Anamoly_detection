import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.models import Model
import os

def build_autoencoder():
    input_img = Input(shape=(224, 224, 3))
    
    # Encoder
    x = Flatten()(input_img)
    encoded = Dense(1024, activation='relu')(x)
    encoded = Dense(512, activation='relu')(encoded)
    encoded = Dense(256, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(512, activation='relu')(encoded)
    decoded = Dense(1024, activation='relu')(decoded)
    decoded = Dense(224 * 224 * 3, activation='sigmoid')(decoded)
    decoded = Reshape((224, 224, 3))(decoded)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def train_autoencoder(autoencoder, x_train, x_test):
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

def save_autoencoder_weights(autoencoder, filepath='autoencoder_weights.h5'):
    autoencoder.save_weights(filepath)

def load_autoencoder_weights(autoencoder, filepath='autoencoder_weights.h5'):
    if os.path.exists(filepath):
        autoencoder.load_weights(filepath)
    else:
        print("Weights file not found. Training may be required.")

def preprocess_mnist():
    from tensorflow.keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))
    return x_train, x_test
