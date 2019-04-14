from __future__ import division

from keras.models import Model, Sequential
from keras.layers import Input, Dense, multiply, Reshape, Flatten, Dropout, LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D
import numpy as np
from utils import load_mnist
from audio_preprocessing_layer import spectrogram, logSpectrogram, logMelSpectrogram
from encoding_layer_ori import Encoding_layer
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
import tensorflow as tf
from keras.optimizers import Adam
import os
from utils import load_coil_10

class Extractor(object):
    def __init__(self, ):

        self.audio_emb_dim = 128
        self.batch_size = 100
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.classes  = 10
        self.imagine_step = 50

        self.audio_sr = 44100  # audio sampling rate
        self.audio_duration = 1.05  # encoded audio duration
        self.audio_len = 2 * int(0.5 * self.audio_sr * self.audio_duration)

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.optimizer = Adam(0.0002, 0.5)

        self.audio_class_model = self.build_audio_C()



    def build_audio_C(self):

        img_input = Input(batch_shape=(self.batch_size, self.img_rows, self.img_cols, self.channels),
                          name='img_input')

        x = Encoding_layer(name='vOICe')(img_input)
        spectro = logMelSpectrogram(name='logSpectrogram')(x)

        # Block 1
        x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu',  padding='same', name='conv1')(spectro)
        #x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

        # Block 2
        x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu',  padding='same', name='conv2')(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

        # Block 3
        x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

        # Block 4
        x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)


        x = Flatten(name='flatten_')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        embeddings = Dense(self.audio_emb_dim, activation='relu', name='embeddings')(x)
        predicts = Dense(self.classes, activation='softmax', name='prediction')(embeddings)

        audio_model = Model(inputs=img_input, outputs=predicts)

        return audio_model


    def train(self, epochs=10):

        # load the dataset
        (x_train, y_train), (x_test, y_test) = load_coil_10()
        y_train = to_categorical(y_train)
        y_test  = to_categorical(y_test)

        x_train = x_train / 255.0
        x_test  = x_test  / 255.0

        print('training the audio net...')
        self.audio_class_model.compile(loss='categorical_crossentropy',
                                       optimizer=self.optimizer,
                                       metrics=['accuracy'])  # SGD(lr=0.01, momentum=0.9),
        
        self.audio_class_model.fit(x_train, y_train, batch_size=self.batch_size, epochs=epochs)
        self.audio_class_model.save('model/audio_embedding_net_coil.h5')
        print('AudioNet model is saved.')

        print('Extracting audio embeddings...')
        audio_emb_extractor = Model(inputs=self.audio_class_model.input,
                                    outputs=self.audio_class_model.get_layer(name="embeddings").output)
        audio_embeddings = audio_emb_extractor.predict(x_train, batch_size=self.batch_size, verbose=True)
        name = 'audio_embeddings/audio_embedding_coil_train_%d.npy' % epochs
        np.save(name, audio_embeddings)


        audio_emb_extractor = Model(inputs=self.audio_class_model.input,
                                    outputs=self.audio_class_model.get_layer(name="embeddings").output)
        audio_embeddings = audio_emb_extractor.predict(x_test, batch_size=self.batch_size, verbose=True)
        name = 'audio_embeddings/audio_embedding_coil_test_%d.npy' % epochs
        np.save(name, audio_embeddings)

        print('Finished')





if __name__ == '__main__':
    print 'building network ...'
    extractor = Extractor()

    extractor.train(epochs=15)


    # C_Blind



