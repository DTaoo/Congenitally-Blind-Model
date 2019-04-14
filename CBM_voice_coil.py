from keras.models import Model
from keras.layers import Input, Dense, multiply, Reshape, Flatten, Dropout, LeakyReLU, BatchNormalization, Conv2D, \
    MaxPooling2D, Embedding, Concatenate, Activation
from keras.layers.convolutional import UpSampling2D
import numpy as np
from audio_preprocessing_layer import spectrogram, logSpectrogram, logMelSpectrogram
from encoding_layer_ori import Encoding_layer
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from scipy import misc
from PIL import Image
import tensorflow as tf
from keras.optimizers import Adam, SGD
from utils import load_coil_10

class C_Blind(object):
    def __init__(self, ):

        self.audio_emb_dim = 128
        self.noise_dim = 20
        self.batch_size = 100
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1

        self.classes = 10
        self.imagine_step = 200

        self.audio_sr = 22050  # audio sampling rate
        self.audio_duration = 1.05 #1.05  # encoded audio duration
        self.audio_len = 2 * int(0.5 * self.audio_sr * self.audio_duration) # * self.channels

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        losses = ['binary_crossentropy', 'categorical_crossentropy']
        # self.optimizer = SGD(lr=0.01, momentum=0.9) #Adam(0.0002, 0.5)
        self.optimizer = Adam(0.00002, 0.5)

        self.audio_dis_model = self.build_audio_D()
        self.audio_dis_model.compile(loss=losses, optimizer=self.optimizer, metrics=['accuracy'])

        # second, build the image generator
        self.visual_model = self.build_img_G_conv_full()

        # generate the image according to the input audio embedding
        audio_embed = Input(batch_shape=(self.batch_size, self.audio_emb_dim))
        audio_noise = Input(batch_shape=(self.batch_size, self.noise_dim))
        encoded_audio = self.visual_model([audio_embed, audio_noise])
        # encoded_audio = Encoding_layer()(img)

        # fix the audio discriminator
        # self.set_trainability(model=self.audio_dis_model, trainable=False)
        self.audio_dis_model.trainable = False
        valid, target_label = self.audio_dis_model(encoded_audio)

        # build the cross-modal GAN
        self.CM_GAN = Model(inputs=[audio_embed, audio_noise], outputs=[valid, target_label])
        self.CM_GAN.compile(loss=losses, optimizer=self.optimizer, metrics=['accuracy'])


    def build_audio_encoder(self):
        img_input = Input(batch_shape=(self.batch_size, self.img_rows, self.img_cols, self.channels),
                          name='img_input')

        output = Encoding_layer(name='vOICe')(img_input)

        audio_encoder = Model(inputs=img_input, outputs=output)

        return audio_encoder

    def build_audio_D(self):
        # load pretrain weights

        audio_input = Input(batch_shape=(self.batch_size, self.audio_len), name='audio_input')
        spectro = logMelSpectrogram(sample_rate = self.audio_sr)(audio_input)

        # Block 1
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(spectro)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

        # Block 2
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv2')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

        # Block 3
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv3/conv3_1')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv3/conv3_2')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

        # Block 4
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv4/conv4_1')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv4/conv4_2')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)

        fea = Flatten(name='flatten_')(x)
        #x = Dense(4096, name='fc1')(x)
        #x = LeakyReLU(alpha=0.2)(x)
        #x = Dense(self.audio_emb_dim, name='embeddings')(x)
        #fea = LeakyReLU(alpha=0.2)(x)

        valid = Dense(1, activation='sigmoid', name='predictions')(fea)
        labels = Dense(self.classes+1, activation='softmax')(fea)

        audio_D = Model(inputs=audio_input, outputs=[valid, labels])
        return audio_D



    def build_img_G_conv_full(self):
        audio_embeddings = Input(batch_shape=(self.batch_size, self.audio_emb_dim))
        audio_noise = Input(batch_shape=(self.batch_size, self.noise_dim))

        audio_input = Concatenate()([audio_embeddings, audio_noise])
        x = Dense(128 * 8 * 8, activation='relu')(audio_input)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((8, 8, 128))(x)

        #x = UpSampling2D()(x)
        #x = Conv2D(256, kernel_size=5, padding="same")(x)
        #x = BatchNormalization(momentum=0.8)(x)
        #x = Activation("relu")(x)
        #x = LeakyReLU(alpha=0.2)(x)       

        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=5, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)
        #x = LeakyReLU(alpha=0.2)(x)       

        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=5, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)
        #x = LeakyReLU(alpha=0.2)(x)    

        x = UpSampling2D()(x)
        x = Conv2D(32, kernel_size=5, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)

        x = Conv2D(self.channels, kernel_size=3, padding="same")(x)
        img = Activation(activation='sigmoid', name='generated_img')(x)

        encoded_audio = Encoding_layer()(img)

        img_G = Model(inputs=[audio_embeddings, audio_noise], outputs=encoded_audio)

        return img_G


    def imagine_image(self, audio_embeddings, label, epoch):
        noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))

        img_extractor = Model(inputs=self.visual_model.inputs,
                              outputs=self.visual_model.get_layer(name="generated_img").output)
        gen_imgs = img_extractor.predict_on_batch([audio_embeddings, noise])

      
        for i in range(audio_embeddings.shape[0]):
            img_name = 'gen_img_%d_%d_label_%d.jpg' % (epoch, i, label[i])
            x = np.squeeze(gen_imgs[i]) * 255
            
            img = Image.fromarray(x.astype('uint8'))
            img = img.convert('L')
            img.save('gen_coil/' + img_name)


    def test_imagine(self, audio_embeddings, labels, epoch):
        sample_num = audio_embeddings.shape[0]
        #rand_idx = np.random.randint(0, sample_num, self.batch_size)

        #rand_embeddings = audio_embeddings[rand_idx]
        #rand_labels = labels[rand_idx]

        noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))

        img_extractor = Model(inputs=self.visual_model.inputs,
                              outputs=self.visual_model.get_layer(name="generated_img").output)
        gen_imgs = img_extractor.predict_on_batch([audio_embeddings, noise])
      
        for i in range(self.batch_size):
            img_name = 'gen_img_%d_%d_label_%d.jpg' % (epoch, i, labels[i])
            x = np.squeeze(gen_imgs[i]) * 255
            
            img = Image.fromarray(x.astype('uint8'))
            img = img.convert('L')
            img.save('gen_coil/' + img_name)



    def train(self, epochs=100):

        # load the dataset
        (x_train, y_train), (x_test, y_test) = load_coil_10()
        y_train = y_train.reshape(-1, 1)
        y_test  = y_test.reshape(-1,1)       

        x_train = x_train / 255.0

        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        # load the audio embeddings
        name = 'audio_embeddings/audio_embedding_coil_train_15.npy'
        train_embeddings = np.load(name)
 
        name = 'audio_embeddings/audio_embedding_coil_test_15.npy'
        test_embeddings = np.load(name)

        # encoding the img
        audio_encoder = self.build_audio_encoder()
        '''
        # fix the imagine idx
        imagine_idx = np.random.randint(0, x_train.shape[0], 100)
        imagine_imgs = x_train[imagine_idx]
        for i in range(100):
            img_name = 'ori_img_%d_label_%d.jpg' % (i, y_train[imagine_idx[i]])
            data = np.squeeze(imagine_imgs[i]*255)
            im = Image.fromarray(data.astype('uint8'))
            im = im.convert('L')
            im.save('gen_coil/' + img_name)
        imagine_audio_label = y_train[imagine_idx]
        '''

        for epoch in range(epochs):
            # train the discriminator
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))

            sampled_idx    = np.random.randint(0, x_train.shape[0], self.batch_size)
            sampled_labels = y_train[sampled_idx]

            real_labels = y_train[idx]
            fake_labels = 10 * np.ones(real_labels.shape)
            all_labels = to_categorical(np.concatenate((real_labels, fake_labels, sampled_labels)))
            real_labels = all_labels[0:self.batch_size]
            fake_labels = all_labels[self.batch_size:self.batch_size * 2]
            sampled_labels_ = all_labels[self.batch_size * 2:self.batch_size * 3]
            sampled_embedding = train_embeddings[sampled_idx]

            real_audio = audio_encoder.predict_on_batch(imgs)
            fake_audio = self.visual_model.predict_on_batch([sampled_embedding, noise])

            d_loss_real = self.audio_dis_model.train_on_batch(x=real_audio, y=[valid, real_labels])
            d_loss_fake = self.audio_dis_model.train_on_batch(x=fake_audio, y=[fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.CM_GAN.train_on_batch(x=[sampled_embedding, noise], y=[valid, sampled_labels_])
            print("%d [D loss: %.3f, acc: %.2f, class_acc: %.2f] [G loss: %.3f, acc: %.2f, class_acc: %.2f]" % (
            epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0], 100 * g_loss[3], 100 * g_loss[4]))

            if epoch % self.imagine_step == 0:
                self.test_imagine(test_embeddings, y_test, epoch)
                #imagine_audio_embeddings = audio_embeddings[np.squeeze(imagine_audio_label)]
                #self.imagine_image(imagine_audio_embeddings, imagine_audio_label, epoch)


if __name__ == '__main__':
    print 'building network ...'
    c_blind = C_Blind()

    c_blind.train(epochs=10000)




