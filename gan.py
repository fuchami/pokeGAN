# coding:utf-8

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import sys

class GAN():
    def __init__(self):
        # mnist入力サイズデータ
        self.img_rows = 28
        self.img_cols = 28
        #白黒(1chan)
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # 潜在変数の次元数
        self.z_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # discriminator model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss= 'binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # generator model
        self.generator = self.build_generator()
        #Generator単体では学習を行わないのでコンパイル不要

        self.combined = self.build_combined()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        
        noise_shape = (self.z_dim,)
        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape),activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()
        return model

    def build_discriminator(self):
        
        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        return model

    def build_combined(self):
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        model = Model(z, valid)
        model.summary
        return model

    def train(self, epochs, batch_size=128, save_interval = 50):
        
        # mnist road
        (X_train, _), (_, _) = mnist.load_data()

        # 値を-1 to 1に規格化
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size /2)

        num_batches = int(X_train.shape[0] / half_batch)
        print('number of batches:', num_batches)

        for epoch in range(epochs):
            
            for iteration in range(num_batches):

                # taining for Discriminator
                noise = np.random.normal(0, 1, (half_batch, self.z_dim))
                gen_imgs = self.generator.predict(noise)

                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]

                d_loss_read = self.discriminator.train_on_batch(imgs, np.ones((half_batch,1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

                d_loss = 0.5 * np.add(d_loss_read, d_loss_fake)


                # training for Generator

                noise = np.random.normal(0,1, (batch_size, self.z_dim))

                # 生成データの正解ラベルは本物(1)
                valid_y = np.array([1] * batch_size)
                g_loss = self.combined.train_on_batch(noise, valid_y)
                
                print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))
                if iteration % save_interval== 0:
                    self.save_imgs(iteration)
            
    def save_imgs(self, iteration):
        r, c = 5,5

        noise = np.random.normal(0,1, (r*c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # 生成画像を0-1に再スケール
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt+=1
        fig.savefig('./gan_images/mnist_%d.png' % iteration)
        plt.close()

if __name__ == '__main__':
    
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, save_interval=1)

                    


        

