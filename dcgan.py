# coding:utf-8

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import sys, glob

class GAN():
    def __init__(self):
        
        self.master_path = '/mnt/HDD1/GAN_work/GANN/poke_data/*'
        
        # 入力サイズデータ
        self.img_rows = 64
        self.img_cols = 64
        #白黒(1chan)
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # 潜在変数の次元数
        self.z_dim = 100

        discriminator_optimizer = Adam(lr=1e-5, beta_1=0.1)
        combined_optimizer = Adam(lr=2e-4, beta_1=0.5)

        # discriminator model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss= 'binary_crossentropy',
            optimizer=discriminator_optimizer,
            metrics=['accuracy'])

        # generator model
        self.generator = self.build_generator()
        #Generator単体では学習を行わないのでコンパイル不要

        self.combined = self.build_combined()
        self.combined.compile(loss='binary_crossentropy', optimizer=combined_optimizer)

        self.X_train = []
    
    def load_img_data(self):
        

        path = self.master_path
        g_path_list = glob.glob(path)
        print (g_path_list)

        X =[]
        cnt = 0

        for g_path in g_path_list:
            
            print (g_path)

            poke_path_list = glob.glob(g_path+'/*')
            for poke_path in poke_path_list:
                
                img = image.load_img(poke_path, target_size=(64,64))
                img = image.img_to_array(img)
                X.append(img)
                
                cnt+=1
        print ("img_files:" ,cnt)
        return (np.array(X))

    def build_generator(self):
        
        noise_shape = (self.z_dim,)
        model = Sequential()


        model.add(Dense(128*16*16, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((16,16,128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())

        model.add(Conv2D(128, kernel_size=3, padding='same'))
        #model.add(Activation('relu'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=3, padding='same'))
        #model.add(Activation('relu'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(3, kernel_size=3, padding='same'))
        model.add(Activation('tanh'))

        model.summary()
        return model

    def build_discriminator(self):
        
        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(ZeroPadding2D(padding=((0,1), (1,0))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(ZeroPadding2D(padding=((0,1), (1,0))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))


        model.add(Flatten())
        model.add(Dense(1))        
        model.add(Activation('sigmoid'))  

        model.summary()

        return model

    def build_combined(self):
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        model = Model(z, valid)
        model.summary()
        return model

    def train(self, epochs, batch_size, save_interval=500):
        """
        # mnist road
        (X_train, _), (_, _) = mnist.load_data()

        # 値を-1 to 1に規格化
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)"""

        X_train = self.load_img_data() 

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=0)

        half_batch = int(batch_size /2)

        num_batches = int(X_train.shape[0] / half_batch)
        print('number of batches:', num_batches)

        for epoch in range(epochs):
            
            for iteration in range(num_batches):

                # taining for Discriminator
                noise = np.random.normal(-1, 1, (half_batch, self.z_dim))
                gen_imgs = self.generator.predict(noise)

                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]

                d_loss_read = self.discriminator.train_on_batch(imgs, np.ones((half_batch,1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

                d_loss = 0.5 * np.add(d_loss_read, d_loss_fake)


                # training for Generator

                noise = np.random.normal(-1,1, (batch_size, self.z_dim))

                # 生成データの正解ラベルは本物(1)
                valid_y = np.array([1] * batch_size)
                g_loss = self.combined.train_on_batch(noise, valid_y)
                
                print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))
                if iteration % save_interval == 0:
                    self.save_imgs(epoch, iteration)
            if epoch % 5000 == 0:
                self.generator.save("./saved_model/dcgan-{}-epoch.h5".format(epoch))

    def save_imgs(self, epoch,iteration):
        r, c = 5,5

        noise = np.random.normal(-1,1, (r*c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # 生成画像を0-1に再スケール
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:, :])
                axs[i,j].axis('off')
                cnt+=1
        fig.savefig('./poke_images/pokemon_%d_%d.png' % (epoch,iteration))
        plt.close()

if __name__ == '__main__':

    master_path = '/mnt/HDD1/GAN_work/pokeGAN/set_Data/'
    gan = GAN()
    
    gan.train(epochs=30000, batch_size=64, save_interval=100)