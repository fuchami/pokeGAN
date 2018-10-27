# coding:utf-8

from keras.datasets import mnist
from keras.preprocessing import image
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob
import argparse
import model

class GAN():
    def __init__(self, args):
        # dataset params
        self.master_path = args.datasetpath
        # define image size and channels
        self.img_rows = args.imgsize
        self.img_cols = args.imgsize
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # 潜在変数の次元数
        self.z_dim = args.zdims
        self.batch_size = args.batchsize
        self.epochs = args.epochs
        self.saveinterval = args.saveinterval
        # define opt params
        discriminator_optimizer = Adam(lr=1e-5, beta_1=0.1)
        combined_optimizer = Adam(lr=2e-4, beta_1=0.5)

        """ ensure directory """
        if not os.path.exists('./images/'):
            os.makedirs('./images/')
        if not os.path.exists('./saved_model/'):
            os.makedirs('./saved_model/')
        if not os.path.exists('./result_image/'):
            os.makedirs('./result_image/')

        # discriminator model
        self.discriminator = model.build_discriminator()
        plot_model(self.discriminator, to_file='./images/model/discriminator.png', show_shapes=True)
        self.discriminator.compile(loss= 'binary_crossentropy',
            optimizer=discriminator_optimizer,
            metrics=['accuracy'])

        # generator model
        self.generator = model.build_generator()
        plot_model(self.generator, to_file='./images/model/generator.png', show_shapes=True)
        #Generator単体では学習を行わないのでコンパイル不要

        self.combined = model.build_combined(self.z_dim, self.generator, self.discriminator)
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

    def train(self):

        X_train = self.load_img_data() 
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=0)

        half_batch = int(self.batch_size /2)

        num_batches = int(X_train.shape[0] / half_batch)
        print('number of batches:', num_batches)

        for epoch in range(self.epochs):
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
                noise = np.random.normal(-1,1, (self.batch_size, self.z_dim))

                # 生成データの正解ラベルは本物(1)
                valid_y = np.array([1] * self.batch_size)
                g_loss = self.combined.train_on_batch(noise, valid_y)
                
                print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))
                if iteration % self.save_interval == 0:
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
        fig.savefig('./result_image/pokemon_%d_%d.png' % (epoch,iteration))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pokemon generate model using DCGAN')
    parser.add_argument('--datasetpath', '-p', type=str, required=True)
    parser.add_argument('--imgsize', '-s', default=64)
    parser.add_argument('--epochs', '-e', default=30000)
    parser.add_argument('--channels', '-c', default=3)
    parser.add_argument('--zdims', '-d', default=100)
    parser.add_argument('--batchsize', '-b', default=64)
    parser.add_argument('--saveinterval', '-i', default=100)
    parser.add_argument('--line_token', '-l', type=str, required=False)

    args = parser.parse_args()

    gan = GAN(args)
    
    gan.train()