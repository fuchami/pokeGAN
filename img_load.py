# coding:utf-8
"""
jpeg画像を読み込んでネットワークに投げるまでの作業
"""
from keras.preprocessing import image
import sys,os
import glob
import numpy as np

class Load_Img_Data():
    
    def __init__(self):
        
        self.X_train = []

    def load(self, master_path):

        g_path_list = glob.glob(master_path)

        for g_path in g_path_list:
            
            poke_path_list = glob.glob(g_path)
            for poke_path in poke_path_list:
                
                img = image.load_img(poke_path, target_size=(64,64))
                imgSrc = image.img_to_array(img)

                self.X_train.append(imgSrc)

        return self.X_train            
                