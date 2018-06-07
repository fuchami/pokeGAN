"""

 画像の水増しを行うPG

 """
from PIL import Image
import glob 
import cv2

master_path = "/mnt/HDD1/GAN_work/GANN/poke_data/*"
g_path_list = glob.glob(master_path)
cnt = 0

for g_path in g_path_list:

    print (g_path)

    poke_path_list = glob.glob(g_path+'/*')
    for poke_path in poke_path_list:
        
        src = cv2.imread(poke_path, 1)

        # 反転
        hflip_img = cv2.flip(src, 1)

        cv2.imwrite("./poke_data/infla/" +str(cnt) +".jpg", hflip_img )

        cnt+=1
        

                

