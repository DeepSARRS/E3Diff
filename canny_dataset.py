
import cv2
import  matplotlib.pyplot as plt
import numpy as np
import os


ppb_dir = r'/datatmp/QJ/dataset/SAR-EO/test/SAR-PPB/'
save_dir = r'/datatmp/QJ/dataset/SAR-EO/test/SAR-canny'

os.makedirs(save_dir, exist_ok=True)
imglist = [i for i in os.listdir(ppb_dir)]

for pi in range(len(imglist)):
    print(pi, '/', len(imglist))
    impath = os.path.join(ppb_dir, imglist[pi])
    savepath = os.path.join(save_dir, imglist[pi])

    ppbimg = cv2.imread(impath)

    ppb_cann = cv2.Canny(ppbimg, 50, 150, L2gradient=True)

    cv2.imwrite(savepath, ppb_cann)