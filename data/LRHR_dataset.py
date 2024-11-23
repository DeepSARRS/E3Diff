from io import BytesIO
import lmdb
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import random
import data.util as Util
import os
import cv2
import numpy as np
import torchvision
import torch





class SAR2EODataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False, opt=None):

    
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR  
        self.split = split
        self.dataroot = dataroot
        self.type = opt['name']
        print(f'===================loading {self.type} dataset===================')
        if self.type == 'SAR_EO':
            self.ppb = True  # needs canny or ppb denoising imgs nor not
            self.canny = True
            self.sr_path = Util.get_paths_from_images(os.path.join(dataroot, 'SAR')) 
            self.hr_path = Util.get_paths_from_images(os.path.join(dataroot, 'EO'))
        elif self.type == 'SEN12':
             
            self.ppb = False  # # needs ppb denoising imgs or not
            self.canny = False
            self.source_dir = os.path.join(dataroot, 'sen1')
            self.A_paths = Util.get_paths_from_images(self.source_dir)

            self.target_dir = os.path.join(dataroot, 'sen2')
            self.B_paths = Util.get_paths_from_images(self.target_dir)

            self.sr_path = sorted([i for i in self.A_paths if i.replace('sen1','sen2') in self.B_paths])
            self.hr_path = [i.replace('sen1','sen2') for i in self.sr_path]
 
        
 
        if self.need_LR:
            self.lr_path = Util.get_paths_from_images(
                '{}/lr_{}'.format(dataroot, l_resolution))
        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        print(f'=================== datalen {self.data_len}/{self.dataset_len} ===================')


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_EO = None
        img_LR = None
        img_ppb = None
        img_canny = None
        img_EO = Image.open(self.hr_path[index]).convert("RGB")  # 0-255
        if self.type=='SAR_EO':
            filename = os.path.basename(self.hr_path[index])
            sarpath = os.path.join(self.dataroot, 'SAR', filename)
            ch = 1
            ppbpath = os.path.join(self.dataroot, 'SAR-PPB', filename)
            cannypath = os.path.join(self.dataroot, 'SAR-canny', filename)

        elif self.type=='SEN12':
            opt_path = self.hr_path[index]
            sarpath =  opt_path.replace('sen2', 'sen1')
            filename = os.path.basename(os.path.dirname(sarpath))+'_'+os.path.basename(sarpath)
            ch=3
            ppbpath = sarpath.replace('sen1','sen1-ppb')
        
        img_SAR = Image.open(sarpath).convert("RGB")
        filename = os.path.basename(self.hr_path[index])
        
 
        if self.ppb or self.canny:    
            if self.type =='SAR_EO':        
                img_ppb = Image.open(ppbpath).convert("RGB")  # 0-255
                img_canny = Image.open(cannypath).convert("RGB")
                [img_SAR, img_ppb, img_EO, img_canny] = Util.transform_augment(
                            [img_SAR, img_ppb, img_EO, img_canny], split=self.split, min_max=(-1, 1))
                img_condition = np.concatenate((img_ppb[0:1], img_canny[0:1]), axis=0)  # condition
            elif self.type =='SEN12': 
                img_ppb = Image.open(ppbpath).convert("RGB")  # 0-255
                [img_SAR, img_ppb, img_EO] = Util.transform_augment(
                            [img_SAR, img_ppb, img_EO], split=self.split, min_max=(-1, 1))
                img_condition = img_ppb
        else:
            [img_SAR, img_EO] = Util.transform_augment(
                        [img_SAR, img_EO], split=self.split, min_max=(-1, 1))
            img_condition = img_SAR
            
 
        return {'HR': img_EO[0:ch], 'LR': img_SAR[0:ch], 'SR': img_condition, 'Index': index, 'filename':filename}

            



def random_blur(image, max_radius=5):
     
    radius = random.randint(1, max_radius)
    
     
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
    
    return blurred_image


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)

 