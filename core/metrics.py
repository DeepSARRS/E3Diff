'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-26 01:51:10
LastEditors: Please set LastEditors
LastEditTime: 2024-11-23 13:46:15
FilePath: /QJ/E3Diff/core/metrics.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import torch

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    bs = tensor.shape[0]
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:

        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB

        
    elif n_dim == 3:
        c = tensor.shape[0]
        if c==1:
            tensor = torch.concat((tensor,tensor,tensor), 0)
            img_np = tensor.numpy()
            img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
        elif tensor.shape[0]>3 or tensor.shape[0]==2:  # grid for single channel
            tensor = tensor.unsqueeze(1).expand(tensor.shape[0],3,tensor.shape[-2],tensor.shape[-1])
            n_img = len(tensor)
            img_np = make_grid(tensor, nrow=int(
                                math.sqrt(n_img)), normalize=False)
            # img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
            img_np = img_np.numpy()
            img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
        else:
            img_np = tensor.numpy()
            img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    
    img_np = np.squeeze(img_np) 
    if len(img_np.shape)==2:
        img_np = np.concatenate((img_np[...,np.newaxis],img_np[...,np.newaxis],img_np[...,np.newaxis]), axis=-1)
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



class cal_metrics():
    def __init__(self, device):
        import lpips
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(device)  # RGB, normalized to [-1,1]
        self.device = device
    

    def cal_fid(self, fakeimg_dir, gtimg_dir, batch_size, dims=2048):
        from pytorch_fid import fid_score
        fid_value = fid_score.calculate_fid_given_paths([fakeimg_dir, gtimg_dir],
                                                            batch_size,
                                                            self.device,
                                                            dims,
                                                            num_workers=0)

        return fid_value
    
    def cal_l2(self, fakeimg, gtimg):
        #  imput:  0-255
        # scale to 0-1
        fakeimg_new = fakeimg/255.
        gtimg_new = gtimg/255.
        l2 = np.mean((fakeimg_new - gtimg_new)**2)
        return l2

    def cal_lpips(self, fakeimg, gtimg):
        # imput: 0-255  array  [h, w, c] / [h, w]
        # loss_fn_vgg = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]
        if len(fakeimg.shape)==2:
            fakeimg = cv2.cvtColor(fakeimg, cv2.COLOR_GRAY2RGB)
        if len(gtimg.shape)==2:
            gtimg = cv2.cvtColor(gtimg, cv2.COLOR_GRAY2RGB)
        
        fakeimg = torch.from_numpy(fakeimg).permute(2, 0, 1).unsqueeze(0).float() / 255.
        gtimg = torch.from_numpy(gtimg).permute(2, 0, 1).unsqueeze(0).float() / 255.

        # calculate lpips
        lpips_val = self.lpips_vgg(fakeimg.to(self.device), gtimg.to(self.device), normalize=True)

        return lpips_val.item()
        

    def calculate_ssim(self, fakeimg, gtimg):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not fakeimg.shape == gtimg.shape:
            raise ValueError('Input images must have the same dimensions.')
        if fakeimg.ndim == 2:
            return ssim(fakeimg, gtimg)
        elif fakeimg.ndim == 3:
            if fakeimg.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(fakeimg, gtimg))
                return np.array(ssims).mean()
            elif fakeimg.shape[2] == 1:
                return ssim(np.squeeze(fakeimg), np.squeeze(gtimg))
        else:
            raise ValueError('Wrong input image dimensions.')
      
    def ssim(self, img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    def calculate_psnr(self, fakeimg, gtimg):
        # img1 and img2 have range [0, 255]
        fakeimg = fakeimg.astype(np.float64)
        gtimg = gtimg.astype(np.float64)
        mse = np.mean((fakeimg - gtimg)**2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))
