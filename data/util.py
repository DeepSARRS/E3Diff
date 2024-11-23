import os
import torch
import torchvision
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


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


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
vflip = torchvision.transforms.RandomVerticalFlip()    
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]   # hwc->chw， normalization
    
    
    if split == 'train':
        num = len(imgs)
        imgs = torch.stack(imgs, 0)    # chw -> nchw
        imgs = hflip(imgs)
        imgs = vflip(imgs)

        random_integer = random.randint(1, 4) 
        imgs = torch.rot90(imgs, random_integer, dims=[2,3])
        if num==4:   # +canny    [img_SAR, img_ppb, img_EO, img_canny]
            imgs[:2, ...] = rand_brightness(imgs[:2, ...], brightness=0.3, maxval=1.)  # [img_SAR, img_ppb, img_EO, ...]
        # else:
        #     imgs[:1, ...] = rand_brightness(imgs[:1, ...], brightness=0.3, maxval=1.)

        imgs = torch.unbind(imgs, dim=0)
    

 
    # [img_SR, img_HR, img_ppb, img_canny]
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]    
    return ret_img




# if self.phase=='train' and self.augment:
#             source, target = paired_data_augmentation(source, target, maxval=255)
 
def rand_brightness(img, brightness=0.4, maxval=255):
    brightness_factor = random.uniform(1.0-brightness, 1.0+brightness)
    img_zero = torch.zeros_like(img)
    img = img_zero * (1.0 - brightness_factor) + img * brightness_factor
    # img = (img / img.max()) * 255.
    # bound = 1 if img.dtype in [np.float32, np.float64] else 255
    img = torch.clip(img, 0, maxval)
    # img = img.astype(np.uint8)
    return img

def paired_data_augmentation(source, target, maxval=255):
    # 0-255
    stack_img = np.concatenate((source, target), axis=-1)   
    random_integer = random.randint(1, 4) 
    stack_rot = np.rot90(stack_img, random_integer, axes=[0,1])
    if random.random()>0.5:
        stack_rot = np.fliplr(stack_rot)        
    if random.random()>0.5:
        stack_rot = np.flipud(stack_rot)     
    source = stack_rot[..., :3]
    target = stack_rot[..., 3:]
 
    source, target = rand_brightness(source, target, maxval=maxval)

    return source, target