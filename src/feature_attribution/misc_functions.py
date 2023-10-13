#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Misc helper functions """

import cv2
import numpy as np
import subprocess

import torch
import torchvision.transforms as transforms


class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())


def create_folder(folder_name):
    try:
        subprocess.call(['mkdir','-p',folder_name])
    except OSError:
        None


def save_saliency_map_no_image(image, saliency_map, filename, binaryMask=False, threshold=0.5):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension

    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    # saliency_map = saliency_map.clip(0,1)

    if binaryMask==True:
        # Binarize the saliency map
        # threshold_value = np.percentile(saliency_map, 100 - threshold*100)
        saliency_map = np.where(saliency_map > threshold, 1, 0)
    
    saliency_map = saliency_map.clip(0,1)

    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224,224))


    image = np.uint8(image * 255).transpose(1,2,0)
    image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    if binaryMask==True:
        colorMap = cv2.COLORMAP_WINTER 
    else:
        colorMap = cv2.COLORMAP_JET 
    color_heatmap = cv2.applyColorMap(saliency_map, colorMap)
    
    # Combine image with heatmap
    # img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    # img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    color_heatmap = color_heatmap / np.max(color_heatmap)


    cv2.imwrite(filename, np.uint8(255 * color_heatmap))


def save_image(image, saliency_map, filename, binaryMask=False, threshold=0.5):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension

    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)

    if binaryMask==True:
        # Binarize the saliency map
        # threshold_value = np.percentile(saliency_map, 100 - threshold*100)
        saliency_map = np.where(saliency_map > threshold, 1, 0)

    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224,224))

    image = np.uint8(image * 255).transpose(1,2,0)
    image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET )
    
    # Combine image with heatmap
    img_with_heatmap =  np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    # color_heatmap = color_heatmap / np.max(color_heatmap)


    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))

def save_saliency_map(image, saliency_map, filename, binaryMask=False, threshold=0.5):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension

    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)

    if binaryMask==True:
        # Binarize the saliency map
        # threshold_value = np.percentile(saliency_map, 100 - threshold*100)
        saliency_map = np.where(saliency_map > threshold, 1, 0)

    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224,224))

    image = np.uint8(image * 255).transpose(1,2,0)
    image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    if binaryMask==True:
        colorMap = cv2.COLORMAP_WINTER 
    else:
        colorMap = cv2.COLORMAP_JET 
    color_heatmap = cv2.applyColorMap(saliency_map, colorMap)
    
    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    # color_heatmap = color_heatmap / np.max(color_heatmap)


    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))
