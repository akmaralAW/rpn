#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:31:21 2020

@author: vince
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from skimage import io, transform
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.optim as optim

# homemade lib
from treeDataset import tree_dataset_zoom
from visualize import visualize_detection, visualize_sample, visualize_region_proposals, vis_bbox2
from detectionStuff import *
from utils import *
from train_det import train

if __name__ == "__main__":
    # main()
    
    BATCH_SIZE = 1
    
    # path to the dataset
    train_path = '/home/akmaral/akmaral_NorLab/ForestProject/train_set/'
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    
    dataset_train = tree_dataset_zoom(train_path, split='train')
        
    # define training and test data loaders
    train_dl = DataLoader(dataset=dataset_train,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=4,
                      collate_fn=dataset_train.collate_fn, # use custom collate function here
                      pin_memory=True)
    for i, (images, target_boxes, target_gp) in enumerate(train_dl):        
        visualize_sample(images, target_boxes, i) # uncomment when testing regular
        images = images.numpy()
        images = np.squeeze(images, axis = 0)
        target_boxes = bbox_changed(target_boxes)
        vis_bbox2(images, target_boxes)
        if i == 5:
            break
    

# =============================================================================
#     for i, (images, target_boxes, target_gp) in enumerate(train_dl):        
#       visualize_sample(images, target_boxes, i) # uncomment when testing regular
#       if i == 20:
#           break 
# =============================================================================




