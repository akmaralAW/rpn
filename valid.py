#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 08:11:02 2020

@author: akmaral
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
from chainercv.visualizations import vis_bbox

if __name__ == "__main__":
    # main()
    
    BATCH_SIZE = 1
    
    # path to the dataset
    test_path = '/home/akmaral/akmaral_NorLab/ForestProject/test_set/'
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    
    dataset_test = tree_dataset_zoom(test_path, split='test')
        
    # define training and test data loaders
    test_dl = DataLoader(dataset=dataset_test,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      collate_fn=dataset_test.collate_fn, # use custom collate function here
                      pin_memory=True)
    
    extractor, head, feature_dim = get_vgg16_extractor_and_head(1, roip_size=7, vgg_pretrained=True) 
    
    model = rpn(in_channel=feature_dim, mid_channel=512,ratio=[1], anchor_size=[16, 24, 32, 40, 48, 64, 70, 80, 96])
    model.cuda()
    model.load_state_dict(torch.load('/home/akmaral/akmaral_NorLab/ForestProject/model_SGD_Adjust_Softmax_final.pt'))
    model.eval()
    for i in range (20): #(len(test_dl)):
        img, box, _ = next(iter(test_dl))
        imgx = img.numpy()
        imgx = np.squeeze(imgx, axis = 0) 
        
        features, image_size = predict_feature_extractor(imgx, extractor)
        delta, score, score_softmax, anchor = model.forward(features, image_size)
        roi, probability = model.predict(delta, score_softmax, anchor, image_size)
        roi = np.flip(roi) #to match Vincent's function for visualization
        #print(roi)
        visualize_region_proposals(img, list(roi), i, probability)





