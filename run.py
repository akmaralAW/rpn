#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:08:04 2020

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
from visualize import visualize_detection, visualize_sample, visualize_region_proposals
from detectionStuff import *
from utils import *
from train_det import train

if __name__ == "__main__":
    # train the model or only do inference
    infer = True
    BATCH_SIZE = 1
    # path to the dataset
    train_path = '/home/akmaral/akmaral_NorLab/ForestProject/train_set/'
    test_path = '/home/akmaral/akmaral_NorLab/ForestProject/test_set/'
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    # use our dataset and defined transformations 
    dataset_train = tree_dataset_zoom(train_path, split='train')
    dataset_test = tree_dataset_zoom(test_path, split='test')    # some thing, but without transforms
    # define training and test data loaders
    train_dl = DataLoader(dataset=dataset_train,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=4,
                      collate_fn=dataset_train.collate_fn, # use custom collate function here
                      pin_memory=True)
    test_dl = DataLoader(dataset=dataset_test,
                      batch_size=BATCH_SIZE,
                      shuffle=False,
                      collate_fn=dataset_test.collate_fn, # use custom collate function here
                      pin_memory=True)
    if infer:
        checkpoint = '/home/akmaral/akmaral_NorLab/ForestProject/checkpoint_detector.pth.tar' # path to model checkpoint, None if none    
    else:
        checkpoint = None
    # visualize an element of a batch
    #image, boxes, grasp_loc = next(iter(train_dl))
    extractor, head, feature_dim = get_vgg16_extractor_and_head(1, roip_size=7, vgg_pretrained=True) 
    # Initialize model or load checkpoint
    if checkpoint is None:            
        model = rpn(in_channel=feature_dim, mid_channel=512, ratio=[1], anchor_size= [32, 64, 128]) #[128, 256, 512])
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer = model.get_optimizer(is_adam=True) #(is_adam=False) for Adam, for FasterRCNN, SGD for RPN
        model.train()
       
        num_epochs = 5
        for epoch in range(num_epochs):
            if epoch <= 1 : # only decay after first epoch
                adjust_learning_rate(optimizer, epoch, init_lr=0.01, lr_decay_epoch=2)
            # train for one epoch, printing every 10 iterations
            train(train_dl, model, extractor, optimizer, epoch, device, print_freq=100)
            # Save checkpoint
            save_checkpoint(epoch, model, optimizer)        
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        
    model.eval()
    for i in range (5): #(len(test_dl)):
        img, box, _ = next(iter(test_dl))
        imgx = img.numpy()
        imgx = np.squeeze(imgx, axis = 0) 
        
        features, image_size = predict_feature_extractor(imgx, extractor)
        delta, score, score_softmax, anchor = model.forward(features, image_size)
        roi = model.predict(delta, score_softmax, anchor, image_size)
        roi = np.flip(roi) #to match Vincent's function for visualization

        visualize_region_proposals(img, list(roi), i)

# =============================================================================
#     for i, (img, box, target_gp) in enumerate(test_dl):
#         imgx = img.numpy()
#         imgx = np.squeeze(imgx, axis = 0) 
#         features, image_size = predict_feature_extractor(imgx, extractor)
#         delta, score, anchor = model.forward(features, image_size)
#         roi = model.predict(delta, score, anchor, image_size)
#         roi = np.flip(roi) #to match Vincent's function for visualization
#         visualize_region_proposals(img, list(roi), i)
#         if i == 5:
#             break
# =============================================================================
