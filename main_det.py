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
from visualize import visualize_detection, visualize_sample, visualize_region_proposals
from detectionStuff import *
from utils import *
from train_det import train
from livelossplot import PlotLosses
from detectionStuff import *

def get_module(pretrained, num_classes):
    model = resnet18(pretrained=pretrained)
    
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    n_outputs = num_classes
    # get number of input features for the classifier
    fc_input = model.fc.in_features
    # replace the pre-trained head with a new one
    model.fc = nn.Sequential(
        nn.Linear(fc_input, fc_input),
        nn.Linear(fc_input, n_outputs)
        )
    
    return model
        
if __name__ == "__main__":
    # main()
    
    BATCH_SIZE = 1
    
    # path to the dataset
    train_path = '/home/akmaral/akmaral_NorLab/ForestProject/train_set/'
    test_path = '/home/akmaral/akmaral_NorLab/ForestProject/test_set/'
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    
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
                      shuffle=True,
                      collate_fn=dataset_test.collate_fn, # use custom collate function here
                      pin_memory=True)
    
    image, boxes, grasp_loc = next(iter(train_dl))
    #visualize_sample(image, boxes) # plot first image + targets of the batch
    
    extractor, head, feature_dim = get_vgg16_extractor_and_head(1, roip_size=7, vgg_pretrained=True) 
    
    model = rpn(in_channel=feature_dim, mid_channel=512,ratio=[1], anchor_size= [4, 8, 16, 24, 32, 48, 64, 80, 96]) 
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = model.get_optimizer(is_adam=True) #(is_adam=False) for Adam, for FasterRCNN, SGD for RPN
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # weight_decay=0.0005)
    #step_size = 4*len(train_dl)
    #clr = cyclical_lr(step_size, min_lr=3e-4, max_lr=3e-3) #min_lr=end_lr/factor, max_lr=end_lr)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    #avg_loss = AverageValueMeter()
    #ma20_loss = MovingAverageValueMeter(windowsize=20)
    liveloss = PlotLosses()
    #model.train()
    
    num_epochs = 6

    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, 0.001, lr_decay_epoch=3) #learning_rate = 0.001 initially 
        # train for one epoch, printing every 10 iterations
        logs = {}
        running_loss = train(train_dl, model, extractor, optimizer, epoch, device, print_freq=100)
        #loss_values.append(epoch_loss)
        epoch_loss = running_loss / len(train_dl)
        #print(epoch_loss)
        prefix = ''
        logs[prefix + 'log loss'] = epoch_loss
           # logs[prefix + 'accuracy'] = epoch_acc.item()
        
        liveloss.update(logs)
        liveloss.send()
# =============================================================================
#         if (epoch > 0 and epoch % 5 == 0):
#             modelweight = model.state_dict()
#             torch.save(modelweight, "model_SGD_Softmax_"+str(epoch)+ ".pt")   
# =============================================================================
    #plt.plot(loss_values)    
    modelweight = model.state_dict()
    torch.save(modelweight, "model_SGD_Adjust_Softmax_final"+ ".pt")  
    #torch.save(modelweight, "epoch_"+str(epoch)+ ".pt")   
    
# =============================================================================
#     model.eval()
#     for i in range (10): #(len(test_dl)):
#         img, box, _ = next(iter(test_dl))
#         imgx = img.numpy()
#         imgx = np.squeeze(imgx, axis = 0) 
#         
#         features, image_size = predict_feature_extractor(imgx, extractor)
#         delta, score, score_softmax, anchor = model.forward(features, image_size)
# 	roi = model.predict(delta, score, anchor, image_size)
#         #roi = model.predict(delta, score_softmax, anchor, image_size)
#         roi = np.flip(roi) #to match Vincent's function for visualization
#         visualize_region_proposals(img, list(roi), i)
# 
# =============================================================================
