#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:51:06 2020

@author: vince
"""
from utils import *
from detectionStuff import *
from livelossplot import PlotLosses

def train(train_loader, model, extractor, optimizer,  epoch, device, print_freq):
    
    liveloss = PlotLosses()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
    model.train()  # training mode enables dropout
    
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()
    
    start = time.time()
    running_loss = 0.0
    # Batches
    for i, (images, target_boxes, target_gp) in enumerate(train_loader):
        data_time.update(time.time() - start)
        
        #BATCH_SIZE = len(target_gp)
        img = images.numpy()
        img = np.squeeze(img, axis = 0) 
        
        box = bbox_changed(target_boxes) 
        box = np.array(box)
     
        features, new_image_size = feature_extractor(img, box, extractor)
        features = features.to(device)
        
        delta, score, _, anchor = model.forward(features, new_image_size)
        
        loss = model.loss(delta, score, anchor, box, new_image_size)
        losses.update(loss.item(), images.size(0))
        
        # loss = model.loss(img, bbox, label)
        optimizer.zero_grad()
        loss.backward()
        #scheduler.step() # magic with lr comes here
        # update model
        optimizer.step()
        running_loss = running_loss + loss.item() * images.size(0)
    
        #loss_value = loss.cpu()
        #loss_value = loss_value.data.numpy()
#       losses.update(loss.item(), images.size(0))

        batch_time.update(time.time() - start)
        start = time.time()
        
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                      batch_time=batch_time,
                                                      data_time=data_time, loss=losses))
         

        del images, target_boxes, target_gp  # free some memory since their histories may be stored
    #epoch_loss = running_loss / len(train_loader)
    #print(running_loss)
    return running_loss
