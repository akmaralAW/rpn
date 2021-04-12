#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:28:11 2020

@author: vince
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from torchvision.models import vgg16
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from skimage import io, transform
from torch.utils.data import DataLoader
import torchvision
#from engine import train_one_epoch, evaluate
from torchvision import utils
from torch.utils.data import Dataset
import pandas as pd  # need Version >0.24.0 check with print(pd.__version__)
from PIL import Image
import random
import torchvision.transforms.functional as FT
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont
from torchnet.meter import AverageValueMeter, MovingAverageValueMeter
from utils import cxcy_to_xy
# from chainercv.visualizations import vis_bbox

from skimage import transform as sktransform

import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage


from torchvision import transforms as T
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 15

def delta2bbox(src_bbox, delta):
    """
    src_bbox: (N_bbox, 4)
    delta:  (N_bbox, 4)
    """
    #---------- debug
    assert src_bbox.shape == delta.shape
    assert isinstance(src_bbox, np.ndarray)
    assert isinstance(delta, np.ndarray)

    #----------
    src_bbox_h = src_bbox[:,2] - src_bbox[:,0]  
    src_bbox_w = src_bbox[:,3] - src_bbox[:,1]
    src_bbox_x = src_bbox[:,0] + src_bbox_h/2 
    src_bbox_y = src_bbox[:,1] + src_bbox_w/2 

    dst_bbox_x = src_bbox_x + src_bbox_h*delta[:,0] 
    dst_bbox_y = src_bbox_y + src_bbox_w*delta[:,1] 
    dst_bbox_h = src_bbox_h * np.exp(delta[:,2])
    dst_bbox_w = src_bbox_w * np.exp(delta[:,3])

    dst_bbox_x_min = (dst_bbox_x - dst_bbox_h / 2).reshape([-1, 1])
    dst_bbox_y_min = (dst_bbox_y - dst_bbox_w / 2).reshape([-1, 1])
    dst_bbox_x_max = (dst_bbox_x + dst_bbox_h / 2).reshape([-1, 1])
    dst_bbox_y_max = (dst_bbox_y + dst_bbox_w / 2).reshape([-1, 1])
    
    dst_bbox = np.concatenate([dst_bbox_x_min, dst_bbox_y_min, dst_bbox_x_max, dst_bbox_y_max], axis=1)   #(N_dst_bbox, 4)
    return dst_bbox

def bbox2delta(src_bbox, dst_bbox):
    """
    src_bbox: (N_bbox, 4)
    dst_bbox: (N_bbox, 4)
    """
    #---------- debug
    assert isinstance(src_bbox, np.ndarray)
    assert isinstance(dst_bbox, np.ndarray)
    assert src_bbox.shape == dst_bbox.shape
    #----------
    src_h = src_bbox[:, 2] - src_bbox[:, 0] + 1.0
    src_w = src_bbox[:, 3] - src_bbox[:, 1] + 1.0
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_h
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_w

    dst_h = dst_bbox[:, 2] - dst_bbox[:, 0] + 1.0
    dst_w = dst_bbox[:, 3] - dst_bbox[:, 1] + 1.0
    dst_ctr_x = dst_bbox[:, 0] + 0.5 * dst_h
    dst_ctr_y = dst_bbox[:, 1] + 0.5 * dst_w

    # eps = np.finfo(src_h.dtype).eps
    # height = np.maximum(src_h, eps)
    # width = np.maximum(src_w, eps)
    height = src_h
    width = src_w

    dx = (dst_ctr_x - src_ctr_x) / height
    dy = (dst_ctr_y - src_ctr_y) / width
    dh = np.log(dst_h / height)
    dw = np.log(dst_w / width)

    dx = dx.reshape([-1,1])
    dy = dy.reshape([-1,1])
    dh = dh.reshape([-1,1])
    dw = dw.reshape([-1,1])
    delta = np.concatenate([dx, dy, dh, dw], axis=1)
    return delta


def bbox_iou(bbox1, bbox2):
    """
    bbox1: (N1, 4)
    bbox2: (N2, 4)
    return iou: (N1,N2)
    """
    #----------debug
    assert isinstance(bbox1, np.ndarray)
    assert isinstance(bbox2, np.ndarray)
    assert len(bbox1.shape) == len(bbox2.shape) == 2
    assert bbox1.shape[1] == bbox2.shape[1] == 4
    #----------
    top_left = np.maximum(bbox1[:,None,:2], bbox2[:,:2])        # (N1,N2,2)
    bottom_right = np.minimum(bbox1[:,None,2:], bbox2[:,2:])    # (N1,N2,2)

    area_inter = np.prod(bottom_right-top_left,axis=2) * (top_left < bottom_right).all(axis=2)  # (N1,N2)
    area_1 = np.prod(bbox1[:,2:]-bbox1[:,:2], axis=1)   # (N1,)
    area_2 = np.prod(bbox2[:,2:]-bbox2[:,:2], axis=1)   # (N2,)
    iou = area_inter / (area_1[:,None] + area_2 - area_inter)   # (N1, N2)
    return iou

class RoIPool(nn.Module):
    def __init__(self, pooled_height, pooled_width):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)

    def forward(self, features, rois, spatial_scale):
        """
        Args:
            features: (N=1, C, H, W)
            rois: (N, 5); 5=[roi_index, x1, y1, x2, y2]
            spatial_scale: feature size / image size, this is important because rois are in image scale!
        Note: both features and rois are required to be Variable type.
              You should transform rois to Variable and set requires_grad to False before pass is to this function.
        """
        #---------- debug
        assert isinstance(features, Variable)
        assert isinstance(rois, Variable)
        #---------- debug
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width))
        if torch.cuda.is_available():
            outputs = outputs.cuda()

        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data.item())
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(
                roi[1:].data.cpu().numpy() * spatial_scale).astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)

            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))

                    is_empty = (hend <= hstart) or(wend <= wstart)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = features[batch_ind]
                        data_pool = torch.max(data[:, hstart:hend, wstart:wend], 1)[0]
                        outputs[roi_ind, :, ph, pw] = torch.max(data_pool, 1)[0].view(-1)
        #---------- debug
        assert outputs.shape[0] == rois.shape[0]
        assert outputs.shape[1] == features.shape[1]
        assert outputs.shape[2] == self.pooled_height
        assert outputs.shape[3] == self.pooled_width
        assert isinstance(outputs, Variable)
        #---------- debug
        return outputs

class ProposalTargetCreator(object):
    """
    This class will be used only in training phase to build head's loss function.
    """
    def __init__(self, n_sample=128,
                 pos_ratio=0.25, 
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_high=0.5,
                 neg_iou_thresh_low=0.0):
            self.n_sample = n_sample
            self.pos_ratio = pos_ratio
            self.pos_iou_thresh = pos_iou_thresh
            self.neg_iou_thresh_high = neg_iou_thresh_high
            self.neg_iou_thresh_low = neg_iou_thresh_low
        
    def make_proposal_target(self, roi, gt_bbox, gt_bbox_label):
        """
        Args:
            roi: (N1, 4)
            gt_bbox: (N2, 4)
            gt_bbox_label: (N2,)
        Note that gt_bbox_label class range from 0 ~ n_class-1, backdround is not included
        
        Returns:
            sample_roi : (Nx, 4)
            target_delta_for_sample_roi : (Nx, 4)
            bbox_bg_label_for_sample_roi : (Nx,)
        Note that bbox_bg_label_for_sample_roi class range from 0 ~ n_class, background(class 0) is included
        """
        #---------- debug
        assert isinstance(roi, np.ndarray)
        assert isinstance(gt_bbox, np.ndarray)
        assert isinstance(gt_bbox_label, np.ndarray)
        assert len(roi.shape) == len(gt_bbox.shape) == 2
        assert len(gt_bbox_label.shape) == 1
        assert roi.shape[1] == gt_bbox.shape[1] == 4
        assert gt_bbox.shape[0] == gt_bbox_label.shape[0]
        #---------- debug

        # concate gt_bbox as part of roi to be chose
        roi = np.concatenate((roi, gt_bbox), axis=0)   

        n_pos = int(self.n_sample * self.pos_ratio)

        iou = bbox_iou(roi, gt_bbox)
        bbox_index_for_roi = iou.argmax(axis=1)
        max_iou_for_roi = iou.max(axis=1)

        # note that bbox_bg_label_for_roi include background, class 0 stand for backdround
        # object class change from 0 ~ n_class-1 to 1 ~ n_class
        bbox_bg_label_for_roi = gt_bbox_label[bbox_index_for_roi] + 1
        
        # Select foreground(positive) RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou_for_roi >= self.pos_iou_thresh)[0]
        n_pos_real = int(min(n_pos, len(pos_index)))
        if n_pos_real > 0:
            pos_index = np.random.choice(pos_index, size=n_pos_real, replace=False)
        
        # Select background(negative) RoIs as those within [neg_iou_thresh_low, neg_iou_thresh_high).
        neg_index = np.where((max_iou_for_roi >= self.neg_iou_thresh_low) & (max_iou_for_roi < self.neg_iou_thresh_high))[0]
        n_neg = self.n_sample - n_pos_real
        n_neg_real = int(min(n_neg, len(neg_index)))
        if n_neg_real > 0:
            neg_index = np.random.choice(neg_index, size=n_neg_real, replace=False)
        
        keep_index = np.append(pos_index, neg_index)
        sample_roi = roi[keep_index]
        bbox_bg_label_for_sample_roi = bbox_bg_label_for_roi[keep_index]
        bbox_bg_label_for_sample_roi[n_pos_real:] = 0   # set negative sample's label to background 0

        target_delta_for_sample_roi = bbox2delta(sample_roi, gt_bbox[bbox_index_for_roi[keep_index]])

        target_delta_for_sample_roi = (target_delta_for_sample_roi - np.array([0., 0., 0., 0.])) / np.array([0.1, 0.1, 0.2, 0.2])
        return sample_roi, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi
    
def _smooth_l1_loss(pred_delta, target_delta, weight, sigma):
    #---------- debug
    assert isinstance(pred_delta, Variable)
    assert isinstance(target_delta, Variable)
    assert isinstance(weight, Variable)
    #---------- debug
    sigma2 = sigma * sigma
    diff = weight * (pred_delta - target_delta)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1.0 / sigma2)).float() # do not back propagat on flag
    flag = Variable(flag, requires_grad=False)
    if torch.cuda.is_available():
        flag = flag.cuda()
        
    res = flag*(sigma2 / 2.)*(abs_diff * abs_diff)  +  (1 - flag)*(abs_diff - 0.5 / sigma2)
    return res.sum()


def delta_loss(pred_delta, target_delta, anchor_label, sigma):
    """
    Args:
        pred_delta: (N,4)
        target_delta: (N,4)
        anchor_label: (N,)
    """
    #---------- debug
    assert isinstance(pred_delta, Variable)
    assert isinstance(target_delta, Variable)
    assert isinstance(anchor_label, Variable)
    assert pred_delta.shape == target_delta.shape
    assert pred_delta.shape[0] == anchor_label.shape[0]
    #---------- debug
    weight = torch.zeros(target_delta.shape)

    pos_index = (anchor_label.data > 0).view(-1,1).expand_as(weight)
    weight[pos_index.cpu()] = 1
    weight = Variable(weight)
    if torch.cuda.is_available():
        weight = weight.cuda()

    loss = _smooth_l1_loss(pred_delta, target_delta, weight, sigma)
    
    # ignore gt_label==-1 for rpn_loss
    loss = loss / (anchor_label.data >=0).sum().type_as(loss) 
    return loss


def nms(roi, thresh=0.6, score=None): #changed to 0.6
    """Pure Python NMS baseline.
    roi: (N, 4)
    score: None or (N,)
    """
    #---------- debug
    assert isinstance(roi, np.ndarray)
    assert (score is None) or (isinstance(score, np.ndarray))
    assert len(roi.shape) == 2
    assert score is None or len(score.shape) == 1

    #----------
    x1 = roi[:, 0]
    y1 = roi[:, 1]
    x2 = roi[:, 2]
    y2 = roi[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    if score is None:    # roi are already sorted in large --> small order
        order = np.arange(roi.shape[0])
    else:               # roi are not sorted
        order = score.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep # list of index of kept roi


def get_vgg16_extractor_and_head(n_class, roip_size=7, vgg_pretrained=False):
    vgg16_net = vgg16(pretrained=True)
    features = list(vgg16_net.features)[0:30]
    
    for layer in features[0:10]:    # freeze top 4 conv2d layers
        for p in layer.parameters():
            p.requires_grad = False
    extractor = nn.Sequential(*features)
    output_feature_channel = 512

    classifier = list(vgg16_net.classifier)
    del(classifier[6])  # delete last fc layer
    classifier = nn.Sequential(*classifier)     # classifier : (N,25088) -> (N,4096); 25088 = 512*7*7 = C*H*W
    if torch.cuda.is_available():
        classifier = classifier.cuda()
    head = _VGG16Head(n_class_bg=n_class+1, roip_size=roip_size, classifier=classifier)
    if torch.cuda.is_available():
        extractor, head = extractor.cuda(), head.cuda()
    return extractor, head, output_feature_channel


class _VGG16Head(nn.Module):
    def __init__(self, n_class_bg, roip_size, classifier):
        """n_class_bg: n_class plus background = n_class + 1"""
        super(_VGG16Head, self).__init__()
        self.n_class_bg = n_class_bg
        self.roip_size = roip_size

        self.roip = RoIPool(roip_size, roip_size)
        self.classifier = classifier
        self.delta = nn.Linear(in_features=4096, out_features=n_class_bg*4)    # Note: predice a delta for each class
        self.score = nn.Linear(in_features=4096, out_features=n_class_bg)

        self._normal_init(self.delta, 0, 0.001)
        self._normal_init(self.score, 0, 0.01)

    def forward(self, feature_map, rois, image_size):
        """
        Args:
            feature_map: (N1=1,C,H,W)
            rois : (N2,4)
        """
        #---------- debug
        assert isinstance(feature_map, Variable)
        assert isinstance(rois, np.ndarray)
        assert len(feature_map.shape) == 4 and feature_map.shape[0] == 1    # batch size should be 1
        assert len(rois.shape) == 2 and rois.shape[1] == 4
        #---------- debug

        # this is important because rois are in image scale, we need to pass this ratio 
        # to roipooing layer to map roi into feature_map scale
        feature_image_scale = feature_map.shape[2] / image_size[0]  
        
        # meet roi_pooling's input requirement
        temp = np.zeros((rois.shape[0], 1), dtype=rois.dtype)
        rois = np.concatenate([temp, rois], axis=1) 

        rois = Variable(torch.FloatTensor(rois))
        if torch.cuda.is_available():
            rois = rois.cuda()

        roipool_out = self.roip(feature_map, rois, spatial_scale=feature_image_scale)

        roipool_out = roipool_out.view(roipool_out.size(0), -1) # (N, 25088)
        if torch.cuda.is_available():
            roipool_out = roipool_out.cuda()

        mid_output = self.classifier(roipool_out)   # (N, 4096)
        delta_per_class = self.delta(mid_output)    # (N, n_class_bg*4)
        score = self.score(mid_output)      # (N, n_class_bg)
        #---------- debug
        assert isinstance(delta_per_class, Variable) and isinstance(score, Variable)
        assert delta_per_class.shape[0] == score.shape[0] == rois.shape[0]
        assert delta_per_class.shape[1] == score.shape[1] * 4 == self.n_class_bg * 4
        assert len(delta_per_class.shape) == len(score.shape) == 2
        #---------- debug
        return delta_per_class, score

    def loss(self, score, delta_per_class, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi):
        """
        Args:
            score: (N, 2)
            delta_per_class: (N, 4*n_class_bg)
            target_delta_for_sample_roi: (N, 4)
            bbox_bg_label_for_sample_roi: (N,)
        """
        #---------- debug
        assert isinstance(score, Variable)
        assert isinstance(delta_per_class, Variable)
        assert isinstance(target_delta_for_sample_roi, np.ndarray)
        assert isinstance(bbox_bg_label_for_sample_roi, np.ndarray)
        #---------- debug
        target_delta_for_sample_roi = Variable(torch.FloatTensor(target_delta_for_sample_roi))
        bbox_bg_label_for_sample_roi = Variable(torch.LongTensor(bbox_bg_label_for_sample_roi))
        if torch.cuda.is_available():
            target_delta_for_sample_roi = target_delta_for_sample_roi.cuda()
            bbox_bg_label_for_sample_roi = bbox_bg_label_for_sample_roi.cuda()

        n_sample = score.shape[0]
        delta_per_class = delta_per_class.view(n_sample, -1, 4)

        # get delta for roi w.r.t its corresponding bbox label
        index = torch.arange(0, n_sample).long()
        if torch.cuda.is_available():
            index = index.cuda()
        delta = delta_per_class[index, bbox_bg_label_for_sample_roi.data]

        head_delta_loss = delta_loss(delta, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi, 1)
        head_class_loss = F.cross_entropy(score, bbox_bg_label_for_sample_roi)

        return head_delta_loss + head_class_loss

    def predict(self, roi, delta_per_class, score, image_size, prob_threshold=0.5):
        """
        Args:
            roi: (N, 4)
            delta_per_class: (N, 4*n_class_bg)
            score: (N, n_class_bg)
        """
        #---------- debug
        assert isinstance(roi, np.ndarray)
        assert isinstance(delta_per_class, Variable)
        assert isinstance(score, Variable)
        #---------- debug
        roi = torch.FloatTensor(roi)
        if torch.cuda.is_available():
            roi = roi.cuda()
        delta_per_class = delta_per_class.data
        prob = F.softmax(score, dim=1).data

        delta_per_class = delta_per_class.view(-1, self.n_class_bg, 4)
        
        #!!!!!
        delta_per_class = delta_per_class * torch.cuda.FloatTensor([0.1, 0.1, 0.2, 0.2]) + torch.cuda.FloatTensor([0., 0., 0., 0.])
        
        roi = roi.view(-1,1,4).expand_as(delta_per_class)
        bbox_per_class = delta2bbox(roi.cpu().numpy().reshape(-1,4), delta_per_class.cpu().numpy().reshape(-1,4))
        bbox_per_class = torch.FloatTensor(bbox_per_class)

        bbox_per_class[:,0::2] = bbox_per_class[:,0::2].clamp(min=0, max=image_size[0])
        bbox_per_class[:,1::2] = bbox_per_class[:,1::2].clamp(min=0, max=image_size[1])

        bbox_per_class = bbox_per_class.numpy().reshape(-1,self.n_class_bg,4)
        prob = prob.cpu().numpy()
        #---------- debug
        assert bbox_per_class.shape[0] == prob.shape[0]
        assert bbox_per_class.shape[2] == 4
        assert bbox_per_class.shape[1] == prob.shape[1] == self.n_class_bg
        #---------- debug
        
        # suppress:
        bbox_out = []
        class_out = []
        prob_out = []
        # skip class_id = 0 because it is the background class
        for t in range(1, self.n_class_bg):
            bbox_for_class_t = bbox_per_class[:,t,:]    #(N, 4)
            prob_for_class_t = prob[:,t]                #(N,)
            mask = prob_for_class_t > prob_threshold    #(N,)
            # debug:
            # print("mask", mask.sum())
            left_bbox_for_class_t = bbox_for_class_t[mask]  #(N2,4)
            left_prob_for_class_t = prob_for_class_t[mask]  #(N2,)
            keep = nms(left_bbox_for_class_t, score=left_prob_for_class_t)
            bbox_out.append(left_bbox_for_class_t[keep])
            prob_out.append(left_prob_for_class_t[keep])
            class_out.append((t-1)*np.ones(len(keep)))

        bbox_out = np.concatenate(bbox_out, axis=0).astype(np.float32)
        prob_out = np.concatenate(prob_out, axis=0).astype(np.float32)
        class_out = np.concatenate(class_out, axis=0).astype(np.int32)
        #---------- debug
        assert isinstance(bbox_out, np.ndarray)
        assert isinstance(prob_out, np.ndarray)
        assert isinstance(class_out, np.ndarray)
        #---------- debug
        return bbox_out, class_out, prob_out
    

    def _normal_init(self, m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()
            
class AnchorTargetCreator(object):
    """
    This class will be used only in training phase to build rpn's loss function.
    Args:
        n_sample (int): The number of anchers to sample.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.
    """
    def __init__(self, n_sample=256, pos_iou_thresh=0.6, neg_iou_thresh=0.2, pos_ratio=0.5): #changed parameters
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio


    def make_anchor_target(self, anchor, gt_bbox, image_size):
        """
        Assign ground truth supervision to sampled subset of anchors.
        Args:
            anchor:  (N1, 4)
            gt_bbox: (N2, 4)
        Return:
            target_dalta: (N1, 4), for bbox regression
            anchor_label: (N1, ), for classification
        """
        #---------- debug
        assert isinstance(anchor, np.ndarray)
        assert isinstance(gt_bbox, np.ndarray)
        assert len(anchor.shape) == len(gt_bbox.shape) == 2
        assert anchor.shape[1] == gt_bbox.shape[1] == 4
        #----------
        img_H, img_W = image_size
        n_anchor = len(anchor)

        index_inside_image = np.where(
            (anchor[:, 0] >= 0) &
            (anchor[:, 1] >= 0) &
            (anchor[:, 2] <= img_H) &
            (anchor[:, 3] <= img_W))[0]

        anchor = anchor[index_inside_image] # rule out anchors that are not fully included inside the image

        bbox_index_for_anchor, anchor_label = self._assign_targer_and_label_for_anchor(anchor, gt_bbox)

        # create targer delta for bbox regression
        target_delta = bbox2delta(anchor, gt_bbox[bbox_index_for_anchor])

        # expand the target_dalta and label to match original length of anchor
        target_delta = self._to_orignal_length(target_delta, n_anchor, index_inside_image, fill=0)
        anchor_label = self._to_orignal_length(anchor_label, n_anchor, index_inside_image, fill=-1)
        
        return target_delta, anchor_label


    def _assign_targer_and_label_for_anchor(self, anchor, gt_bbox):
        """ 
        assign a label for each anchor, and the targer bbox index(with max iou) for each anchor.
        label: 1 is positive, 0 is negative, -1 is don't care
        """
        #---------- debug
        assert len(anchor.shape) == len(gt_bbox.shape) == 2
        assert anchor.shape[1] == gt_bbox.shape[1] == 4
        #---------- debug
        
        label = np.zeros(anchor.shape[0], dtype=np.int32) - 1   # init label with -1
        
        bbox_index_for_anchor, max_iou_for_anchor, anchor_index_for_bbox = self._anchor_bbox_ious(anchor, gt_bbox)

        # 1. assign anchor with 0 whose max_iou is small than neg_iou_thresh
        label[max_iou_for_anchor < self.neg_iou_thresh] = 0

        # 2. for each gt_bbox, assign anchor with 1 who has max iou with the gt_bbox
        label[anchor_index_for_bbox] = 1

        # 3. assign anchor with 0 whose max_iou is large than pos_iou_thresh
        label[max_iou_for_anchor>self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.n_sample * self.pos_ratio)  # default: 128
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        
        # subsample negative labels if we have too many
        n_neg = int(self.n_sample - np.sum(label==1))
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        #---------- debug
        assert len(bbox_index_for_anchor.shape) == len(label.shape) == 1
        assert bbox_index_for_anchor.shape[0] == label.shape[0] == anchor.shape[0]
        # print(np.sum(label == 1)) # change anchor generate parameter, if neg samples and pos samples are not roughly equal to n_sample/2
        # print(np.sum(label == 0))
        assert np.sum(label == 0) + np.sum(label == 1) <= self.n_sample
        #---------- debug
        
        return bbox_index_for_anchor, label

    def _anchor_bbox_ious(self, anchor, gt_bbox):
        iou = bbox_iou(anchor, gt_bbox)
        
        bbox_index_for_anchor = iou.argmax(axis=1)   # (anchor.shape[0],)
        max_iou_for_anchor = iou.max(axis=1)

        anchor_index_for_bbox = iou.argmax(axis=0)   # (bbox.shape[0],)
        max_iou_for_bbox = iou.max(axis=0)

        return bbox_index_for_anchor, max_iou_for_anchor, anchor_index_for_bbox

    def _to_orignal_length(self, data, length, index, fill):
        shape = list(data.shape)
        shape[0] = length
        ret = np.empty(shape, dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
        return ret


class ProposalCreator(object):
    """
    make proposal rois, this will be used both in training phase and in test phase.
    """
    def __init__(self, nms_thresh=0.7, #0.7
                 n_train_pre_nms=6000, #12000
                 n_train_post_nms=1000, #2000
                 n_test_pre_nms=3000,   #6000
                 n_test_post_nms=5,    #300
                 min_roi_size=24):
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = 6000 #12000
        self.n_train_post_nms = 1000 #2000
        self.n_test_pre_nms = 3000 #6000
        self.n_test_post_nms = 5
        self.min_roi_size = 24 #16

    def make_proposal(self, anchor, delta, score, image_size, is_training):
        """
        image_size used for clip anchor inside image field.
        anchor: (N, 4)
        delta:  (N, 4)
        score:   (N,)
        """
        #---------- debug
        assert isinstance(anchor, np.ndarray) and isinstance(delta, np.ndarray) and isinstance(score, np.ndarray)
        assert len(anchor.shape) == 2 and len(delta.shape) == 2 and len(score.shape) == 1
        assert anchor.shape[0] == delta.shape[0] == score.shape[0]
        #----------
        if is_training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
#             print("testing")
        #print(n_pre_nms) - 3000
        # 1. clip the roi into the size of image
        roi = delta2bbox(anchor, delta)
        roi[:,slice(0,4,2)] = np.clip(roi[:,slice(0,4,2)], a_min=0, a_max=image_size[0])
        roi[:,slice(1,4,2)] = np.clip(roi[:,slice(1,4,2)], a_min=0, a_max=image_size[1])
        
        # 2. remove roi where H or W is less than min_roi_size
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= self.min_roi_size) & (ws >= self.min_roi_size))[0]
        roi = roi[keep, :]
        score = score[keep] #942, 1
        
        
        # 3. keep top n_pre_nms rois according to score, and the left roi are sorted according to score
        order = score.argsort()[::-1]
        order = order[:n_pre_nms]
        score = score[order]
        #print(score[order]) # HERE
        roi = roi[order,:]
        #print(roi.shape) # 18*18*k*2, 4
        
        # 4. apply nms, ans keep top n_post_nms roi
        # note that roi is already sorted according to its score value
        keep = nms(roi, self.nms_thresh)
        keep = keep[:n_post_nms]
        roi = roi[keep,:] # n_test_post_nms, 4
        #print(roi.shape)
        #print(score[keep]) # correct one
        return roi, score[keep]


def generate_anchor(feature_height, feature_width, image_size, ratio=[1], anchor_size =  [4, 8, 16, 24, 32, 48, 64, 80, 96]):
    #---------- debug                                                                          
    assert len(image_size) == 2
    #----------
    anchor_base = []
    for ratio_t in ratio:
        for anchor_size_t in anchor_size:
            h = anchor_size_t*math.sqrt(ratio_t)
            w = anchor_size_t*math.sqrt(1/ratio_t)
            anchor_base.append([-h/2, -w/2, h/2, w/2])
    anchor_base = np.array(anchor_base) # default shape: [9,4]

    K = len(ratio) * len(anchor_size)   # default: 9
    image_height = image_size[0]
    image_width = image_size[1]
    stride_x = image_height / feature_height
    stride_y = image_width / feature_width
    anchors = np.zeros([feature_height, feature_width, K, 4])
    for i in range(feature_height):
        for j in range(feature_width):
            x = i*stride_x + stride_x/2
            y = j*stride_y + stride_y/2
            shift = [x,y,x,y]
            anchors[i, j] = anchor_base+shift

    anchors = anchors.reshape([-1,4])
    #----------
    assert isinstance(anchors, np.ndarray)
    assert anchors.shape[0] == feature_height*feature_width*len(ratio)*len(anchor_size)
    assert anchors.shape[1] == 4
    #----------
    return anchors


def image_normalize(img):
    """
    Normalize an image to match the input distribution of pytorch pretrained model.
    Note: the pixel value of the image should be in range [0,1], if the original image
          is in range [0,255], do not forget to div it by 255 before pass it to this function.
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def adjust_image_size(image, min_size=300, max_size=600):  #min_size=600, max_size=1000
    C, H, W = image.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    image = sktransform.resize(image, (C, H * scale, W * scale), mode='reflect')
    return image


def resize_bbox(bbox, image_size_in, image_size_out):
    """
    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
    """
    bbox = bbox.copy()
    y_scale = float(image_size_out[0]) / image_size_in[0]
    x_scale = float(image_size_out[1]) / image_size_in[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def random_flip(image, bbox, vertical_random=False, horizontal_random=False):
    vertical_flip, horizontal_flip = False, False
    H,W = image.shape[1], image.shape[2]
    bbox = bbox.copy()

    if vertical_random:
        vertical_flip =  random.choice([True, False])
    if horizontal_random:
        horizontal_flip = random.choice([True, False])

    if vertical_flip:
        image = image[:,::-1,:]
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max

    if horizontal_flip:
        image = image[:,:,::-1]
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    
    return image, bbox

def feature_extractor(image, gt_bbox, extractor):
        """
        image: (C=3,H,W), pixels should be in range 0~1 and normalized.
        gt_bbox: (N2,4)
        gt_bbox_label: (N2,)
        """
        original_image_size = image.shape[1:]
        # image, gt_bbox = random_flip(image, gt_bbox, horizontal_random=False) #True
        
        image = adjust_image_size(image)
        new_image_size = image.shape[1:] #for loss, training, rpn.forward
       # image_size = image.shape[2:] #for prediction
        gt_bbox = resize_bbox(gt_bbox, original_image_size, new_image_size)

        # image = image_normalize(image)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

        image = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            image = image.cuda()

        features = extractor(image)
        return features, new_image_size 
    
def predict_feature_extractor(image, extractor):
    #""" image: (N=1,3,H,W)
    #    """
       # #---------- debug
        assert isinstance(image, np.ndarray)
        #---------- debug
       # if self.training == True:
       #     raise Exception("Do not call predict in training mode, you should call .eval() to set the model in eval mode!")
        original_image_size = image.shape[1:]        
        image = adjust_image_size(image)
        new_image_size = image.shape[1:]
        #image = image_normalize(image)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        
        image = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            image = image.cuda()
#         print("image before the extractor:")
#         print(image.size())
        features = extractor(image)
        image_size = image.shape[2:]
        return features, image_size
    
class rpn(nn.Module):
    def __init__(self, in_channel, mid_channel, ratio=[1], anchor_size =  [4, 8, 16, 24, 32, 48, 64, 80, 96]): 
        super(rpn, self).__init__()

        self.ratio = ratio
        self.anchor_size = anchor_size
        self.K = len(ratio)*len(anchor_size)    # default: 9 : 9 ahcnors per spatial channel in feature maps

        self.mid_layer = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1) 
        self.score_layer = nn.Conv2d(mid_channel, 2*self.K, kernel_size=1, stride=1, padding=0)
        self.delta_layer = nn.Conv2d(mid_channel, 4*self.K, kernel_size=1, stride=1, padding=0)
        
        self._normal_init(self.mid_layer, 0, 0.01)
        self._normal_init(self.score_layer, 0, 0.01)
        self._normal_init(self.delta_layer, 0, 0.01)

        self.proposal_creator = ProposalCreator()
        self.anchor_target_creator = AnchorTargetCreator()

    def forward(self, features, image_size):
        """
        Batch size are fixed to one.
        features: (N-1, C, H, W)
        """
        #---------- debug
        assert isinstance(features, Variable)
        assert features.shape[0] == 1
        #---------- debug

        _, _, feature_height, feature_width = features.shape
        image_height, image_width = image_size[0], image_size[1]

        mid_features = F.relu(self.mid_layer(features))
        #print(mid_features.shape)
        delta = self.delta_layer(mid_features)
        delta = delta.permute(0,2,3,1).contiguous().view([feature_height*feature_width*self.K, 4])
        
        score = self.score_layer(mid_features) # 1 x 2k x 18 x18
        score_softmax = score.permute(0, 2, 3, 1).contiguous()
        score_softmax = F.softmax(score_softmax.view(1, feature_height, feature_width, self.K, 2), dim=4)
        #print(score_softmax.shape) # (1, 18, 18, 3, 2)
        score_softmax_fg = score_softmax[:, :, :, :, 1].contiguous()
        score_softmax_fg = score_softmax_fg.view(1, -1) # (1, 942)
        
        score = score.permute(0,2,3,1).contiguous().view([feature_height*feature_width*self.K, 2])
      
        # ndarray: (feature_height*feature_width*K, 4)
        anchor = generate_anchor(feature_height, feature_width, image_size, self.ratio, self.anchor_size)
        #---------- debug
        assert isinstance(delta, Variable) and isinstance(score, Variable) and isinstance(anchor, np.ndarray)
        assert delta.shape == (feature_height*feature_width*self.K, 4)
        assert score.shape == (feature_height*feature_width*self.K, 2)
        #---------- debug
        return delta, score, score_softmax_fg, anchor

    def loss(self, delta, score, anchor, gt_bbox, image_size):
        #print(score.shape) # 942, 2
        #---------- debug
        assert isinstance(delta, Variable)
        assert isinstance(score, Variable)
        assert isinstance(anchor, np.ndarray)
        assert isinstance(gt_bbox, np.ndarray)
        #---------- debug
        target_delta, anchor_label = self.anchor_target_creator.make_anchor_target(anchor, gt_bbox, image_size)
        target_delta = Variable(torch.FloatTensor(target_delta))
        anchor_label = Variable(torch.LongTensor(anchor_label))
        #anchor_label = Variable(torch.FloatTensor(anchor_label))
        if torch.cuda.is_available():
            target_delta, anchor_label = target_delta.cuda(), anchor_label.cuda()

        rpn_delta_loss = delta_loss(delta, target_delta, anchor_label, 3)
        rpn_class_loss = F.cross_entropy(score, anchor_label, ignore_index=-1)   # ignore loss for label value -1
        return rpn_delta_loss + rpn_class_loss

    def predict(self, delta, score, anchor, image_size):
        #---------- debug
        assert isinstance(delta, Variable)
        assert isinstance(score, Variable)
        assert isinstance(anchor, np.ndarray)
        #---------- debug
        delta = delta.data.cpu().numpy()
        #score = score.data.cpu().numpy()
        #score_fg = score[:,1]
        score = score[0].data.cpu().numpy() # for score_softmax
        score_fg = score #for score_softmax
        roi, probability = self.proposal_creator.make_proposal(anchor, delta, score_fg, image_size, is_training=self.training)
    
        #---------- debug
        assert isinstance(roi, np.ndarray)
        #---------- debug
        return roi, probability


    def _normal_init(self, m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()
            
    def get_optimizer(self, is_adam=False): #is_adam=False initially, but it is for FasterRCNN
        lr = 0.001
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        if False:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

def bbox_changed (frac_bbox):
    # target image dims
    im_width = 224
    im_height = 224
      
    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
#         [im_width, im_height, im_width, im_height]).unsqueeze(0)
        [im_height, im_width, im_height, im_width]).unsqueeze(0)
    # transform bbox from fractional to pixel     
    det_boxes = frac_bbox[0] * original_dims
#     print(det_boxes)
#     print(det_boxes[:, :2])
#     print(det_boxes[:, 2:])
#     cxcy[:, :2] + (cxcy[:, 2:]
    det_boxes = cxcy_to_xy(det_boxes)
#     print(det_boxes)
    return det_boxes

def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_factor= 0.1, lr_decay_epoch=10): #decay = 0.1
    """Sets the learning rate to the initial LR decayed by lr_decay_factor every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch == 0:
        lr = init_lr * (lr_decay_factor ** (epoch // lr_decay_epoch))
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
def cyclical_lr(stepsize, min_lr=1e-4, max_lr=1e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda
