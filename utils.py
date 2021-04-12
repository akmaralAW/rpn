import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

import torch.nn as nn
from torchvision import transforms as T
import torchvision.transforms.functional as FT
from PIL import Image, ImageDraw, ImageFont
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

ia.seed(1)

def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from top-left coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in top-left coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
# =============================================================================
#     return torch.cat([cxcy[:, :2] + (cxcy[:, 2:]),
#                      cxcy[:, :2]], 1)  # Vincent
# =============================================================================

    return torch.cat([cxcy[:, :2], 
            (cxcy[:, :2] + cxcy[:, 2:])], 1) # returns [y, x, y + h, x + w]


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def bbox_zoom_at(image, boxes, gp):
    """
    Perform a zooming in operation.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated grasping point coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-in) image 
    # ASSUMED TO BE SQUARE
    target_w = image.width
    target_h = image.height
    bbox_w = boxes[0][2].clone()    # bbox are 2:1, we want the width because its the smallest 
    
    gp0_copy = gp[0][:].clone()
    # Generate random center point near the grasping point
    # so the center isn't always the grasping point
    delta_center_u = bbox_w / 4 # keep at least 2/3 of the bb zone 
    zoom_center = gp0_copy[0:2].clone()    # need to clone, otherwise pass it as referenced
    zoom_center[0] = random.uniform(gp0_copy[0] - delta_center_u, gp0_copy[0] + delta_center_u) # mettre distribution normale
    zoom_center[1] = random.uniform(gp0_copy[1], gp0_copy[1]  + delta_center_u)       
    
    # Calculate margins between bb boundaries and image boundaries
    left_margin = gp0_copy[0] - bbox_w / 2
    right_margin = gp0_copy[0] + bbox_w / 2
    bot_margin = gp0_copy[1] +  bbox_w / 2
    top_margin = gp0_copy[1] -  bbox_w / 2
    # Calculate margins between generated zoom center boundaries and image boundaries 
    left_margin_gen = zoom_center[0] - bbox_w / 2
    right_margin_gen = zoom_center[0] + bbox_w / 2
    bot_margin_gen = zoom_center[1] +  bbox_w / 2
    top_margin_gen = zoom_center[1] -  bbox_w / 2
    
    # Check each margins of original bb center
    delta = 1 # one pixel, for rounding   
    left_check = left_margin > 0 + delta
    right_check = right_margin + delta < target_w
    bot_check = bot_margin + delta < target_h
    top_check = top_margin > 0 + delta  # top is at (0,0)
    # Check each margins of generated bb center
    left_check_gen = left_margin_gen > 0 + delta
    right_check_gen = right_margin_gen + delta < target_w
    bot_check_gen = bot_margin_gen + delta < target_h
    top_check_gen = top_margin_gen > 0 + delta  # top is at (0,0)
    
    # if the bb is in the image
    if((left_check * right_check * bot_check * top_check) and 
       (left_check_gen * right_check_gen * bot_check_gen * top_check_gen)):
        zoom_center = zoom_center
    else:
        zoom_center[0] = (not left_check)*(bbox_w / 2) + (not right_check)*(target_w - bbox_w / 2) + right_check*left_check*(gp0_copy[0].clone())
        zoom_center[1] = (not top_check)*(bbox_w / 2) + (not bot_check)*(target_h - bbox_w / 2) + top_check*bot_check*(gp0_copy[1].clone())
    
    # Calculate the new grasping point, in fractional form
    gp_frac = gp0_copy.clone()
    gp_frac[0] = (gp0_copy[0] - (zoom_center[0] - bbox_w / 2)) / bbox_w
    gp_frac[1] = (gp0_copy[1] - (zoom_center[1] - bbox_w / 2)) / bbox_w
    gp[0][:] = gp_frac # fractional form
    
    # New bb is all the image
    bb = np.zeros((1, 4))
    bb[0,:] = ([0, 0, 1, 1])     
    bb = torch.FloatTensor(bb) 
    boxes[0][:] = bb
    
    new_img = image.crop((int(zoom_center[0] - bbox_w / 2), int(zoom_center[1] - bbox_w / 2), 
                    int(zoom_center[0] + bbox_w / 2), int(zoom_center[1] + bbox_w / 2)))
    return new_img.resize([target_w, target_h]), boxes, gp

def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, gp, dims=(224, 224), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (224, 224).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates
    
    # resize grasping location
    new_gp = gp[:, 0:2] / old_dims[0, 0:2]
    new_gp = torch.cat((new_gp, gp[:,2].unsqueeze(-1)), 1)

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims
        new_gp = new_gp[:, 0:2] * new_dims[0, 0:2]

    return new_image, new_boxes, new_gp


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image

# =============================================================================
# class ImgAugTransform:
#     def __init__(self):
#         self.aug_pipeline = iaa.Sequential([
#             # iaa.Sometimes(0.5, iaa.GaussianBlur((0, 1.0))), # apply Gaussian blur with a sigma between 0 and 3 to 50% of the images
#             # apply one of the augmentations: Dropout or CoarseDropout
#             iaa.OneOf([
#                 iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
#                 iaa.CoarseDropout((0.03, 0.10), size_percent=(0.02, 0.05), per_channel=0.2),
#             ]),
#             # apply from 0 to 3 of the augmentations from the list
#             iaa.SomeOf((0, 1),[
#                 iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
#                 iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
#                 iaa.AdditiveGaussianNoise(
#                     loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#                 # Change brightness of images (50-150% of original value).
#                 iaa.Multiply((0.5, 1.0), per_channel=0.5),
#                 iaa.Grayscale(alpha=(0.0, 1.0)),
# 
#                 # In some images distort local areas with varying strength.
#                 # iaa.Fliplr(1.0), # horizontally flip
#                 # iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.25, 0.25))), # crop and pad 50% of the images
#                 # iaa.Sometimes(0.5, iaa.Affine(rotate=5)) # rotate 50% of the images
#             ])
#         ],
#         random_order=True # apply the augmentations in random order
#         )
#         
#     def __call__(self, img, bb, gp):
#         # convert image in numpy format
#         im = np.array(img)
#         # transform gp into Keypoints and bb into Bounding Boxes
#         bbs = []
#         kps = []
#         for i in range(len(gp)):
#             kps.append(Keypoint(x=gp[i, 0], y=gp[i, 1]))
#             bbs.append(BoundingBox(x1=bb[i, 0], y1=bb[i, 1], x2=bb[i, 0] + bb[i, 2], y2=bb[i, 1] + bb[i, 3]))
#             
#         kps = KeypointsOnImage(kps, (224, 224, 3))
#         bbs = BoundingBoxesOnImage(bbs, (224, 224, 3))
#       
#         # Augment keypoints and images.
#         image_aug, kps_aug, bbs_aug = self.aug_pipeline(image=im, keypoints=kps, bounding_boxes=bbs)
#         
#         # convert keypoints and bounding boxes to tensors
#         for i, (kp_aug, bb_aug) in enumerate(zip(kps_aug, bbs_aug)):
#             gp[i, 0] = kp_aug.x
#             gp[i, 1] = kp_aug.y
#             
#             bb[i, 0] = bb_aug.x1
#             bb[i, 1] = bb_aug.y1
#             bb[i, 2] = bb_aug.x2 - bb_aug.x1
#             bb[i, 3] = bb_aug.y2 - bb_aug.y1
#         
#         image_aug = Image.fromarray(image_aug)
#         
#         return image_aug, bb, gp
# =============================================================================
class ImgAugTransform:
    def __init__(self):

        self.aug_pipeline = iaa.SomeOf((1, 6), [
            iaa.CoarseDropout((0.03, 0.10), size_percent=(0.02, 0.05), per_channel=0.2),
            # iaa.Fliplr(1.0),
            iaa.GaussianBlur(0.5),
            iaa.Dropout(0.05),
            iaa.AdditiveGaussianNoise(scale=0.1*255),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.MultiplyHue((0.5, 1.5)),
            iaa.AddToHueAndSaturation((-20, 20), per_channel=True),
            iaa.Grayscale(alpha=(0.0, 1.0))
            ], random_order=True)
    
# =============================================================================
#         iaa.Sequential([
#             # iaa.Sometimes(0.5, iaa.GaussianBlur((0, 1.0))), # apply Gaussian blur with a sigma between 0 and 3 to 50% of the images
#             # apply one of the augmentations: Dropout or CoarseDropout
#             iaa.OneOf([
#                 iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
#                 iaa.CoarseDropout((0.03, 0.10), size_percent=(0.02, 0.05), per_channel=0.2),
#             ]),
#             # apply from 0 to 3 of the augmentations from the list
#             iaa.SomeOf((1, None),[
#                 iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
#                 iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
#                 #iaa.Fliplr(1.0), # horizontally flip
#                 iaa.Grayscale(alpha=(0.0, 0.8)),
#                 iaa.AdditiveGaussianNoise(scale=0.1*255),
#                 #iaa.MultiplyHueAndSaturation(mul_saturation=(0.5, 1.5)),
#                 # iaa.AddToHueAndSaturation((-5, 5)),
#                 # iaa.GaussianBlur((0, 3.0)),
#                 # iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.25, 0.25))), # crop and pad 50% of the images
#                 # iaa.Sometimes(0.5, iaa.Affine(rotate=5)) # rotate 50% of the images
#             ])
#         ],
#         random_order=True # apply the augmentations in random order
#         )
# =============================================================================
     
    def __call__(self, img, bb, gp):
        # convert image in numpy format
        im = np.array(img)
        
        # transform gp into Keypoints and bb into Bounding Boxes
        bbs = []
        kps = []
        for i in range(len(gp)):
            kps.append(Keypoint(x=gp[i, 0], y=gp[i, 1]))
            bbs.append(BoundingBox(x1=bb[i, 0], y1=bb[i, 1], x2=bb[i, 0] + bb[i, 2], y2=bb[i, 1] + bb[i, 3]))
            
        kps = KeypointsOnImage(kps, (224, 224, 3))
        bbs = BoundingBoxesOnImage(bbs, (224, 224, 3))
      
        # Augment keypoints and images.
        image_aug, kps_aug, bbs_aug = self.aug_pipeline(image=im, keypoints=kps, bounding_boxes=bbs)
        
        # convert keypoints and bounding boxes to tensors
        for i, (kp_aug, bb_aug) in enumerate(zip(kps_aug, bbs_aug)):
            gp[i, 0] = kp_aug.x
            gp[i, 1] = kp_aug.y
            
            bb[i, 0] = bb_aug.x1
            bb[i, 1] = bb_aug.y1
            bb[i, 2] = bb_aug.x2 - bb_aug.x1
            bb[i, 3] = bb_aug.y2 - bb_aug.y1
        
        image_aug = Image.fromarray(image_aug)
        
        return image_aug, bb, gp

def transform_aug(image, boxes, grasp_loc, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'TRAIN', 'TEST'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_grasp_loc = grasp_loc
    
    # Skip the following operations for evaluation/testing 
    if split=='TRAIN':
        # Define an augmentation pipeline
        aug = ImgAugTransform()
        new_image, new_boxes, new_grasp_loc = aug(new_image, new_boxes, new_grasp_loc)

    # new_image, new_boxes, new_grasp_loc = bbox_zoom_at(new_image, new_boxes, new_grasp_loc) # uncomment when testing zoom at
    
    # Resize image to (224, 224) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes, new_grasp_loc = resize(new_image, new_boxes, new_grasp_loc, dims=(224, 224)) # uncomment when testing regular
    
    # Apply Tensor and nomalization on image
    tfs = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
                ])
    
    new_image = tfs(np.array(new_image))

    return new_image, new_boxes, new_grasp_loc 

def transform(image, boxes, grasp_loc, split): #original transform without augmentation
    assert split in {'TRAIN', 'TEST'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_grasp_loc = grasp_loc

    # Resize image to (224, 224) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes, new_grasp_loc = resize(new_image, new_boxes, new_grasp_loc, dims=(224, 224))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_grasp_loc



# =============================================================================
# 
# 
# class UnNormalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
# 
#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         im = tensor.clone()
#         for t, m, s in zip(im, self.mean, self.std):
#             t.mul_(s).add_(m)
#             # The normalize code -> t.sub_(m).div_(s)
#         return im
# =============================================================================

    
def save_checkpoint(epoch, model, optimizer):
    """
    Save model checkpoint.
    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint.pth.tar'
    torch.save(state, filename)
    
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    

def evaluate(test_loader, model, device):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()
    
    acc = torch.zeros((4))

    with torch.no_grad():
        # Batches
        for i, (images, target_boxes, target_gp) in enumerate(test_loader):        
            BATCH_SIZE = len(target_gp)
            
            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 224, 224)
            target_gp = [l.to(device) for l in target_gp]
            
            # only select the first grasping point
            new_target_gp = target_gp[:][0][0, :].unsqueeze_(0)    # unsqueeze adds a dimension to the tensor
            for j in range(1, len(target_gp)):
                new_target_gp = torch.cat((new_target_gp, target_gp[:][j][0].unsqueeze_(0)), dim=0)
    
            # Forward prop.
            predicted_gp = model(images)    # (N, 8732, 4)
            
            # pass predicted values through sigmoid (infers that the gp is centered)
            sig = nn.Sigmoid()
            predicted_gp[:, :2] = sig(predicted_gp[:, :2])
            
            # prediction accuracy
            acc[0] += torch.sum(torch.pairwise_distance(predicted_gp[:, :2], new_target_gp[:, :2]))
            acc[1] += torch.sum(torch.abs(predicted_gp[:, 0] - new_target_gp[:, 0]))
            acc[2] += torch.sum(torch.abs(predicted_gp[:, 1] - new_target_gp[:, 1]))
            acc[3] += torch.sum(torch.abs(predicted_gp[:, 2] - new_target_gp[:, 2]))
            
        del predicted_gp, images, target_boxes, target_gp  # free some memory since their histories may be stored
    
    return acc.cpu().detach().numpy() / ((i+1) * BATCH_SIZE)

    
def compute_accuracy(pred_frac_gp, true_frac_gp):
    # target image dims
    im_width = 224
    im_height = 224    
    
    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [im_width, im_height, im_width, im_height]).unsqueeze(0)
    
    # transform grasping point from fractional to pixel     
    true_gp = true_frac_gp[0][:, 0:2] * original_dims[0, 0:2]
    true_gp = torch.cat((true_gp, true_frac_gp[0][:,2].unsqueeze(-1)), 1) 
    det_gp = pred_frac_gp[0][:, 0:2] * original_dims[0, 0:2]
    det_gp = torch.cat((det_gp, pred_frac_gp[0][:,2].unsqueeze(-1)), 1)
    
    # Annotate
    