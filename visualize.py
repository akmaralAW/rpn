import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

import torch.nn as nn
import torchvision.transforms.functional as FT
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont
from chainercv.visualizations.vis_image import vis_image

def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from top-left coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in top-left coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
   # return torch.cat([cxcy[:, :2] + (cxcy[:, 2:]),  # x_min, y_min
#                      cxcy[:, :2]], 1)  # x_max, y_max
    return torch.cat([cxcy[:, :2], 
                      (cxcy[:, :2] + cxcy[:, 2:])], 1)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        im = tensor.clone()
        for t, m, s in zip(im, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return im
    
def visualize_sample(normalized_image, frac_bbox, n):
    # target image dims
    im_width = 224
    im_height = 224
    
    # denormalize image
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = unorm(normalized_image[0])  # first image of batch
    image = T.ToPILImage(mode='RGB')(image)
    
    
    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [im_width, im_height, im_width, im_height]).unsqueeze(0)
    
    # transform bbox from fractional to pixel     
    det_boxes = frac_bbox[0] * original_dims
    det_boxes = cxcy_to_xy(det_boxes)
      
    # Annotate
    annotated_image = image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(len(det_boxes)):
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline='gold')
        draw.rectangle(xy=[l + 1. for l in box_location], outline='gold')  # a second rectangle at an offset of 1 pixel to increase line thickness
        
        
    del draw
    annotated_image.show() 
   # annotated_image.save('/home/akmaral/akmaral_NorLab/ForestProject/regions_gt/img_'+str(n)+'.png',"PNG")
    
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
    det_boxes = cxcy_to_xy(det_boxes)
#     print(det_boxes)
    return det_boxes
 

def visualize_detection(normalized_image, frac_gp, detected_frac_gp):
    # target image dims
    im_width = 224
    im_height = 224
    
    # denormalize image
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = unorm(normalized_image[0])  # first image of batch
    image = T.ToPILImage(mode='RGB')(image)
    
    
    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [im_width, im_height, im_width, im_height]).unsqueeze(0)
    
    # transform bbox from fractional to pixel     
    true_gp = frac_gp[0][:, 0:2] * original_dims[0, 0:2]
    true_gp = torch.cat((true_gp, frac_gp[0][:,2].unsqueeze(-1)), 1) 
    det_gp = detected_frac_gp[0][:, 0:2] * original_dims[0, 0:2]
    det_gp = torch.cat((det_gp, detected_frac_gp[0][:,2].unsqueeze(-1)), 1)
    
    # Annotate
    annotated_image = image
    draw = ImageDraw.Draw(annotated_image)
    
    # Draw grasping point
    for i in range(len(det_gp)):
        draw.point(xy=true_gp[i][0:2].round().tolist(), fill='red')  # center point
        # draw a around the center point to make it more visible
        draw.point(xy=[l + 1. for l in true_gp[i][0:2].round().tolist()], fill='red')  # center point
        
        draw.point(xy=det_gp[i][0:2].round().tolist(), fill='yellow')  # center point
        # draw a around the center point to make it more visible
        draw.point(xy=[l + 1. for l in det_gp[i][0:2].round().tolist()], fill='yellow')  # center point
        
        
    del draw
    
    annotated_image.show()
    
def visualize_region_proposals(normalized_image, det_boxes, n, probability):
    # target image dims
    im_width = 224
    im_height = 224
    print(probability.shape)
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = unorm(normalized_image[0])  # first image of batch
    image = T.ToPILImage(mode='RGB')(image)
    
    original_dims = torch.FloatTensor(
        [im_width, im_height, im_width, im_height]).unsqueeze(0)
    
    annotated_image = image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    for i in range(len(det_boxes)):
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline='gold')
        draw.rectangle(xy=[l + 0.2 for l in box_location], outline='gold')  # a second rectangle at an offset of 1 pixel to increase line thickness

        text_size = font.getsize("{:.2f}".format(probability[i]))
        text_location = [box_location[2] + 2., box_location[3] - text_size[1]]
        textbox_location = [box_location[2], box_location[3] - text_size[1], box_location[2] + text_size[0] + 4.,
                            box_location[3]]
        draw.rectangle(xy=textbox_location, fill='black')
        draw.text(xy=text_location, text="{:.2f}".format(probability[i]), fill='white',
                  font=font)
        
    del draw
    #annotated_image.show()
    annotated_image.save('/home/akmaral/akmaral_NorLab/ForestProject/regions/img_'+str(n)+'.png',"PNG")
    
def vis_bbox2(img, bbox, label=None, score=None, label_names=None,
             instance_colors=None, alpha=1., linewidth=10.,
             sort_by_score=True, ax=None):

    from matplotlib import pyplot as plt

    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    if sort_by_score and score is not None:
        order = np.argsort(score)
        bbox = bbox[order]
        score = score[order]
        if label is not None:
            label = label[order]
        if instance_colors is not None:
            instance_colors = np.array(instance_colors)[order]

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    if instance_colors is None:
        # Red
        instance_colors = np.zeros((len(bbox), 3), dtype=np.float32)
        instance_colors[:, 0] = 100
    instance_colors = np.array(instance_colors)

    for i, bb in enumerate(bbox):
#         if i == 20:
#             break
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        color = instance_colors[i % len(instance_colors)] / 255
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=color, linewidth=linewidth, alpha=alpha))

        caption = []

        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 30})
    return ax