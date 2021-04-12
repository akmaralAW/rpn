import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd  # need Version >0.24.0 check with print(pd.__version__)
from PIL import Image

# homemade library
from utils import transform_aug, transform


class tree_dataset_zoom(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param transforms: transformation to apply on images
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        # load all image files names, sorting them to ensure that they are aligned
        imgs = list(sorted(os.listdir(os.path.join(data_folder, "data"))))
        self.imgs_RGB = imgs[0::2]
        self.imgs_depth = imgs[1::2]
        # load labels from csv files
        self.labels = pd.read_csv(data_folder + '/label/AllLabels.csv', index_col=False) 

        # assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # load images ad masks
        img_path = os.path.join(self.data_folder, "data", self.imgs_RGB[i])
        # Read image
        image = Image.open(img_path, mode='r').convert('RGB')
        # image = Image.open(img_path, mode='r').transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')
        
        img_name = 'image_' + str(i).rjust(5, '0')  # make a string of 5 characters long, padding as necessary

        # use the key to find every labels associated to the image
        label_idx = self.labels['image_id'][:] == img_name
        grasp_uv = self.labels[['Grasp_u', 'Grasp_v', 'Grasp_angle']][label_idx].to_numpy()
        grasp_uv[:][:,1] = image.height - grasp_uv[:][:,1]  # invert v axis
        
# =============================================================================
#         bbox_topleft_u = self.labels['BBox_u'][label_idx].to_numpy()
#         bbox_topleft_v = image.height - self.labels['BBox_v'][label_idx].to_numpy()  # invert v axis
#         bbox_w = self.labels['BBox_w'][label_idx].to_numpy()
#         bbox_h = self.labels['BBox_h'][label_idx].to_numpy()
# =============================================================================
        # get bounding box coordinates for each tree
        bbox_w = self.labels['BBox_w'][label_idx].to_numpy()
        bbox_h = self.labels['BBox_h'][label_idx].to_numpy() / 2 # making it square
        bbox_topleft_u = self.labels['BBox_u'][label_idx].to_numpy()
        bbox_topleft_v = image.height - (self.labels['BBox_v'][label_idx].to_numpy() - bbox_h) # invert v axis
        num_trees = len(bbox_topleft_u)
        boxes = np.zeros((num_trees, 4))
        for i in range(num_trees):
             #boxes[i,:] = ([bbox_topleft_u[i], bbox_topleft_v[i], bbox_w[i], bbox_h[i]])   #Vincent
             boxes[i,:] = ([bbox_topleft_v[i], bbox_topleft_u[i], bbox_h[i], bbox_w[i]]) # changed

        # convert to tensor
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        grasp_loc = torch.FloatTensor(grasp_uv)  # (n_objects)

        # Apply transformations
        #image, boxes, grasp_loc = transform(image, boxes, grasp_loc, split=self.split)

        if (i%2 == 0):
            image, boxes, grasp_loc = transform_aug(image, boxes, grasp_loc, split=self.split)
        
        else:
            image, boxes, grasp_loc = transform(image, boxes, grasp_loc, split=self.split)
            
        return image, boxes, grasp_loc

    def __len__(self):
        return len(self.imgs_RGB)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py
        """

        images = list()
        boxes = list()
        grasp_loc = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            grasp_loc.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, grasp_loc  # tensor (N, 3, 300, 300), 3 lists of N tensors each