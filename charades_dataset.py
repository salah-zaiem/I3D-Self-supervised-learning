import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

from PIL import Image


import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, name):
  frames = []
  #Loading the 5 frames
  for i in range(1,6) :
      direction =  image_dir + name  + "/frame" +str(i) + ".jpg" 
      img = cv2.imread(direction)
      img = cv2.resize(img,dsize=(224,224))
      frames.append(np.asarray(img, dtype=np.float32))
  frames = np.asarray(frames)
  return frames




def make_dataset(split_file, root, mode, num_classes=157):
    dataset = []
    num_frames = 5

    with open(split_file, 'r') as f:
        data = f.read().splitlines()
    splits= [i.split() for i in data]
    dataset = [ ]
    for ind, split_data in enumerate(splits) : 
        name =split_data[0] 
        add = split_data[1]
        num_frames = split_data[2]
        label_number = int(split_data[3])
        dataset.append((name, label_number, 4, num_frames))
        
    return dataset


class Charades(data_utl.Dataset):
    #Dataset element

    def __init__(self, split_file, root, mode, transforms=None):
        
        self.data = make_dataset(split_file, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.num_elements = 120

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        name, label, dur, nf = self.data[index]

        imgs = load_rgb_frames(self.root, name)
        label_array = np.zeros(( self.num_elements, 1), dtype= np.float32)
        label_array= np.array([[label]]) 
        return video_to_tensor(imgs), torch.from_numpy(label_array)

    def __len__(self):
        return len(self.data)
