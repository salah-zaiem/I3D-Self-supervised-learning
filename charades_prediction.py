# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:10:26 2019

@author: zaiem
"""

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
    return torch.from_numpy(pic.transpose([4,3,0,1,2]))


def load_rgb_frames(image_dir, name, num_frames):
  frames = []
  for i in range(0,num_frames) :
      direction =  image_dir + name  + "/frame" +str(i) + ".jpg" 
      #print(direction)

      img = cv2.imread(direction)
      img = cv2.resize(img,dsize=(224,224))
      frames.append(np.asarray(img, dtype=np.float32))
  frames = np.asarray(frames)
  frames = np.reshape(frames, (num_frames,224,224,3,1))
  return frames



