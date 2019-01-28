# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:08:10 2019

@author: zaiem
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
import tqdm
args = parser.parse_args()
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np

from pytorch_prediction import InceptionI3d

from charades_prediction import video_to_tensor, load_rgb_frames

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def forward(self, x):
    for end_point in self.VALID_ENDPOINTS:
        if end_point in self.end_points:
            x = self._modules[end_point](x) # use _modules to work with dataparallel
        
    embedding = (self.avg_pool(x))
    return embedding


def run(init_lr=0.1, max_steps=3, mode='rgb', root='/proxy/', train_split='./train.txt',test_split ="./test.txt",  batch_size=8*5, save_model='nope'):
    # setup dataset

    
    # setup the model

    temp = torch.load('customloss.pt')
    i3d = temp

    i3d.forward = forward
    
    i3d.cuda()


    i3d = nn.DataParallel(i3d)

    
    
    num_frames = 5 
    train_file = open("newgoodtrain.txt")
    elements = [i.split()[0] for i in train_file.readlines() ]
    print("first element")
    train_file.close()
    print(elements[0])
    predictions = []
    print(len(elements))
    for i in tqdm.tqdm(range(len(elements ))) : 
        dir_name = elements[i]
        imgs = load_rgb_frames( "./testing_samples/" , dir_name, num_frames)
        element_to_predict= (video_to_tensor(imgs).cuda())
        prediction = i3d.module.extract_features(element_to_predict)
        prediction =np.reshape( prediction.cpu().detach().numpy() , (528,1))
        predictions.append(prediction)
    predictions = np.hstack(predictions)
    np.save("proxyvectors.npy", predictions)

if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model)
