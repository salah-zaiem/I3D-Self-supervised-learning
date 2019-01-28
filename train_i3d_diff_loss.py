import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()
import warnings
import time
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

import numpy as np

from pytorch_i3d import InceptionI3d
from Proxy_I3D import ProxyNetwork

from charades_dataset import Charades as Dataset

def run(init_lr=0.001, max_steps=20, mode='rgb', root='/proxy/', train_split='./scott.txt',test_split ="./scottt.txt",  batch_size=8*5, save_model='nope'):
    
    # This table contains the distance between two possible ordering sequences
    # It is therefore a 120*120 table
    distance_dict = np.load("distance_dict.npy")
    distance_dict = torch.from_numpy(distance_dict).float().cuda()
    root = "./proxy/"
    dataset = Dataset(train_split,  root, mode, )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    val_dataset = Dataset(test_split, root, mode)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        #Imagenet Pretraining
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        #You can modify the number of outputs in the file Siamese_I3D.py
        
        i3d = ProxyNetwork()
        
    i3d.cuda()
    
    
    i3d = nn.DataParallel(i3d)
    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(  optimizer, [300, 1000])


    num_steps_per_update = 1 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print ('Step {}/{}'.format(steps, max_steps))
        t1 = time.time()
        processed_elements = 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                processed_elements +=40
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())
                
                #Custom loss implementation
                # Depending on the "real" labels

                per_frame_logits = i3d(inputs)
                for i in range(labels.shape[0]) : 
                    #print(i)
                    per_frame_logits [i] *= distance_dict[labels[i][0][0]]
                    
                # upsample to input size
                #per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
                per_frame_logits = per_frame_logits.squeeze()
                labels=labels.squeeze()
                labels = labels.type(torch.LongTensor)
                labels = labels.cuda()
                # compute localization loss
                loc_loss = F.cross_entropy(per_frame_logits, labels)
                tot_loc_loss += loc_loss.item()
                #Class loss 

                loss =loc_loss /num_steps_per_update
                tot_loss += loss.item()
                loss.backward()
                # 10800 is the number of elements in the training set
                len_training_set = 10800
                print("processed elements  : " + str( processed_elements)  +" / " + str(len_training_set))
                print(time.time() -t1 )

            if phase == 'train':
                steps += 1
                optimizer.step()
                optimizer.zero_grad()
                lr_sched.step()
                if steps % 1 == 0:
                    print ('{} Train Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                    # save model
                    torch.save(i3d, "customloss"+str(steps)+'.pt')
                    tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                print ('{}  Val Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss, tot_cls_loss, (tot_loss*num_steps_per_update)) )
    


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model)
