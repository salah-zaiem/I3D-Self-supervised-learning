import torch.nn as nn
import torch
from pytorch_i3d import InceptionI3d
from pytorch_i3d import Unit3D

class ProxyNetwork(nn.Module):
    def __init__(self, num_outputs = 120):
        super(ProxyNetwork, self).__init__()
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        self.i3d = i3d
        self.siam_logits = Unit3D(in_channels=112 + 288 + 64 + 64, output_channels=128,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='siam_logits')
        self.siam_avg_pool = nn.AvgPool3d(kernel_size=[2, 14, 14],
                                     stride=(1, 1, 1))
        self.fc1 = nn.Linear(128 , 512)
        self.fc2 = nn.Linear(512, num_outputs)
        

       

    def forward(self, x):
        x = self.i3d.extract_features_bis(x)
        x = self.siam_avg_pool(x)
        x = self.siam_logits(x)
        x = x.squeeze()
        res =x 

        res = self.fc1(res)
        res = self.fc2(res)
        return res

    def extract_features(self, x):
        VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
    )

        for end_point in VALID_ENDPOINTS:
            x = self.i3d._modules[end_point](x)
        return self.avg_pool(x)


