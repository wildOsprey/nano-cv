## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

from torchvision import models
import time

import torch.nn.functional as F
import torch.nn as nn

class CustomDetector(torch.nn.Module):
    def __init__(self, out_channel=136):
        super(CustomDetector, self).__init__()
        # 1 224 224 -> 16 224 224
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.relu1 = nn.ReLU()
        # 16 224 224 -> 16 112 112
        self.pool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        # 16 112 112 -> 32 112 112
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.relu2 = nn.ReLU()
        # 32 112 112 -> 32 56 56
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        # 32 56 56 -> 64 56 56
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.relu3 = nn.ReLU()
        # 64 56 56 -> 64 28 28
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        # 64 28 28 ->128 28 28
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.relu4 = nn.ReLU()
        # 128 28 28 ->128 14 14
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        # 128 14 14 -> 128*14*14
        self.fc1 = nn.Linear(in_features=128*14*14, out_features=500, bias=True)
        #128*14*14 -> 500
        self.fc2 = nn.Linear(in_features=500, out_features=out_channel, bias=True)

        if torch.cuda.is_available():
            self.conv1 = self.conv1.cuda()

            self.conv2 = self.conv2.cuda()

            self.conv3 = self.conv3.cuda()

            self.conv4 = self.conv4.cuda()

            self.relu1 = self.relu1.cuda()

            self.relu2 = self.relu2.cuda()

            self.relu3 = self.relu3.cuda()

            self.relu4 = self.relu4.cuda()
            
            self.pool = self.pool.cuda()

            self.pool2 = self.pool2.cuda()

            self.pool3 = self.pool3.cuda()

            self.pool4 = self.pool4.cuda()

            self.fc1 = self.fc1.cuda()

            self.fc2 = self.fc2.cuda()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = F.dropout(x, training=self.training)
            
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = F.dropout(x, training=self.training)

        x = x.view(-1, 128*14*14)
        x = self.fc1(x)
        x = self.fc2(x)
        out = x
        return out

        
    
class Net2(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        #1 224 224 -> 3 224 224
        self.layer0 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=1, dilation=1, groups=1, bias=True)
        #3 224 224 -> 1000
        self.net =  models.vgg16(pretrained=True)
        if torch.cuda.is_available():
            self.layer1 = self.layer1.cuda()
            self.net = self.net.cuda()
            self.layer0 =  self.layer0.cuda()
        #1000 -> 136
        self.layer1 = torch.nn.Linear(1000,136)
        #leave features and retrain classifier layers
        for p in self.net.features.parameters():
            p.requires_grad=False
            

    def forward(self,x):
        x0 = self.layer0(x)
        x1 = self.net(x0)
        x1 = F.dropout(x1, training=self.training)
        y = self.layer1(x1)
        return y
    
