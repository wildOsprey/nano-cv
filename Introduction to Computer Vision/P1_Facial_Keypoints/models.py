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

#https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
#I used cuda based on tutorial (I've trained the net on my machine)
#an architecture is inspired by 
#http://neuralnetworksanddeeplearning.com/chap6.html
#http://cs231n.github.io/convolutional-networks/
#https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
#(it is a common practice to use conv-relu-pool blocks like in lenet or conv-pool-relu to reduce computations)

class CustomDetector(torch.nn.Module):
    def __init__(self, out_channel=136):
        super(CustomDetector, self).__init__()
        
       
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.relu= nn.ReLU()
        
        self.pool = nn.MaxPool2d(kernel_size=2, padding=0)
        
        self.fc1 = nn.Linear(in_features=128*14*14, out_features=500, bias=True)
        
        self.fc2 = nn.Linear(in_features=500, out_features=out_channel, bias=True)

        if torch.cuda.is_available():
            self.conv1 = self.conv1.cuda()

            self.conv2 = self.conv2.cuda()

            self.conv3 = self.conv3.cuda()

            self.conv4 = self.conv4.cuda()

            self.relu = self.relu.cuda()

            self.pool = self.pool.cuda()

            self.fc1 = self.fc1.cuda()

            self.fc2 = self.fc2.cuda()
        
    def forward(self, x):
        # 1 224 224 -> 16 224 224
        x = self.relu(self.conv1(x))
        # 16 224 224 -> 16 112 112
        x = self.pool(x)
        
        # 16 112 112 -> 32 112 112
        x = self.relu(self.conv2(x))
        # 32 112 112 -> 32 56 56
        x = self.pool(x)
        
        x = F.dropout(x, training=self.training)
        
        # 32 56 56 -> 64 56 56
        x = self.relu(self.conv3(x))
        # 64 56 56 -> 64 28 28
        x = self.pool(x)
        
        x = F.dropout(x, training=self.training)
        
        # 64 28 28 ->128 28 28
        x = self.relu(self.conv4(x))
        # 128 28 28 ->128 14 14
        x = self.pool(x)

        x = F.dropout(x, training=self.training)
        # 128 14 14 -> 128*14*14
        x = x.view(-1, 128*14*14)
        x = self.fc1(x)
        #128*14*14 -> 500
        x = self.fc2(x)
        out = x
        return out   
    
class Net(nn.Module):
#sources about transfer learning
#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#https://towardsdatascience.com/transfer-learning-using-pytorch-part-2-9c5b18e15551
#but just changed the last fc layer didn't work for me, in the summary it showed 136 out channels, but durinf training there was an error that shapes didn't fit (136 ground truth to 1000 expected), so on one forum I found a proposition to add extra layer.
    def __init__(self):
        super(Net,self).__init__()
        #1 224 224 -> 3 224 224
        self.layer0 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=1, bias=True)
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
    
class Net2(nn.Module):

    def __init__(self):

        super(Net2, self).__init__()

        # Covolutional Layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2)

        # Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features = 36864, out_features = 1000) # The number of input gained by "print("Flatten size: ", x.shape)" in below
        self.fc2 = nn.Linear(in_features = 1000,    out_features = 1000)
        self.fc3 = nn.Linear(in_features = 1000,    out_features = 136) # the output 136 in order to having 2 for each of the 68 keypoint (x, y) pairs

        # Dropouts
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)




    def forward(self, x):

        # First - Convolution + Activation + Pooling + Dropout
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop1(x)
        #print("First size: ", x.shape)

        # Second - Convolution + Activation + Pooling + Dropout
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        #print("Second size: ", x.shape)

        # Third - Convolution + Activation + Pooling + Dropout
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        #print("Third size: ", x.shape)

        # Forth - Convolution + Activation + Pooling + Dropout
        x = self.drop4(self.pool(F.relu(self.conv4(x))))
        #print("Forth size: ", x.shape)

        # Flattening the layer
        x = x.view(x.size(0), -1)
        #print("Flatten size: ", x.shape)

        # First - Dense + Activation + Dropout
        x = self.drop5(F.relu(self.fc1(x)))
        #print("First dense size: ", x.shape)

        # Second - Dense + Activation + Dropout
        x = self.drop6(F.relu(self.fc2(x)))
        #print("Second dense size: ", x.shape)

        # Final Dense Layer
        x = self.fc3(x)
        #print("Final dense size: ", x.shape)

        return x
