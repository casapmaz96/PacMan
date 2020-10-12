import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torchvision.transforms as trans
from collections import namedtuple
"""This code is written with the help of the Official Pytorch DQN Tutorial provided in the lab material"""
"""The official Pytorch tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Observance: represents state transitions"""
Observation = namedtuple('observation', ('state', 'action', 'nextState', 'reward'))

class ERMemory(object):
    """ Experience replay memory """
    def __init__(self, size=10000):
        self.size = size; self.memory = []; self.pos = 0

    def observe(self, *args):
        """Add new state transition observence to memory"""
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.pos] = Observation(*args)
        self.pos = (self.pos+1) % self.size	##If capacity is full, start overwriting past observances
								##in a first come first served order

    def batch(self, bsize):
        """Sample a random batch of observances from memory"""
        return random.sample(self.memory, bsize)
class DQN(nn.Module):

    def __init__(self, h, w, outputs=5):
        super(DQN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=5, mode='bilinear', align_corners=False)
    #    self.maxpool1 = nn.MaxPool2d(3)
        self.conv1 = nn.Conv2d(6, 10, kernel_size=5, stride=1) #10-16, k=5
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=5, stride=1) #16-32
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        #self.conv4 = nn.Conv2d(10, 16, kernel_size=3, stride=1)
        #self.bn4 = nn.BatchNorm2d(16)
        #self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        #self.bn5 = nn.BatchNorm2d(32)
#        self.conv1 = nn.Conv2d(6, 10, kernel_size=3, stride=1)
#        self.bn1 = nn.BatchNorm2d(10)
#        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1)
#        self.bn2 = nn.BatchNorm2d(16)
#        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
#        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(5*w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(5*h)))

        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear( 256, 100)

        self.head = nn.Linear(100, outputs)

    def forward(self, x):
        x = self.upsample(x)
        #print(rx)
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.relu(self.bn2(self.conv2(x)))
        x = func.relu(self.bn3(self.conv3(x)))
        #x = torch.tanh(self.bn4(self.conv4(x)))
        #x = torch.tanh(self.bn5(self.conv5(x)))
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc2(x)
        #x = self.fc3(x)
 
        return self.head(x)

