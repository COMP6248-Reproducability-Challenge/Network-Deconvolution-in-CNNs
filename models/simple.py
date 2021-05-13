import torch
import torch.nn as nn
import torch.nn.functional as F

from .deconv import *


class MySimpleCnn1(nn.Module):
    def __init__(self, channels_in=3, hidden_num=64, kernel_size=32, num_outputs=10):
        super(MySimpleCnn1, self).__init__()
        print('MySimpleCnn1')
        self.conv1 = nn.Conv2d(channels_in, num_outputs, kernel_size)
        #self.conv2 = nn.Conv2d(hidden_num, num_outputs, kernel_size)

    def forward(self, x):
        out = self.conv1(x)
        # out = F.relu(out)
        # out = self.conv2(out)

        return out.view(x.shape[0], -1)


class MySimpleCnn2(nn.Module):
    def __init__(self, channels_in=3, hidden_num=64, kernel_size=32, num_outputs=10):
        super(MySimpleCnn2, self).__init__()
        print('MySimpleCnn2')
        self.conv1 = nn.Conv2d(channels_in, num_outputs, kernel_size)
        self.bn1 = nn.BatchNorm2d(num_outputs)
        #self.conv2 = nn.Conv2d(hidden_num, num_outputs, kernel_size)
        #self.bn2 = nn.BatchNorm2d(num_outputs)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        #out = F.relu(out)
        #out = self.conv2(out)
        #out = self.bn2(out)

        return out.view(x.shape[0], -1)


class MySimpleCnn3(nn.Module):
    def __init__(self, channels_in=3, hidden_num=32, kernel_size=32, num_outputs=10):
        super(MySimpleCnn3, self).__init__()
        print('MySimpleCnn3')
        print(kernel_size)
        #self.conv1 = FastDeconv(channels_in, hidden_num, kernel_size)
        #self.conv2 = FastDeconv(hidden_num, num_outputs, kernel_size)

        self.conv = FastDeconv(channels_in, num_outputs, kernel_size)

    def forward(self, x):
        out = self.conv(x)
        # out = F.relu(out)
        # out = self.conv2(out)

        return out.view(x.shape[0], -1)
