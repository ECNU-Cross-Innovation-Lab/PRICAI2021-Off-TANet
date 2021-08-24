from torch import nn
import torch
from torchstat import stat

class ResBlock(nn.Module):
    def __init__(self,in_c,out_c):
        super(ResBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_c,out_channels = out_c,kernel_size = 3,padding = 1),
            nn.ReLU(),
        )
        self.botneck = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        
    def forward(self,x):
        x_sp = self.botneck(x)
        x = self.conv(x)
        x = x + x_sp
        x = self.pool(x)
        return x

class MACNN(nn.Module):
    def __init__(self):
        super(MACNN,self).__init__()
        self.res_vgg = nn.Sequential(
            ResBlock(3,32),
            ResBlock(32,64),
            ResBlock(64,128),
            ResBlock(128,256),
        )
        self.aconv = nn.Sequential(
            nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = 3,dilation = 4,padding = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = 3,dilation = 2,padding = 2),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(7 * 7 * 256,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,3)
        )
    
    def forward(self,x):
        x = self.res_vgg(x)
        x = self.aconv(x)
        x = x.view(-1,7 * 7 * 256)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    stat(MACNN(),(3,112,112))