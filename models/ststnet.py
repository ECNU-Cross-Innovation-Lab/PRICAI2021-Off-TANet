from torch import nn
import torch
from torchstat import stat

class SingleStream(nn.Module):
    def __init__(self,out_c):
        super(SingleStream,self).__init__()
        self.conv = nn.Conv2d(in_channels = 3,out_channels = out_c,kernel_size = 3,padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 3,stride = 3,padding = 1)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class STSTNet(nn.Module):
    def __init__(self):
        super(STSTNet,self).__init__()
        self.stream1 = SingleStream(3)
        self.stream2 = SingleStream(5)
        self.stream3 = SingleStream(8)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.dense = nn.Sequential(
            nn.Linear(400,400),
            nn.ReLU(),
            nn.Linear(400,3)
        )

    def forward(self,x):
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x3 = self.stream3(x)
        x = torch.cat([x1,x2,x3],dim = 1)
        x = self.pool(x)
        x = x.view(-1,400)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    stat(STSTNet(),(3,28,28))