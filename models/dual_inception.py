import torch
from torch import nn
from torchstat import stat
class InceptionBlock(nn.Module):
    def __init__(self,in_c,out_c):
        super(InceptionBlock,self).__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_c,out_channels = out_c,kernel_size = 5,padding = 2),
            nn.ReLU(),
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_c,out_channels = out_c,kernel_size = 3,padding = 1),
            nn.ReLU(),
        )
        self.path3 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3,stride = 1,padding = 1),
            nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1),
            nn.ReLU(),
        )
        self.path4 = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        
    def forward(self,x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = self.path4(x)
        y = torch.cat([x1,x2,x3,x4],dim = 1)
        y = self.pool(y)
        return y
        
class DualInception(nn.Module):
    def __init__(self):
        super(DualInception,self).__init__()
        self.icpt1 = nn.Sequential(
            InceptionBlock(in_c = 1,out_c = 6),
            InceptionBlock(in_c = 24,out_c = 16)
        )
        self.icpt2 = nn.Sequential(
            InceptionBlock(in_c = 1,out_c = 6),
            InceptionBlock(in_c = 24,out_c = 16)
        )
        self.dense = nn.Sequential(
            nn.Linear(6272,1024),
            nn.ReLU(),
            nn.Linear(1024,3)
        )
    
    def forward(self,x):
        splited = torch.chunk(x,dim = 1,chunks = 2)
        y1 = self.icpt1(splited[0])
        y2 = self.icpt2(splited[1])
        y = torch.cat([y1.view(-1,3136),y2.view(-1,3136)],dim = 1)
        y = self.dense(y)
        return y

if __name__ == '__main__':
    stat(DualInception(),(2,28,28))