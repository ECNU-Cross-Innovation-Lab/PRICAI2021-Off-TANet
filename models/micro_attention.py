from torch import nn
import torch
from torchstat import stat

class MABlock(nn.Module):
    def __init__(self,in_c,out_c):
        super(MABlock,self).__init__()
        self.convl_1 = nn.Sequential(
            nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.convl_2 = nn.Sequential(
            nn.Conv2d(in_channels = out_c,out_channels = out_c,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.convm = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.convr = nn.Conv2d(in_channels = out_c * 3,out_channels = out_c * 3,kernel_size = 1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self,x):
        xl1 = self.convl_1(x)
        xl2 = self.convl_2(xl1)
        xm = self.convm(x)
        xl3 = xl2 + xm
        xr1 = torch.cat([xl1,xl2,xm],dim = 1)
        xr2 = self.convr(xr1)
        xr3 = torch.mean(xr2,dim = 1)
        xr3 = self.softmax(xr3.view(xr3.shape[0],xr3.shape[1] * xr3.shape[2])).reshape(xl3.shape[0],1,xl3.shape[2],xl3.shape[3])
        xl4 = xl3 + xl3 * torch.cat([xr3 for i in range(xl3.shape[1])],dim = 1)
        return xl4

class MicroAttention(nn.Module):
    def __init__(self):
        super(MicroAttention,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 32,kernel_size = 7,stride = 2,padding = 3),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1)
        self.ma = nn.Sequential(
            MABlock(32,64),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            MABlock(64,128),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            MABlock(128,256),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
        )
        self.dense = nn.Sequential(
            nn.Linear(7 * 7 * 256,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,3),
        )
    
    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.ma(x)
        x = x.view(-1,7 * 7 * 256)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    stat(MicroAttention(),(3,224,224))