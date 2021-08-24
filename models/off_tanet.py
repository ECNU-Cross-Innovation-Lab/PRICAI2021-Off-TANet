import torch
from torch import nn
import numpy as np
from torchstat import stat

class Residual(nn.Module):
    def __init__(self,in_c,out_c):
        super(Residual,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_c,out_channels = out_c,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )
        self.botneck = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        
    def forward(self,x):
        x_prim = x
        x = self.conv(x)
        x = self.botneck(x_prim) + x
        x = self.pool(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self,in_c,out_c,fm_sz,pos_bias = False):
        super(SelfAttention,self).__init__()
        self.w_q = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.w_k = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.w_v = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.pos_code = self.__getPosCode(fm_sz,out_c)
        self.softmax = nn.Softmax(dim = 2)
        self.pos_bias = pos_bias

    def __getPosCode(self,fm_sz,out_c):
        x = []
        for i in range(fm_sz):
            x.append([np.sin,np.cos][i % 2](1 / (10000 ** (i // 2 / fm_sz))))
        x = torch.from_numpy(np.array([x])).float()
        return torch.cat([(x + x.t()).unsqueeze(0) for i in range(out_c)])
    
    def forward(self,x):
        q,k,v = self.w_q(x),self.w_k(x),self.w_v(x)
        pos_code = torch.cat([self.pos_code.unsqueeze(0) for i in range(x.shape[0])]).to(x.device)
        if self.pos_bias:
            att_map = torch.matmul(q,k.permute(0,1,3,2)) + pos_code
        else:    
            att_map = torch.matmul(q,k.permute(0,1,3,2)) + torch.matmul(q,pos_code.permute(0,1,3,2))
        am_shape = att_map.shape
        att_map = self.softmax(att_map.view(am_shape[0],am_shape[1],am_shape[2] * am_shape[3])).view(am_shape)
        return att_map * v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,in_c,out_c,head_n,fm_sz,pos_bias = False):
        super(MultiHeadSelfAttention,self).__init__()
        self.sa_blocks = [SelfAttention(in_c = in_c,out_c = out_c,fm_sz = fm_sz,pos_bias = pos_bias) for i in range(head_n)]
        self.sa_blocks = nn.ModuleList(self.sa_blocks)
        
    def forward(self,x):
        results = [sa(x) for sa in self.sa_blocks]
        return torch.cat(results,dim = 1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, padding = 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        x = torch.cat([avg_out, max_out], dim = 1)
        x = self.conv1(x)
        return self.sigmoid(x)

class EnhancedResidual(nn.Module):
    def __init__(self,in_c,out_c,fm_sz,net_type = 'ta'):
        super(EnhancedResidual,self).__init__()
        self.net_type = net_type
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = in_c,out_channels = in_c,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )
        self.botneck = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        if net_type == 'ta':
            self.spa = SpatialAttention()
            self.ca = ChannelAttention(in_planes = in_c,ratio = in_c)
            self.sa = MultiHeadSelfAttention(in_c = in_c,out_c = in_c // 4,head_n = 4,fm_sz = fm_sz)
        elif net_type == 'sa':
            self.sa = MultiHeadSelfAttention(in_c = in_c,out_c = out_c // 4,head_n = 4,fm_sz = fm_sz)
        elif net_type == 'cbam':
            self.spa = SpatialAttention()
            self.ca = ChannelAttention(in_planes = in_c,ratio = in_c)
    
    def forward(self,x):
        x0 = self.botneck(x)
        x = self.conv1(x)
        if self.net_type == 'sa':
            x = self.sa(x)
            #x = self.conv2(x)
        elif self.net_type == 'cbam':
            x = self.ca(x) * x
            x = self.spa(x) * x
            x = self.conv2(x)
        elif self.net_type == 'ta':
            x = self.ca(x) * x
            x = self.spa(x) * x
            x = self.sa(x)
            x = self.conv2(x)
        x = x + x0
        x = self.pool(x)
        return x
        
class BottleneckTransformer(nn.Module):
    def __init__(self,in_c,out_c,fm_sz,head_n = 4):
        super(BottleneckTransformer,self).__init__()
        self.botneck = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.sa = nn.Sequential(
            MultiHeadSelfAttention(in_c = in_c,out_c = out_c // head_n,head_n = head_n,fm_sz = fm_sz),
            MultiHeadSelfAttention(in_c = out_c,out_c = out_c // head_n,head_n = head_n,fm_sz = fm_sz)
        )
    
    def forward(self,x):
        x0 = self.botneck(x)
        x = self.sa(x)
        x = x + x0
        x = self.pool(x)
        return x

class OffTANet(nn.Module):
    def __init__(self,net_type = 'ta'):
        super(OffTANet,self).__init__()
        #[N,3,112,112]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 8,kernel_size = 7,stride = 2,padding = 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            #nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1)
        )
        #[N,8,56,56]
        self.res1 = Residual(in_c = 8,out_c = 18)
        #[N,18,28,28]
        self.res2 = Residual(in_c = 18,out_c = 28)
        #[N,28,14,14]
        if net_type == 'res':
            self.eres = Residual(in_c = 28,out_c = 8)
        elif net_type == 'bot':
            self.eres = BottleneckTransformer(in_c = 28,out_c = 8,fm_sz = 14,head_n = 4)
        else:
            self.eres = EnhancedResidual(in_c = 28,out_c = 8,fm_sz = 14,net_type = net_type)
        #[N,8,14,14]
        #[N,8,7,7]
        self.dense = nn.Sequential(
            nn.Linear(392,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,3)
        )
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.eres(x)
        x = x.view(-1,392)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    stat(OffTANet(net_type = 'ta'),(3,112,112))