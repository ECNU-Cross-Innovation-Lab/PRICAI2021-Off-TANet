from torch import nn,optim
import torch
from torchstat import stat

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 6,kernel_size = 5,stride = 1,padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6,out_channels = 16,kernel_size = 5,stride = 1,padding = 2)
        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride = 2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x

class OffApexNet(nn.Module):
    def __init__(self):
        super(OffApexNet, self).__init__()
        self.cnn1 = CNN()
        self.cnn2 = CNN()
        self.dense = nn.Sequential(
            nn.Linear(784*2,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,3)
        )
    
    def forward(self,flow):
        splited = torch.chunk(flow,dim = 1,chunks = 2)
        x = self.cnn1(splited[0])
        y = self.cnn2(splited[1])
        return self.dense(torch.cat((x.view(-1,784),y.view(-1,784)),dim = 1))

if __name__ == '__main__':
    stat(OffApexNet(),(2,28,28))