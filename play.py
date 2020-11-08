from torch import nn
import torch

class LatentKeys(nn.Module):
    def __init__(self,conv_dim=64):
        super(LatentKeys, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        for i in range(5):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        layers.append(nn.AvgPool2d(4,stride=2))
        #layers.append(nn.Conv2d(1024,512*18,kernel_size=1, stride=1, padding=0, bias=False))

        #layers.append(nn.Conv2d(curr_dim,18*512, kernel_size=4, stride=4, padding=1, bias=False))
        #layers.append(nn.InstanceNorm2d(18*512, affine=True))
        #layers.append(nn.ReLU(inplace=True))
        #curr_dim

        self.main = nn.Sequential(*layers)
        self.dense = nn.Linear(2048*3*3,512*18)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.main(x)
        x = x.view(-1,2048*3*3)
        x = self.dense(x)
        return self.sigmoid(x)
l = LatentKeys(conv_dim=64)

x = torch.rand((1,3,256,256),dtype=torch.float32)

y = l(x)

print(y.shape)

y = y.view(-1,18,512)
print(y.shape)