import torch
from torch import nn

class MyNet(nn.Module):
    def __init__(self, size=2) -> None:
        super(MyNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(size, 64),nn.ReLU(),
            nn.Linear(64, 16),nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
    x = torch.randn(32,2,dtype=torch.float32)
    net = MyNet(size=2)
    
    print(x.shape)
    print(net(x).shape)