import torch
from torch.utils.data import Dataset
import random
from util import *


class MyDataset(Dataset):
    def func(self, x, y):
        return fitFunc(x, y)
    
    def __len__(self):
        return self.len

    def __init__(self, width = 10, height = 10):
        super().__init__()
        self.len = width*height
        self.x, self.y = getGrid(-5, 5, width, -5, 5, height)
        self.z = self.func(self.x, self.y)
        
    def __getitem__(self, index):
        xy = torch.tensor((self.x[index], self.y[index]), dtype=torch.float32)
        z = torch.tensor([self.z[index]], dtype=torch.float32)
        return xy, z


if __name__ == '__main__':
    data = MyDataset()
    print(data[35][0])
    print(data[35][1])