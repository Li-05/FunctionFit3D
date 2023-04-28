import torch
import matplotlib.pyplot as plt
from net import *
from util import *
from torch.utils.data import DataLoader
from data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = ".\\param\\func.pth"
net = MyNet(size=2).to(device)
net.load_state_dict(torch.load(weight_path))
net.eval()  # 将模型设为评估模式

data_loader = DataLoader(MyDataset(width=20, height=20), batch_size=1, shuffle=False)

x_array = []
y_array = []
_z_array = []

for i, (xy, z) in enumerate(data_loader):
    xy, z = xy.to(device), z.to(device)
    _z = net(xy).cpu().item()
    x = xy[0][0].cpu().item()
    y = xy[0][1].cpu().item()
    x_array.append(x)
    y_array.append(y)
    _z_array.append(_z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_array, y_array, _z_array, c=_z_array, cmap='viridis')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(elev=30, azim=120)  # 设置视角
plt.savefig('.\\param\\fit_curve.png', dpi=300, bbox_inches='tight')
plt.show()