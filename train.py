import torch
from net import *
from data import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epoch = 300

if __name__ == '__main__':
    net = MyNet(size=2).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    lossF = torch.nn.MSELoss()

    data_loader = DataLoader(MyDataset(width=10, height=10), batch_size=batch_size, shuffle=True)
    epoch_losses = []
    for index in range(epoch):
        total_loss = 0.0
        for i, (xy, z) in enumerate(data_loader):
            xy, z = xy.to(device), z.to(device)
            _z = net(xy)
            loss = lossF(_z, z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {index+1} loss: {avg_loss:.4f}")

    # 绘制loss曲线图
    plt.plot(epoch_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(".\\param\\loss_curve.png")  # 保存loss曲线图
    plt.show()

    torch.save(net.state_dict(), ".\\param\\func.pth")