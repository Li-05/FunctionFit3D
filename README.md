# FunctionFit3D
二元函数拟合

## loss图
![avatar](/param/loss_curve.png)

## 拟合图
![avatar](/param/fit_curve.png)

## 网络结构
```
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
```
