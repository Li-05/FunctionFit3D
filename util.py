import numpy as np

def getGrid(x_min, x_max, x_num, y_min, y_max, y_num):
    x = np.linspace(x_min, x_max, x_num)
    y = np.linspace(y_min, y_max, y_num)
    xx, yy = np.meshgrid(x, y)
    return xx.reshape((-1)), yy.reshape((-1))

#需要拟合的曲线
def fitFunc(x, y):
    return x*y

if __name__ == '__main__':
    x, y = getGrid(-1, 1, 2, -1, 1, 3)
    print(x)
    print(y)