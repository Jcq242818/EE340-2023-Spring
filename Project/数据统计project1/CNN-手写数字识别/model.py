"""构建LeNet网络模型"""
import torch
import torch.nn as nn
import torch.nn.modules.conv as nc  # 导入卷积层的类文件
import torch.nn.modules.pooling as np  # 导入池化层的类文件
import torch.nn.modules.linear as nl  # 导入线性层的类文件


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nc.Conv2d(1, 6, (3, 3))  # (int, int)是一个元组，[int, int]则是一个list
        self.pool1 = np.MaxPool2d(2, 2)
        self.conv2 = nc.Conv2d(6, 16, (3, 3))
        self.pool2 = np.MaxPool2d(2, 2)
        self.fc3 = nl.Linear(16 * 6 * 6, 120)
        self.fc4 = nl.Linear(120, 84)
        self.fc5 = nl.Linear(84, 10)  # 最后的张量只有一个特征维度，且该维度的大小为10

    def forward(self, x):  # 定义前向传播函数
        x = self.pool1(torch.relu_(self.conv1(x)))
        x = self.pool2(torch.relu_(self.conv2(x)))
        # 此处相当于把x变换为(x通过第二层最大池化层后的size是N*16*6*6,通过size变换将第二次卷积的输出拉伸为一行,即N*1536)
        # 注意x的size(0)属性表述训练的mini-bitch的大小,也就是N
        x = x.view(x.size(0), -1)  # 在pytorch里面，view函数相当于numpy的reshape.
        x = torch.relu_(self.fc3(x))
        x = torch.relu_(self.fc4(x))
        x = self.fc5(x)  # 最后的输出，注意这里还没有使用softmax激活函数最后进行分类概率归一化
        return x


# 接下来我们可以对上面的模型进行一个小测试
if __name__ == "__main__":  # 进入主函数入口
    model = LeNet()
    ret = model.forward(torch.randn(1, 1, 32, 32))
    print(ret.shape)
