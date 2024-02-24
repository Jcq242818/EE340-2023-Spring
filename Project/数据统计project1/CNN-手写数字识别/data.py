""" MNIST数据集的载入"""
#注意，本项目使用LeNet网路进行训练，LeNet网络的输入是32*32，注意的是MNIST数据集的图片是黑白的，因此通道数为1
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# 对数据进行归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#导入MNIST数据集，数据集会下载到当前根目录(和本文件同一个目录的文件夹下)的data文件夹下
data_train = MNIST('./data', train = True, download=True, transform = transform)

data_test = MNIST('./data', train=False, download=True, transform= transform)
#分别创建两个DataLoader载入训练集与测试集的数据
# 注意batch-size表示每批样本的大小，一次训练迭代一个batch.因此len(data_train_loader)表示mini-batch的数目
#batch_idx表示batch批的数目下标
data_train_loader = DataLoader(data_train, batch_size=256 ,shuffle= True, num_workers=0)  # 训练集的数据被随机打乱

data_test_loader = DataLoader(data_test, batch_size=1024 , shuffle= False, num_workers=0) # 测试集数据不用做随机排列
