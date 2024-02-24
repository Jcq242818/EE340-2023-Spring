"""LeNet网络的训练与测试"""
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as nf
from data import data_train_loader
from model import LeNet
from torch.utils.tensorboard import SummaryWriter

model = LeNet()  # 定义LeNet模型
model.train()  # 切换模型到训练状态
lr = 0.01  # 定义学习率
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 定义随机梯度下降优化器
writer = SummaryWriter()  # 定义Tensorboard输出类
# 下面开始进行模型训练
train_loss = 0
correct = 0
total = 0
epoch = 2  # 设置迭代期为2
iter_num = 0  # 统计迭代的总步数
for i in range(epoch):  # 进行2次完整的迭代
    for batch_idx, (inputs, targets) in enumerate(data_train_loader):
        # print(batch_idx,(inputs, targets))
        iter_num += 1  # 每次循环迭代的步数加1
        optimizer.zero_grad()  # 首先梯度全部清零，防止上一次的梯度叠加
        outputs = model.forward(inputs)  # 从模型的输入推断出输出
        loss = nf.cross_entropy(outputs, targets)
        # print(loss.item())
        writer.add_scalar("Loss/train", loss, iter_num)  # 输出损失函数，添加的是标量数据
        loss.backward()  # 执行反向传播
        optimizer.step()  # 执行优化计算，根据反向传播得到的梯度更新权值与偏差
        # print(loss.item())
        train_loss += loss.item()
        # 首先，torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示），第二个值是value所在的index（也就是predicted）。那么，这个
        # 下划线_表示的就是具体的value，也就是输出的最大值。那么为什么用下划线_，可不可以用其他的变量名称来代替，比如x？答案自然是可以的。
        _, predicted = outputs.max(1)
        total += targets.size(0)  # 记录接触到的样本的总数目, 因为一次循环迭代训练一个batch，因此其样本数目每个循环后增加256
        correct += predicted.eq(targets).sum().item()  # 记录训练时分类正确的次数，如果正确则每次+1
        print(batch_idx, len(data_train_loader), 'Loss: %.3f | Acc: %.3f %%(%d/%d)'
              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
# 等两次完整的迭代进行完毕后，保存训练好的模型及其参数
save_info = {  # 保存的信息: 1.迭代步数 2.优化器的状态字典 3.模型的状态字典
    "iter_num": iter_num, "optimizer": optimizer.state_dict(), "model": model.state_dict()
}
save_path = "./model.pth"  # 将模型存储的位置在当前根目录的文件夹中
torch.save(save_info, save_path)
