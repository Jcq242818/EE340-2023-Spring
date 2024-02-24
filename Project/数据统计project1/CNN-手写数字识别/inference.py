"""LeNet网络的推断与测试"""
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as nf
from data import data_test_loader
from model import LeNet
from train import save_path
from torch.utils.tensorboard import SummaryWriter

"""" 模型保存时的存储信息
save_info = {  # 保存的信息: 1.迭代步数 2.优化器的状态字典 3.模型的状态字典
    "iter_num": iter_num, "optimizer": optimizer.state_dict(), "model": model.state_dict()}
"""
load_info = torch.load(save_path)
model = LeNet()  # 定义LeNet模型
model.load_state_dict(load_info["model"])  # 载入之前训练好保存的模型参数
model.eval()  # 切换模型到测试状态
writer = SummaryWriter()  # 定义Tensorboard输出类
# 下面开始进行模型测试
test_loss = 0
correct = 0
total = 0
iter_num = 0
with torch.no_grad():  # 因为测试的时候模型的参数就不会改动了，因此测试的时候必须关闭计算图
    for batch_idx, (inputs, targets) in enumerate(data_test_loader):
        iter_num += 1
        outputs = model.forward(inputs)  # 从模型的输入推断出输出
        loss = nf.cross_entropy(outputs, targets)
        # 删去训练部分的优化和更新权重的部分，剩下的代码与模型训练的代码相同
        test_loss += loss.item()
        # 首先，torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示），第二个值是value所在的index（也就是predicted）。那么，这个
        # 下划线_表示的就是具体的value，也就是输出的最大值。那么为什么用下划线_，可不可以用其他的变量名称来代替，比如x？答案自然是可以的。
        _, predicted = outputs.max(1)
        total += targets.size(0)  # 记录接触到的样本的总数目, 因为一次循环迭代训练一个batch，因此其样本数目每个循环后增加256
        correct += predicted.eq(targets).sum().item()  # 记录训练时分类正确的次数，如果正确则每次+1
        print(batch_idx, len(data_test_loader), 'Loss: %.3f | Acc: %.3f %%(%d/%d)'
              % (+test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        acc = correct / total  # 定义输出的准确度
        writer.add_scalar("Acc/test", acc, iter_num)  # 输出损失函数，添加的是标量数据
