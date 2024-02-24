import torch
from torch import nn,optim, tensor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision.models.resnet import resnet18
import os
import time
import torch.nn.functional as nf
from tqdm import tqdm #进度条库
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.manifold import TSNE
import seaborn as sns
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1.数据获取
# read the data
data = pd.read_csv('UCI_Credit_Card.csv')
models_type = ["MLP"]
fig_address = ['./Output/MLP']
# glimpse at the data
data.head()
data.info()

# 2.数据清洗
# relabel EDUCATION information
fil = (data.EDUCATION == 5) | (data.EDUCATION == 6) | (data.EDUCATION == 0)
data.loc[fil, 'EDUCATION'] = 4

# relabel MARRIAGE information
data.loc[data.MARRIAGE == 0, 'MARRIAGE'] = 3

# rename column
data = data.rename(columns={'default.payment.next.month':'def_pay','PAY_0':'PAY_1'})

# data.to_csv('output.csv', index=False)
# tmp = data.iloc[0, 1:24].values
# print(tmp)

# 4.制作数据集和对应的标签
tensor_all = torch.zeros(30000,23)
label = torch.zeros([30000])
for i in range(30000):
    tensor_all[i,:] = torch.tensor(data.iloc[i, 1:24].values)
    label[i] = data.iloc[i,24]
# print(tensor_all.shape)
# print(label)

## 划分训练集与测试集
# mask_one = label[:] == 1
# mask_zero = label[:] == 0
# # print(mask_one,mask_zero)
# #从mask_one和mask_zero里面抽70%组成训练集。同样地，从mask_one和mask_zero里面抽30%组成验证集
# one_true_indices = torch.nonzero(mask_one).squeeze()
# one_true_threshold = int(0.5 * len(one_true_indices))
# # print(one_true_indices)
# # train_one_indices = true_indices[:threshold]
# # test_one_indices = true_indices[:threshold]
# zero_true_indices = torch.nonzero(mask_zero).squeeze()
# # zero_true_threshold = int(0.7 * len(zero_true_indices))
# zero_true_threshold = one_true_threshold
# train_one_indices = one_true_indices[:one_true_threshold]
# train_zero_indices = zero_true_indices[:zero_true_threshold]
# train_all = torch.cat((train_zero_indices,train_one_indices)).tolist()


# test_one_indices = one_true_indices[one_true_threshold:]
# test_zero_indices = zero_true_indices[zero_true_threshold:]
# test_all = torch.cat((test_zero_indices,test_one_indices)).tolist()


train_data = tensor_all[0:20000,:].long()
train_label = label[0:20000].long()
test_data = tensor_all[20000:,:].long()
test_label = label[20000:].long()


# 构建Dataset子类
class MyDataset(Dataset):
    def __init__(self, data, label, transform=None, target_transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data = self.data[index,:]
        label = self.label[index]
        return data, label

    def __len__(self):
        return self.data.size(0)
# 对数据进行归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = MyDataset(data = train_data,label=train_label, transform=transform)
test_data = MyDataset(data = test_data,label = test_label, transform=transform)
data_train_loader = DataLoader(train_data, batch_size=256 ,shuffle= True, num_workers=0)  # 训练集的数据被随机打乱
data_test_loader = DataLoader(test_data, batch_size=1024 , shuffle= False, num_workers=0) # 测试集数据不用做随机排列
# # 5.构建CNN模型
# model = resnet18(pretrained=True).to(device)
class fc_part(nn.Module):
        # fc 全连接层
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(23, 6)
            self.dropout1 = nn.Dropout(p=0.5)  # dropout训练
            self.fc2 = nn.Linear(6,4)
            self.dropout2 = nn.Dropout(p=0.5)
            self.fc3 = nn.Linear(4,2)
            # self.fc4 = nn.Linear(8,2)

        def forward(self, x):
            x = nf.relu(self.fc1(x))
            x = self.dropout1(x)
            x = nf.relu(self.fc2(x))
            x = self.dropout2(x)
            x = nf.relu(self.fc3(x))
            # x = nf.relu(self.fc4(x))
            return x

model = fc_part().to(device)

def conf_matrix_drawing (conf_matrix,model_type,adress):
    plt.matshow(conf_matrix, cmap=plt.cm.Blues)
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
           plt.annotate(conf_matrix[j, i], xy=(i, j),
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(model_type)
    plt.savefig(adress)
    # show the imag
    # plt.show()


# 混淆矩阵--输入是一次epoch中所有的类别标签
# 混淆矩阵--输入是一次epoch中所有的类别标签
def plot_cm(labels, pre):
    conf_matrices = []
    reports = []
    conf_matrix = confusion_matrix(labels, pre)
    conf_matrices.append(conf_matrix)
    report = classification_report(labels, pre)
    reports.append(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(report)
    conf_matrix_drawing(conf_matrix,models_type[0],fig_address[0])
    # conf_numpy = confusion_matrix(labels, pre)
    # conf_numpy = conf_numpy.astype('float') / conf_numpy.sum(axis=1)
    # conf_numpy_norm = np.around(conf_numpy, decimals=0)
    # print(conf_numpy_norm)
    # conf_df = pd.DataFrame(conf_numpy)#将data和all_label_names制成DataFrame
    # plt.figure(1, figsize=(8, 7))
    # sns.heatmap(conf_numpy_norm, annot=True, cmap="Blues")  # 将data绘制为混淆矩阵
    # plt.title('confusion matrix', fontsize=15)
    # plt.ylabel('True labels', fontsize=14)
    # plt.xlabel('Predict labels', fontsize=14)
    # plt.savefig( './Output/Test_ConfMatrix_for_CNN.png')

## 6.模型训练
# 定义训练参数
model.train()  # 切换模型到训练状态
learning_rate = 0.01
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay = 5e-4)  # lr学习率，momentum冲量
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.99, last_epoch=-1)
comment = f'learning_rate{learning_rate}'
writer = SummaryWriter(comment=comment)
train_loss = 0.0  # 这整个epoch的loss清零
total = 0
correct = 0
num_epoch = 10
iter_num = 0

for epoch in range(1, num_epoch+1):
    print(epoch, '\n')
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0
    predicted_list = []
    labels_list = []
    for batch_idx, (inputs, targets) in enumerate(data_train_loader):
        iter_num += 1
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model.forward(inputs.to(torch.float32))
        # print(outputs.shape, targets.shape)
        loss = nf.cross_entropy(outputs, targets)
        writer.add_scalar("Loss/train", loss, iter_num)
        loss.backward()
        optimizer.step()
        # 把运行中的loss累加起来
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        _, predicted = outputs.max(1)
        total += inputs.shape[0]
        correct += predicted.eq(targets).sum().item()
        train_total += targets.size(0)
        train_correct += torch.sum(predicted == targets).item()
        train_loss += loss.item()
        # if batch_idx % 10 == 9:  # 不想要每一次都出loss，浪费时间，选择每xx次出一个平均损失,和准确率
        #     print('[epoch: %d, batch_idx: %d]: loss: %.3f , acc: %.2f %%'
        #             % (epoch, batch_idx + 1, loss / 100, 100. * correct / total))
        #     writer.add_scalar('train accuracy per 10 batches', 100. * correct / total, iter_num)
        #     loss = 0.0
        #     correct = 0
        #     total = 0
    scheduler.step()  # 优化并更新学习率

    test_loss = 0.0
    test_correct = 0.0
    test_total = 0.0
    iter_num = 0
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0
    predicted_list = []
    labels_list = []
    ##模型测试
    with torch.no_grad():
            # model.eval()
            with tqdm(data_test_loader, desc='Test') as t:
                for data in t:
                    iter_num += 1
                    inputs, labels = data
                    outputs = model.forward(inputs.to(torch.float32))
                    l = nf.cross_entropy(outputs, labels)
                    test_loss += l.item()

                    _, predicted = torch.max(outputs, axis=1)
                    predicted_list.append(predicted)
                    labels_list.append(labels)

                    test_total += labels.size(0)
                    test_correct += torch.sum(predicted == labels).item()
                    print(test_total)
                    print(test_correct)
                    writer.add_scalar("Loss/test", test_loss, iter_num)
                    t.set_postfix(test_loss=l.item(),
                                    test_accuracy=test_correct / test_total)
    # print(predicted_list,labels_list)
    writer.add_scalar('train_loss', train_loss/train_total, epoch)
    writer.add_scalar('train_accuracy', train_correct/train_total, epoch)
    writer.add_scalar('test_loss', test_loss/test_total, epoch)
    writer.add_scalar('test_accuracy', test_correct/test_total, epoch)

        # predicted_list = torch.reshape([predicted_list,-1])
    #把之前一个epoch的10个1024张量转成一个列表放进去(能放进去所有epoch中的张量)
    predicted_list = [aa.tolist() for aa in predicted_list]

        # print(predicted_list,type(predicted_list))
    #把列表里的每一个元素拆开放到这个列表里，这个列表应该包含10000 = 1w个元素
    pred_list_total = [i for item in predicted_list for i in item]
    # print(len(pred_list_total))
    labels_list = [aa.tolist() for aa in labels_list]
        # labels_list = torch.tensor(labels_list)
    labels_list_total = [i for item in labels_list for i in item]
        # labels_list = torch.reshape([labels_list,-1])
#  print('epoch %d, train acc %.3f,test acc %.3f, time %.1f sec'%(epoch,train_correct / train_total, test_correct/test_total,time.time()-start))
    train_acc_sum = torch.tensor([round(train_correct/train_total, 5)])
    test_acc = torch.tensor([test_correct/test_total])
    if epoch == 1:
        train_acc_all = train_acc_sum
        test_acc_all = test_acc
            # print(train_acc_all,type(test_acc_all))
    else:
        train_acc_all = torch.cat([train_acc_all, train_acc_sum])
        test_acc_all = torch.cat([test_acc_all, test_acc])
            # print(train_acc_all,type(train_acc_all))

plot_cm(labels_list_total, pred_list_total)
plt.show()
torch.save(model.state_dict(), './model_MLP.pth')  # 保存模型的状态字典，也就是保存模型的参数信息

# 将预测结果写入两个txt文件中，debug用
with open('pred.txt', 'w') as file:
    # 遍历列表中的元素并将其写入文件
    for item in pred_list_total:
        file.write(str(item) + '\n')
with open('labels.txt', 'w') as file:
    # 遍历列表中的元素并将其写入文件
    for item in labels_list_total:
        file.write(str(item) + '\n')
