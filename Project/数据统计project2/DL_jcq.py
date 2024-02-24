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

from sklearn.metrics import accuracy_score, recall_score, f1_score,confusion_matrix,classification_report
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1.数据获取
# read the data
models_type = ["MLP"]
fig_address = ['./Output/MLP']
data = pd.read_csv('UCI_Credit_Card.csv')

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

data = data.iloc[:, 1:]
# data = np.array(data, type(float))
data = data.to_numpy(dtype=np.float32)

# 删除数据的第一列并对其进行标准化操作
## 划分训练集与测试集
for ii in range(data.shape[1]-1):
    meanVal=np.mean(data[:,ii])
    stdVal=np.std(data[:,ii])
    data[:,ii]=(data[:,ii]-meanVal)/stdVal

train_data, test_data = train_test_split(data, test_size = 0.2, random_state=27893,stratify=data[:,-1])

# 构建Dataset子类
class My_UCI_dataset(Dataset):
  def __init__(self, data):
    self.X = data

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    data = self.X[index,0:-1]
    data = torch.tensor(data, dtype=torch.float32)
    label = self.X[index,-1]
    label = torch.tensor(label, dtype=torch.int64)
    data_pair = {'X': data, 'y': label}
    return data_pair


train_dataset = My_UCI_dataset(data = train_data)
test_dataset = My_UCI_dataset(data = test_data)
data_train_loader = DataLoader(train_dataset, batch_size=16 ,shuffle= True, num_workers=0,drop_last=True)  # 训练集的数据被随机打乱
data_test_loader = DataLoader(test_dataset, batch_size=16 , shuffle= False, num_workers=0,drop_last=True) # 测试集数据不用做随机排列

# # 5.构建CNN/MLP模型
class Net(nn.Module):
        # fc 全连接层
        def __init__(self):
            super().__init__()
            #Conv2d[ channels, output, height_2, width_2 ]
            self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 10, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=1),
        )
            self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(10, 20, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=1),
        )
            self.fc1 = nn.Linear(460,20)
            self.fc2 = nn.Linear(20,2)

        def forward(self, x):
            batch_size = x.size(0)
            x = x.view(batch_size,1,-1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(batch_size, -1)
            # print(x.shape)
            x = nf.relu(self.fc1(x))
            x = nf.relu(self.fc2(x))
            return x
model = Net().to(device)
# 可选模型是CNN还是MLP,下面一行代码是MLP

# model = nn.Sequential(nn.Linear(23, 22),nn.ReLU(),nn.Dropout(0.4),nn.Linear(22, 16),nn.ReLU(),nn.Dropout(0.2),nn.Linear(16, 2))


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight,a=-0.1,b=0.1)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

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
def plot_cm(labels, pre):
    conf_matrices = []
    reports = []
    conf_matrix = confusion_matrix(labels, pre)
    conf_matrices.append(conf_matrix)
    report = classification_report(labels, pre)
    reports.append(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    # print(report)
    conf_matrix_drawing(conf_matrix,models_type[0],fig_address[0])
    

## 6.模型训练
# 定义训练参数
model.train()  # 切换模型到训练状态
learning_rate = 0.01
momentum = 0.9
num_epoch = 10
iter_num = 0
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, \
momentum=momentum, weight_decay = 5e-4)  # lr学习率，momentum冲量
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(\
    optimizer =optimizer,T_max=num_epoch)
comment = f'learning_rate{learning_rate}'
train_loss = 0.0  # 这整个epoch的loss清零
total = 0
correct = 0

log=np.zeros([num_epoch,3])#train_loss,val_loss,val_accuracy
for epoch in range(1, num_epoch+1):
    print(epoch, '\n')
    train_loss = []
    train_correct = 0.0
    train_total = 0.0
    predicted_list = []
    labels_list = []
    for batch_idx, data_pair in enumerate(data_train_loader,1):
        inputs = data_pair['X'].to(device)
        targets = data_pair['y'].to(device).reshape(-1)
        iter_num += 1
        optimizer.zero_grad()
        print(inputs.shape)
        # forward + backward + update
        outputs = model.forward(inputs)
        # print(outputs.shape, targets.shape)
        loss = nf.cross_entropy(outputs, targets)
        # writer.add_scalar("Loss/train", loss, iter_num)
        loss.backward()
        optimizer.step()
        # 把运行中的loss累加起来
        train_loss += [loss.item()]
        _, predicted = torch.max(outputs, axis=1)
        # _, predicted = outputs.max(1)
        total += inputs.shape[0]
        correct += predicted.eq(targets).sum().item()
        train_total += targets.size(0)
        train_correct += torch.sum(predicted == targets).item()
        if batch_idx % 10 == 9:  # 不想要每一次都出loss，浪费时间，选择每xx次出一个平均损失,和准确率
            print('[epoch: %d, batch_idx: %d]: loss: %.3f , acc: %.2f %%'
                    % (epoch, batch_idx + 1, loss / 100, 100. * correct / total))
            # writer.add_scalar('train accuracy per 10 batches', 100. * correct / total, iter_num)
            loss = 0.0
            correct = 0
            total = 0
    log[epoch-1,0]=np.mean(train_loss)
    scheduler.step()  # 优化并更新学习率
    test_loss = []
    test_correct = 0.0
    test_total = 0.0
    ##模型测试
    print('-----------------------------------------validation!!!------------------------------------' )
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(data_test_loader,1):
            inputs = data['X'].to(device)
            labels = data['y'].to(device).reshape(-1)
            outputs = model.forward(inputs)
            l = nf.cross_entropy(outputs, labels)
            test_loss += [l.item()]

            _, predicted = torch.max(outputs, axis=1)
            predicted_list.append(predicted)
            labels_list.append(labels)

            test_total += labels.size(0)
            test_correct += torch.sum(predicted == labels).item()
            if batch_idx % 10 == 9:  # 不想要每一次都出loss，浪费时间，选择每xx次出一个平均损失,和准确率
                print('[epoch: %d, batch_idx: %d]: loss: %.3f , acc: %.2f %%'
                        % (epoch, batch_idx + 1, np.mean(test_loss), 100. * test_correct / test_total))
            # print(test_total)
            # print(test_correct)
        log[epoch-1,1]=np.mean(test_loss)
        log[epoch-1,2]=np.around(test_correct / test_total,decimals=5)
    # print(predicted_list,labels_list)
    # writer.add_scalar('train_loss', train_loss/train_total, epoch)
    # writer.add_scalar('train_accuracy', train_correct/train_total, epoch)
    # writer.add_scalar('test_loss', test_loss/test_total, epoch)
    # writer.add_scalar('test_accuracy', test_correct/test_total, epoch)

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
    # print(len(labels_list_total), len(pred_list_total))
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
# torch.save(model.state_dict(), './model_CNN.pth')  # 保存模型的状态字典，也就是保存模型的参数信息
# 统计其他信息
accuracy = accuracy_score(labels_list_total, pred_list_total)
f1 = f1_score(labels_list_total, pred_list_total)
recall = recall_score(labels_list_total, pred_list_total)
print(accuracy,f1,recall)
# 将预测结果写入两个txt文件中，debug用
with open('pred.txt', 'w') as file:
    # 遍历列表中的元素并将其写入文件
    for item in pred_list_total:
        file.write(str(item) + '\n')
with open('labels.txt', 'w') as file:
    # 遍历列表中的元素并将其写入文件
    for item in labels_list_total:
        file.write(str(item) + '\n')

#plot the curve of loss and acc
print(log)
x=np.arange(num_epoch)
x=x+1
plt.figure(figsize=(5,3))
plt.plot(x,log[:,0],linestyle='-',color='r',label='train loss',linewidth=2)
plt.plot(x,log[:,1],linestyle='-',color='b',label='val loss',linewidth=2)
plt.plot(x,log[:,2],linestyle='--',color='g',label='val acc',linewidth=2)
plt.title('Training loss and accuracy',fontsize=10)
plt.xlabel('epoch',fontsize=10)
plt.ylabel('value',fontsize=10)
plt.legend(fontsize=10)
plt.ylim(0,1)
plt.grid()
plt.show()



