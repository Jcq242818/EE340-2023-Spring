## Convolutional Neural Network

### **Introduction of CNN**

​	Convolutional neural network (CNN) is a kind of neural network specially used to process data with similar grid structure. Convolutional networks are neural networks that use convolutional operations in place of matrix multiplication in at least one layer of the network.

​	The basic structure of a CNN usually consists of the following parts: **input layer, convolution layer, pooling layer, activation function layer, full-connection layer and softmax layer, **As shown in Fig.1 below.

![](.\图片\1.png)

<div align = 'center'><b>Fig.1 The basic structure of convolutional neural network</div>

​	The functions of these basic parts are shown below:

- **Input layer:** In image processing CNN, the `input` layer generally represents the pixel matrix of an image. A picture can be represented by a three-dimensional matrix. The length and width of the 3D matrix represent the size of the image, while the depth of the 3D matrix represents the color channel of the image. For example, a black and white image has a depth of 1, while in RGB color mode, the image has a depth of 3.
- **Convolutional layer:** The core of convolutional neural network is the convolutional layer, and the core part of the convolutional layer is the convolution operation. The operation of inner product (multiplicative and summation of elements one by one) on images (data of different data Windows) and filter matrix (a set of fixed weights: since multiple weights of each neuron are fixed, it can be regarded as a constant filter filter) is the so-called convolution operation, which is also the source of the name of convolutional neural network.

- **Pooling layer:** The core of pooling layer is Pooling operation which uses the overall statistical characteristics of the adjacent area of an input matrix as the output of the location, including Average Pooling, Max Pooling, etc. Pooling simply specifies a value on the region to represent the entire region. Hyperparameters of the pooling layer: pooling window and pooling step. Pooling can also be thought of as a convolution operation. **(My understanding is about the function of the pooling layer is to select some way to reduce dimension compression in order to speed up the computation and retain the typical features in the window, so as to facilitate the next step of convolution/full connection).**

- **Activation function layer:  **The activation function here usually refers to the nonlinear activation function, the most important characteristic of activation function is its ability to add nonlinearity into convolutional neural network in order to solve the problems with complex patterns such as computer vision or image processing. The common activation functions include Sigmoid, tanh and Relu. Generally, Relu is used as the activation function of convolutional neural network. The Relu activation function provides a very simple nonlinear transformation method. The function image of Relu is shown below:

<img src=".\图片\2.png" style="zoom:80%;" />

<div align = 'center'><b>Fig.2 The function image of Relu</div>

- **Full-connection layer:** After the processing of multi-wheel convolution layer and pooling layer, the final classification results are generally given by one or two full-connection layers at the end of CNN. After several rounds of processing of convolution layer and pooling layer, it can be considered that the information in the image has been abstracted into features with higher information content. We can regard the convolution layer and pooling layer as the process of automatic image feature extraction. After the extraction is complete, we still need to use the full-connection layer to complete the sorting task.

- **Softmax layer:** Through the softmax layer, we can get the probability distribution problem that the current sample belongs to different categories. The softmax function will convert the output values of multiple classes into probability distributions in the range of [0, 1]. The function of the softmax layer is shown below:

<img src=".\图片\3.png" style="zoom:80%;" />

<div align = 'center'><b>Fig.3 The function of the softmax layer</div>

### The codes in Credit Risk Analysis by CNN

​	In this part, I will split the entire program code into 9 sections, as follows:

- **1. Dataset acquisition and clarity**

```python
## 1.数据获取
## read the data
models_type = ["MLP"]
fig_address = ['./Output/MLP']
data = pd.read_csv('UCI_Credit_Card.csv')

## glimpse at the data
data.head()
data.info()

## 2.数据清洗
## relabel EDUCATION information
fil = (data.EDUCATION == 5) | (data.EDUCATION == 6) | (data.EDUCATION == 0)
data.loc[fil, 'EDUCATION'] = 4

## relabel MARRIAGE information
data.loc[data.MARRIAGE == 0, 'MARRIAGE'] = 3

## rename column
data = data.rename(columns={'default.payment.next.month':'def_pay','PAY_0':'PAY_1'})

data = data.iloc[:, 1:]
## data = np.array(data, type(float))
data = data.to_numpy(dtype=np.float32)
```

- **2. Standardized the dataset and the obtained the training set and test set **

Here, we use the mean and variance of each column of data information to standardize the data set, and then divide the data set into the training set and the test set with a ratio of 0.2. The result is 24,000 credit data in the training set and 6,000 credit data in the test set.

````python
### 划分训练集与测试集
for ii in range(data.shape[1]-1):
    meanVal=np.mean(data[:,ii])
    stdVal=np.std(data[:,ii])
    data[:,ii]=(data[:,ii]-meanVal)/stdVal

train_data, test_data = train_test_split(data, test_size = 0.2, random_state=27893,stratify=data[:,-1])
````

- **3. Build a subclass of data set and load the training and test sets using Dataloader**

```python
## 构建Dataset子类
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
data_train_loader = DataLoader(train_dataset, batch_size=16 ,shuffle= True, num_workers=0,drop_last=True)  ## 训练集的数据被随机打乱
data_test_loader = DataLoader(test_dataset, batch_size=16 , shuffle= False, num_workers=0,drop_last=True) ## 测试集数据不用做随机排列
```

- **4. Constructed the model of convolutional neural network**

The convolutional neural network model was built with the following code. Since our data is 1-dimensional, a one-dimensional convolutional module called **Conv1d** was used when building the model.

```python
class Net(nn.Module):
        ## fc 全连接层
        def __init__(self):
            super().__init__()
            ##Conv2d[ channels, output, height_2, width_2 ]
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
            ## print(x.shape)
            x = nf.relu(self.fc1(x))
            x = nf.relu(self.fc2(x))
            return x
model = Net().to(device)
```

- **5. Model weight initialization**

```python
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight,a=-0.1,b=0.1)
        m.bias.data.fill_(0.01)
model.apply(init_weights)
```

- **6. Hyperparameter adjustment before training**

```python
model.train()  ## 切换模型到训练状态
learning_rate = 0.01
momentum = 0.9
num_epoch = 10
iter_num = 0
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, \
momentum=momentum, weight_decay = 5e-4)  ## lr学习率，momentum冲量
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(\
    optimizer =optimizer,T_max=num_epoch)
comment = f'learning_rate{learning_rate}'
train_loss = 0.0  ## 这整个epoch的loss清零
total = 0
correct = 0
```

- **7. Model training**

```python
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
        ## forward + backward + update
        outputs = model.forward(inputs)
        ## print(outputs.shape, targets.shape)
        loss = nf.cross_entropy(outputs, targets)
        ## writer.add_scalar("Loss/train", loss, iter_num)
        loss.backward()
        optimizer.step()
        ## 把运行中的loss累加起来
        train_loss += [loss.item()]
        _, predicted = torch.max(outputs, axis=1)
        ## _, predicted = outputs.max(1)
        total += inputs.shape[0]
        correct += predicted.eq(targets).sum().item()
        train_total += targets.size(0)
        train_correct += torch.sum(predicted == targets).item()
        if batch_idx % 10 == 9:  ## 不想要每一次都出loss，浪费时间，选择每xx次出一个平均损失,和准确率
            print('[epoch: %d, batch_idx: %d]: loss: %.3f , acc: %.2f %%'
                    % (epoch, batch_idx + 1, loss / 100, 100. * correct / total))
            ## writer.add_scalar('train accuracy per 10 batches', 100. * correct / total, iter_num)
            loss = 0.0
            correct = 0
            total = 0
    log[epoch-1,0]=np.mean(train_loss)
    scheduler.step()  ## 优化并更新学习率
    test_loss = []
    test_correct = 0.0
    test_total = 0.0
```

- **8. Model testing (Model verification) during training**

```python
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
            if batch_idx % 10 == 9:  ## 不想要每一次都出loss，浪费时间，选择每xx次出一个平均损失,和准确率
                print('[epoch: %d, batch_idx: %d]: loss: %.3f , acc: %.2f %%'
                        % (epoch, batch_idx + 1, np.mean(test_loss), 100. * test_correct / test_total))
            ## print(test_total)
            ## print(test_correct)
        log[epoch-1,1]=np.mean(test_loss)
        log[epoch-1,2]=np.around(test_correct / test_total,decimals=5)
    ##把之前一个epoch的10个1024张量转成一个列表放进去(能放进去所有epoch中的张量)
    predicted_list = [aa.tolist() for aa in predicted_list]

        ## print(predicted_list,type(predicted_list))
    ##把列表里的每一个元素拆开放到这个列表里，这个列表应该包含10000 = 1w个元素
    pred_list_total = [i for item in predicted_list for i in item]
    ## print(len(pred_list_total))
    labels_list = [aa.tolist() for aa in labels_list]
        ## labels_list = torch.tensor(labels_list)
    labels_list_total = [i for item in labels_list for i in item]
    train_acc_sum = torch.tensor([round(train_correct/train_total, 5)])
    test_acc = torch.tensor([test_correct/test_total])
    if epoch == 1:
        train_acc_all = train_acc_sum
        test_acc_all = test_acc
            ## print(train_acc_all,type(test_acc_all))
    else:
        train_acc_all = torch.cat([train_acc_all, train_acc_sum])
        test_acc_all = torch.cat([test_acc_all, test_acc])
            ## print(train_acc_all,type(train_acc_all))
```

- **9. Plot the confusion matrix of model testing and output the relevant statistics during model training and testing**

```python
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
    ## show the imag
    ## plt.show()


## 混淆矩阵--输入是一次epoch中所有的类别标签
def plot_cm(labels, pre):
    conf_matrices = []
    reports = []
    conf_matrix = confusion_matrix(labels, pre)
    conf_matrices.append(conf_matrix)
    report = classification_report(labels, pre)
    reports.append(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    ## print(report)
    conf_matrix_drawing(conf_matrix,models_type[0],fig_address[0])
    
plot_cm(labels_list_total, pred_list_total)
plt.show()
## torch.save(model.state_dict(), './model_CNN.pth')  ## 保存模型的状态字典，也就是保存模型的参数信息
## 统计其他信息
accuracy = accuracy_score(labels_list_total, pred_list_total)
f1 = f1_score(labels_list_total, pred_list_total)
recall = recall_score(labels_list_total, pred_list_total)
print(accuracy,f1,recall)
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
```

### Experimental result

- **The result in the training process**

​	<img src="E:\Desktop\Proj2\图片\4.png" style="zoom:60%;" />

<div align = 'center'><b>Fig.4 The loss of the model and the accuracy during the testing and testing process with the number of iterations</div>

​	Figure 4 reflects the loss of the model during training and testing process and the accuracy during testing process with the number of iterations. It can be seen that the convolutional neural network converges very quickly in the training process, and the loss of the model and the classification accuracy converge to a stable value in less than a few epochs **(we have tried to train 200 epochs before, and found that the performance of the model is not significantly different from that of training 10 epochs)**.

- **The Confusion matrix during testing**

![](E:\Desktop\Proj2\图片\5.png)

<div align = 'center'><b>Fig.5 The function of the softmax layer</div>

​	As can be seen from the confusion matrix above: Similar to previous machine learning methods, the model is able to correctly classify the first class **(label 0)**. However, for the second category **(label 1)**, the model does not properly classify it well.

- **Other statistics during testing**

​	Then, we collected some statistics of the model during the testing process, including the values of the accuracy, F1 and Recall.

​                                              	![](E:\Desktop\Proj2\图片\6.png)             

<div align = 'center'><b>Fig.6 Other statistics of the model during testing process</div>

### Conclusion and Experience

- **Conclusion:**
  - After few iterations by using our CNN model, the average accuracy of the model on the test set reached 81.8%.
  - All the results of evaluating the testing set by our model fully show that the convolutional neural networks and machine learning methods are similar for credit risk analysis.
  - Although CNN is mainly used for image feature extraction, lower dimensional convolutional neural networks can still extract effective features from simple one-dimensional sequences, so as to realize the classification of one-dimensional data.
  
- **Experience:**
  
  - In the training of CNN model, the selection of hyperparameters is very important, and selecting a set of appropriate hyperparameters can significantly improve the testing effect of the model.
  
  - Before model training, it is necessary to carefully check whether the pre-processing and standardization process of the dataset is correct. If there is a problem in the preprocessing process, then the subsequent model training and testing will not get reliable results.
  
    

## Multi-Layer Perceptron

### Introduction of MLP

MLP (Multi-Layer Perceptron) is a feed-forward neural network model that maps sets of input data onto a set of appropriate outputs. The structure of MLP network can refer to the following figure:

![MLP](https://th.bing.com/th/id/R.47c55639cf17d5771daf46a9c121b568?rik=CijKChaqm6F3Hw&riu=http%3a%2f%2fwww.howcodex.com%2fassets%2fhow_codex%2fimages%2fdetail%2ftensorflow%2fimages%2fschematic_representation.jpg&ehk=B1%2f%2bhV0SdBqYUksDlLPLbPW8gU1bsn5jKuvNgUfM%2blo%3d&risl=&pid=ImgRaw&r=0)

An MLP consists of multiple layers of nodes in a directed graph, with each layer fully connected to the next one. Except for the input nodes, each node is a neuron (or processing element) with a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

And the dataset we use is `UCI_Credit_Card` dataset, which is a real dataset about credit card default payment in Taiwan. The dataset contains 30,000 records and 25 a#ttributes. The target attribute is default payment (Yes = 1, No = 0). The dataset can be downloaded from [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

### Neural Network Structure

The way of data preprocessing is the same as the previous models. And I will start with the neural network structure:

```python
model = nn.Sequential(nn.Linear(23, 22),nn.ReLU(),nn.Dropout(0.4),
        nn.Linear(22, 16),nn.ReLU(),nn.Dropout(0.2),nn.Linear(16, 2))
```

The sturcutre of our MLP network can also be shown in the following figure:

![MLP-us](https://smangic-markdown-image.oss-cn-shenzhen.aliyuncs.com/img/image-20230611213053730.png)


Here we have 3 hidden layers, and the activation function is ReLU. And we also use dropout to avoid overfitting. The dropout rate is 0.4, 0.2, 0.2 respectively. And the output layer has 2 nodes, which is the number of classes. And ReLU is a non-linear activation function, which is defined as f(x) = max(0,x). It is a very popular activation function, and it is widely used in deep learning. The reason why we use ReLU is that it can avoid the gradient vanishing problem. And it is also easy to compute.

And we also need to initialize the weights and bias:

```python
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight,a=-0.1,b=0.1)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
```

We use uniform distribution to initialize the weights and bias. And the bias is initialized to 0.01. Uniform distribution means that the weights and bias are randomly initialized between -0.1 and 0.1.

After that, we can start to train our model, we initialize the `learning_rate` to 0.01 and `momentum` to 0.9. And the optimizer we use is SGD, and the learning rate is 0.01. And the loss function is CrossEntropyLoss. And we train our model for 10 epochs. 

### Result

The result of MLP is shown in the following figures:

![image-20230611214320203](https://smangic-markdown-image.oss-cn-shenzhen.aliyuncs.com/img/image-20230611214320203.png)

figure on the left is the training loss curve and also the performance on the validation set. And the figure on the right is the confusion matrix of the final result after 10 epoch of training. 

According to the result of MLP we can find that the accuracy has already converged after 2 epochs. And the accuracy on the validation set is 0.82. This can also be shown from the confusion matrix, the accuracy is 0.82. As we already know, this is a highly unbalanced dataset, the people that will not default is much more than the people that will default. If we just classify all the people as not default, the accuracy will be 0.78, which is our original result. 
