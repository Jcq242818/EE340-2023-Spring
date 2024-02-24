""" 展示MNIST数据集载入的数据"""
import img as img
import matplotlib.pyplot as plt
from data import data_train_loader

figure = plt.figure()
num_of_images = 60

for imgs, targets in data_train_loader:
    break

for index in range(num_of_images):  # 载入训练集index为0-59共60张图片
    plt.subplot(6, 10, index + 1)
    plt.axis("off")
    img = imgs[index, ...]
    plt.imshow(img.numpy().squeeze(), cmap='gray_r')

plt.show()
