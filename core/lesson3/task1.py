import tensorflow as tf
import torch
import torchvision
import matplotlib.pyplot as plt

DOWNLOAD_CIFAR = True

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_data = torchvision.datasets.CIFAR10(
    root='.',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_CIFAR,  # 没下载就下载, 下载了就不用再下了
)
print(train_data)
# plt.show()
# for images, labels in train_data:
#     images = images.numpy().transpose(1, 2, 0)  # 把channel那一维放到最后
#     plt.title(str(classes[labels]))
#     plt.imshow(images)
#     plt.pause(1)