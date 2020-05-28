import torch
from torch import optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import time

# 配置参数
DOWNLOAD_CIFAR = True
batch_size = 32  # 每次喂入的数据量
lr = 0.01  # 学习率
step_size = 10  # 每n个epoch更新一次学习率
epoch_num = 50  # 总迭代次数
num_print = int(50000//batch_size//4)  #每n次batch打印一次

# cifar10训练数据加载
train_data = torchvision.datasets.CIFAR10(
    root='.',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_CIFAR,  # 没下载就下载, 下载了就不用再下了
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True)
# cifar10测试数据加载
test_data = torchvision.datasets.CIFAR10(
    root='.',  # 保存或者提取位置
    train=False,  # this is test data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_CIFAR,  # 没下载就下载, 下载了就不用再下了
)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=False)

# 按batch_size 打印出dataset里面一部分images和label
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def image_show(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()


def label_show(loader):
    global classes
    dataiter = iter(loader)  # 迭代遍历图片
    images, labels = dataiter.__next__()
    image_show(make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    return images, labels


label_show(train_loader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from .Vgg16_Net import *
import Vgg_Net
from torch import nn
model = Vgg_Net.Vgg16Net().to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()
# SGD优化器
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
# 动态调整学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

# 训练

loss_list = []
start = time.time()

for epoch in range(epoch_num):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 将梯度初始化为零
        outputs = model(inputs)  # 前向传播求出预测的值
        loss = criterion(outputs, labels).to(device)  # 求loss,对应loss += (label[k] - h) * (label[k] - h) / 2

        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 更新所有参数

        running_loss += loss.item()
        loss_list.append(loss.item())
        if i % num_print == num_print - 1:
            print('[%d epoch, %d] loss: %.6f' % (epoch + 1, i + 1, running_loss / num_print))
            running_loss = 0.0
    lr_1 = optimizer.param_groups[0]['lr']
    print('learn_rate : %.15f' % lr_1)
    scheduler.step()

end = time.time()
print('time:{}'.format(end-start))

torch.save(model, './model.pkl')   #保存模型
model = torch.load('./model.pkl')  #加载模型

# test
model.eval()
correct = 0.0
total = 0
with torch.no_grad():  # 测试集不需要反向传播
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU
        outputs = model(inputs)
        pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        total += inputs.size(0)
        correct += torch.eq(pred,labels).sum().item()
print('Accuracy of the network on the 10000 test images: %.2f %%' % (100.0 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
    c = (pred == labels.to(device)).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += float(c[i])
        class_total[label] += 1
#每个类的ACC
for i in range(10):
    print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))