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
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
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
print("gpu:"+str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from .Vgg16_Net import *
import Vgg_Net
from torch import nn

# 训练
def train():
    model = Vgg_Net.Vgg16Net().to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
    # 动态调整学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

    loss_list = []
    start = time.time()

    for epoch in range(epoch_num):  # 总迭代次数
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):  # 所有训练数据迭代
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 将梯度初始化为零
            outputs = model(inputs)  # 前向传播求出预测的值，执行forward方法
            """
            outputs:
            [[-0.0266,  0.5358, -0.3468, -0.1389, -0.1288,  0.1302,  0.1038,  0.1826,
              0.2703,  0.1062],
            [-0.2149, -0.0244,  0.0479, -0.0068, -0.2742,  0.0387,  0.2560,  0.0376,
              0.2295, -0.1721],
            [-0.1498, -0.1273, -0.4098,  0.0313, -0.5201,  0.2215, -0.0673,  0.1680,
             -0.1639, -0.4174],
            [-0.0640, -0.0261, -0.5154,  0.2030, -0.1493, -0.0703,  0.2868,  0.2170,
              0.1241, -0.1331],
            [-0.0161,  0.0281, -0.3022, -0.0332, -0.0155, -0.0289,  0.2279,  0.4005,
              0.2232, -0.2622],
            [ 0.2364,  0.2173, -0.6466, -0.0405, -0.1710,  0.2120,  0.2642,  0.2840,
              0.4728, -0.4197],
            [ 0.0879,  0.5019, -0.2392, -0.4850, -0.1819,  0.3153,  0.6553, -0.1862,
              0.2969,  0.6380],
            [ 0.1018,  0.0263, -0.1556,  0.2929, -0.1174,  0.0699, -0.0149,  0.1355,
             -0.1446,  0.1433],
            [-0.3238, -0.2502, -0.1036,  0.2361, -0.2662,  0.0756,  0.2014,  0.0312,
              0.3952,  0.3009],
            [-0.1350,  0.1339, -0.3466, -0.0710, -0.4508,  0.1587,  0.0382,  0.5787,
              0.4512,  0.2845],
            [ 0.1281,  0.1061, -0.2931,  0.1341, -0.1897, -0.1002,  0.0475,  0.1717,
             -0.0076, -0.2419],
            [-0.3958,  0.0850,  0.0747, -0.1007, -0.0377,  0.0474,  0.0908,  0.2405,
              0.2466,  0.0971],
            [-0.3176,  0.1415, -0.2624, -0.0157, -0.1046,  0.0581,  0.2433,  0.1292,
             -0.0959, -0.2234],
            [-0.0234,  0.2162,  0.0726, -0.0264, -0.2789,  0.2553,  0.1036,  0.3583,
              0.2102, -0.1269],
            [-0.0701,  0.0391, -0.3510, -0.2412, -0.0931, -0.1524,  0.2578, -0.1859,
              0.2260, -0.2094],
            [ 0.4602,  0.0532, -0.6286, -0.1957, -0.5670,  0.4385,  0.3584,  0.1377,
             -0.0484,  0.3123],
            [-0.0969,  0.0412, -0.1346, -0.0923, -0.1049,  0.1889,  0.1183, -0.0882,
             -0.0077, -0.0329],
            [-0.1104,  0.1053, -0.2942,  0.2716, -0.2655, -0.0145, -0.2775,  0.2698,
              0.0997,  0.1039],
            [-0.1129,  0.0689, -0.0250, -0.2881, -0.0927,  0.0124,  0.4238,  0.1465,
              0.2668,  0.0835],
            [-0.2694,  0.3224, -0.1559,  0.0255, -0.0972,  0.0797,  0.0306,  0.4332,
              0.0708, -0.2640],
            [-0.2616,  0.0881, -0.3115, -0.1832,  0.0900,  0.1409, -0.0051, -0.1406,
              0.0271,  0.1856],
            [-0.4518,  0.5644, -0.2829,  0.3016, -0.3686,  0.1036, -0.1915, -0.0955,
              0.0654,  0.1372],
            [-0.0369,  0.1134, -0.2771, -0.0593, -0.2686, -0.1298,  0.2715, -0.2984,
              0.1061, -0.1124],
            [-0.4889, -0.0904, -0.2117, -0.1932, -0.4522,  0.3583, -0.2878,  0.0865,
              0.1668, -0.2309],
            [-0.2477,  0.1430,  0.0178, -0.0437, -0.0444,  0.0692,  0.1722,  0.2627,
              0.1049,  0.0860],
            [-0.0569, -0.3431, -0.0428, -0.2178, -0.4421,  0.2615,  0.3176,  0.1864,
              0.4022,  0.1547],
            [ 0.0599, -0.0488, -0.2485, -0.0268, -0.0656,  0.0416,  0.2021,  0.1802,
             -0.0341,  0.2157],
            [ 0.2398, -0.2537, -0.1936,  0.5982, -0.2710, -0.0850,  0.2154,  0.2468,
             -0.0176,  0.0167],
            [-0.2395,  0.3046,  0.1719,  0.4087,  0.2399, -0.0062,  0.1878,  0.6821,
              0.1283, -0.7710],
            [-0.3658, -0.3710, -0.2099, -0.1416,  0.1876,  0.0602,  0.6381, -0.0795,
              0.4198,  0.3850],
            [-0.4488,  0.3403, -0.4576, -0.1742, -0.4726,  0.1484,  0.3400,  0.3729,
              0.2505, -0.1280],
            [ 0.2261,  0.4476, -0.1350,  0.2389, -0.2699,  0.2459,  0.2519,  0.3733,
              0.1752,  0.3251]]
            labels:
            [0, 3, 4, 3, 2, 0, 3, 8, 2, 5, 5, 6, 3, 5, 1, 9, 4, 7, 1, 1, 2, 5, 6, 0,
            6, 4, 5, 8, 1, 2, 7, 8]
            """
            loss = criterion(outputs, labels).to(device)  # 求loss

            loss.backward()  # 反向传播求梯度
            optimizer.step()  # 更新所有参数

            running_loss += loss.item()
            loss_list.append(loss.item())
            if i % num_print == num_print - 1:
                print('[%d epoch, %d] loss: %.6f' % (epoch + 1, i + 1, running_loss / num_print))
                running_loss = 0.0
        lr_1 = optimizer.param_groups[0]['lr']
        print('learn_rate : %.15f' % lr_1)
        scheduler.step()  # 调整学习率

    end = time.time()
    print('time:{}'.format(end-start))
    torch.save(model, './model.pkl')   #保存模型


# train()  # 训练
model = torch.load('./model.pkl')  #加载模型

# test
model.eval()
correct = 0.0
total = 0
with torch.no_grad():  # 测试集不需要反向传播
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将输入和目标在每一步都送入GPU
        outputs = model(inputs)  # 输出[32,10]
        pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        total += inputs.size(0)
        """
        labels:
            [0, 3, 4, 3, 2, 0, 3, 8, 2, 5, 5, 6, 3, 5, 1, 9, 4, 7, 1, 1, 2, 5, 6, 0,
            6, 4, 5, 8, 1, 2, 7, 8]
        """
        correct += torch.eq(pred, labels).sum().item()
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