from torch import nn
import torch
import numpy as np
import fcn
import os.path as osp


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):
    pretrained_model = \
        osp.expanduser('./fcn32s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vM2oya3k0Zlgtekk',
            path=cls.pretrained_model,
            md5='8acf386d722dc3484625964cbe2aba49',
        )

    def __init__(self, n_class=21):  # n_class为数据的类别
        super(FCN32s, self).__init__()

        # conv1 输入3通道，卷积3*3，输出64通道
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)  # padding为了适应任意大小的输入图片
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # ceil_mode 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()  # 随机将输入张量中整个通道设置为0。对于每次前向调用，被置0的通道都是随机的。

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)  # 输出通道跟类别保持一致，一个类别一个热图
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)  # 反卷积操作

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()  # 剪裁的尺寸跟输入图像的尺寸有关

        return h

    def copy_params_from_fcn32(self, fcn32):

        self.conv1_1.weight = torch.nn.Parameter(fcn32['conv1_1.weight'])
        self.conv1_1.bias = torch.nn.Parameter(fcn32['conv1_1.bias'])
        self.conv1_2.weight = torch.nn.Parameter(fcn32['conv1_2.weight'])
        self.conv1_2.bias = torch.nn.Parameter(fcn32['conv1_2.bias'])

        self.conv2_1.weight = torch.nn.Parameter(fcn32['conv2_1.weight'])
        self.conv2_1.bias = torch.nn.Parameter(fcn32['conv2_1.bias'])
        self.conv2_2.weight = torch.nn.Parameter(fcn32['conv2_2.weight'])
        self.conv2_2.bias = torch.nn.Parameter(fcn32['conv2_2.bias'])

        self.conv3_1.weight = torch.nn.Parameter(fcn32['conv3_1.weight'])
        self.conv3_1.bias = torch.nn.Parameter(fcn32['conv3_1.bias'])
        self.conv3_2.weight = torch.nn.Parameter(fcn32['conv3_2.weight'])
        self.conv3_2.bias = torch.nn.Parameter(fcn32['conv3_2.bias'])
        self.conv3_3.weight = torch.nn.Parameter(fcn32['conv3_3.weight'])
        self.conv3_3.bias = torch.nn.Parameter(fcn32['conv3_3.bias'])

        self.conv4_1.weight = torch.nn.Parameter(fcn32['conv4_1.weight'])
        self.conv4_1.bias = torch.nn.Parameter(fcn32['conv4_1.bias'])
        self.conv4_2.weight = torch.nn.Parameter(fcn32['conv4_2.weight'])
        self.conv4_2.bias = torch.nn.Parameter(fcn32['conv4_2.bias'])
        self.conv4_3.weight = torch.nn.Parameter(fcn32['conv4_3.weight'])
        self.conv4_3.bias = torch.nn.Parameter(fcn32['conv4_3.bias'])

        self.conv5_1.weight = torch.nn.Parameter(fcn32['conv5_1.weight'])
        self.conv5_1.bias = torch.nn.Parameter(fcn32['conv5_1.bias'])
        self.conv5_2.weight = torch.nn.Parameter(fcn32['conv5_2.weight'])
        self.conv5_2.bias = torch.nn.Parameter(fcn32['conv5_2.bias'])
        self.conv5_3.weight = torch.nn.Parameter(fcn32['conv5_3.weight'])
        self.conv5_3.bias = torch.nn.Parameter(fcn32['conv5_3.bias'])

        self.fc6.weight = torch.nn.Parameter(fcn32['fc6.weight'])
        self.fc6.bias = torch.nn.Parameter(fcn32['fc6.bias'])

        self.fc7.weight = torch.nn.Parameter(fcn32['fc7.weight'])
        self.fc7.bias = torch.nn.Parameter(fcn32['fc7.bias'])

        self.score_fr.weight = torch.nn.Parameter(fcn32['score_fr.weight'])
        self.score_fr.bias = torch.nn.Parameter(fcn32['score_fr.bias'])

        self.upscore.weight = torch.nn.Parameter(fcn32['upscore.weight'])
