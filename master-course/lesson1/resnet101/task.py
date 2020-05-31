import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import torch.nn.functional as F
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, img_size=None):
        super(CustomImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)
        # self.files = sorted(glob.glob("%s/*.*" % folder_path))  # 获取指定目录路径下的所有文件名
        self.img_size = img_size

    def __getitem__(self, index):
        sample, target = ImageFolder.__getitem__(self, index)
        # img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        # img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(sample, 0)
        # Resize
        img = resize(img, self.img_size)

        #return img_path, img

        # print("test")
        return img, target

    #def __len__(self):
    #    return len(self.files)


BATCH_SIZE = 32
epoch_num = 50  # 总迭代次数

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = CustomImageFolder("./flower_data/train", transform=data_transform, img_size=500)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_dataset = CustomImageFolder("./flower_data/train", transform=test_transform, img_size=500)
train_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for epoch in range(epoch_num):  # 总迭代次数
    running_loss = 0.0
    for inputs, labels in train_loader:  # 所有训练数据迭代
        print("test", inputs.shape)
        plt.imshow(np.transpose(inputs[0].numpy(), (1, 2, 0)))
        plt.show()