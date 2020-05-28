import torch
import FCN_Net
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 加载网络结构与参数
def fcn32():
    model = FCN_Net.FCN32s()
    model_file = model.download()
    state_dict = torch.load(model_file)
    model.copy_params_from_fcn32(state_dict)
    return model

model = fcn32()
model.eval()

# 图片数据预处理
input_image = Image.open("result/dog.jpg")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# 只有正向，无反向
with torch.no_grad():
    output = model.forward(input_batch)

# 打印输出
for data in output[0]:
    plt.imshow(data)
    plt.show()
plt.imshow(output[0].argmax(0))
plt.show()
