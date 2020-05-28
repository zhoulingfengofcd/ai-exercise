import cv2
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class ImageTransformer:

    def image_crop(self, path, x_size,y_size):
        """
        图像切割
        :param path: 图片路径
        :param x_size: 水平方向切几块
        :param y_size: 垂直方向切几块
        :return:
        """
        image = Image.open(path)
        size = image.size
        width, height =size[0]//x_size, size[1]//y_size
        print(width, height)
        for i in range(y_size):  # y方向循环
            for j in range(x_size):  # x方向循环
                module = image.crop((j*width, i*height, (j+1)*width, (i+1)*height))
                plt.imshow(module)
                plt.show()

    def rotate(self, img, aug_value, scale):  # your code here
        """
        旋转与缩放
        :param img: 图像
        :param aug_value: 旋转角度
        :param scale: 缩放值
        :return:
        """
        # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        M = cv2.getRotationMatrix2D((0.5 * img.shape[1], 0.5 * img.shape[0]), aug_value, scale)
        # 第三个参数为变换后的图像大小
        new_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        plt.imshow(new_img)
        plt.show()
        return new_img

    def perspective_transform(self, img, org_position, new_position):  # your code here
        """
        投影变换
        :param img: 图像
        :param org_position: 原图中需要被转换物体的四个顶点
        :param new_position: 设置在新图像中原图像的四个顶点的位置
        :return:
        """
        height, width = img.shape[:2]
        # 选取原图中需要被转换物体的四个顶点
        pts1 = np.float32(org_position)

        # 设置在新图像中原图像的四个顶点的位置
        pts2 = np.float32(new_position)

        # 计算转换M矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # 应用M矩阵到原图像
        dst = cv2.warpPerspective(img, M, (width, height))

        plt.figure(figsize=(10, 10), dpi=120)  # ,dpi = 120
        plt.subplot(121), plt.imshow(img), plt.title('Input')
        plt.subplot(122), plt.imshow(dst), plt.title('Output')

        plt.show()

    # ??? 不懂这个函数，想干嘛？？？？？？？？？？？有啥意义？
    def image_color_shift(self):
        pass


if __name__ == "__main__":
    transformer = ImageTransformer()
    img = cv2.imread('chedaoxian.jpg')  # your image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #transformer.image_crop(r'gugong.jpg', 2, 2)  # 原图切割4块
    #transformer.rotate(img, 90, 1)  # 旋转90度，不缩放
    #transformer.perspective_transform(img,
    #                                 [[0, 600], [1277, 600], [500, 400], [800, 400]],
    #                                 [[0, 848], [1277, 848], [0, 150], [1277, 150]]
    #                                 )

    ori_width, ori_height = img.shape[1], img.shape[0]
    paddingSize = 10000
    # 上左右填充
    img = cv2.copyMakeBorder(img, paddingSize, 0, paddingSize, paddingSize, cv2.BORDER_CONSTANT, value=0)
    print(img.shape)
    new_width, new_height = img.shape[1], img.shape[0]
    k = 10000
    kx = k
    ky = k
    transformer.perspective_transform(img,
                                      [[paddingSize, new_height], [paddingSize, paddingSize], [paddingSize+ori_width, paddingSize], [paddingSize+ori_width, new_height]],
                                      [[paddingSize, new_height], [paddingSize-kx, paddingSize-ky], [paddingSize+ori_width+kx, paddingSize-ky], [paddingSize+ori_width, new_height]]
                                      )



    # img = cv2.imread("saidao3.bmp")
    #
    # print(img)
    # paddingSize = 200
    # # 上左右填充
    # img = cv2.copyMakeBorder(img, paddingSize, 0, paddingSize, paddingSize, cv2.BORDER_CONSTANT, value=0)
    # width, height = img.shape[1], img.shape[0]
    # print(img.shape)
    # k = 400
    # kx = 3*k
    # ky = k
    # transformer.perspective_transform(img,
    #                                   [[paddingSize, height], [paddingSize, paddingSize], [paddingSize+180, paddingSize], [paddingSize+180, height]],
    #                                   [[paddingSize, height], [paddingSize-kx, paddingSize-ky], [paddingSize+180+kx, paddingSize-ky], [paddingSize+180, height]]
    #                                   )
    # plt.imshow(img)
    # plt.show()