"""
面积与房价曲线拟合
"""
import numpy as np
import matplotlib.pyplot as plt

# 定义x和y
x = np.array([50, 80, 100, 200])  # 房价面积
y = np.array([82, 118, 172, 302])  # 房价
plt.scatter(x, y)
plt.grid()


# plt.show()

# MAE Loss
def MAE_loss(y, y_hat):
    return np.mean(np.abs(y_hat - y))


def MSE_loss(y, y_hat):
    return np.mean(np.square(y_hat - y))


def linear(x, k, b):
    y = k * x + b
    return y


# 数据预处理，数据规范化
max_x = np.max(x)
max_y = 1
x = x / max_x
y = y / max_y


# K的梯度
def gradient_k(x, y, y_hat):
    n = len(y)
    gradient = 0
    for xi, yi, yi_hat in zip(list(x), list(y), list(y_hat)):
        gradient += (yi_hat - yi) * xi
    return gradient / n


# b的梯度
def gradient_b(y, y_hat):
    n = len(y)
    gradient = 0
    for yi, yi_hat in zip(list(y), list(y_hat)):
        gradient += (yi_hat - yi)
    return gradient / n


try_times = 1000
min_loss = float('inf')
# 初始化
current_k = 10
current_b = 10
learn_rate = 0.1

for i in range(try_times):
    y_hat = [linear(xi, current_k, current_b) for xi in list(x)]
    current_loss = MSE_loss(y, y_hat)
    if current_loss < min_loss:
        min_loss = current_loss
        best_k, best_b = current_k, current_b
        print('best_k is {}, best_b is {}'.format(best_k, best_b))

    # 计算k和b的梯度
    k_gradient = gradient_k(x, y, y_hat)
    b_gradient = gradient_b(y, y_hat)
    # 更新权重
    current_k = current_k - learn_rate * k_gradient
    current_b = current_b - learn_rate * b_gradient

best_k = best_k / max_x * max_y
best_b = best_b / max_x * max_y
print(best_k, best_b)

x = x * max_x
y = y * max_y
y_hat = best_k * x + best_b

plt.plot(x, y_hat, color='red')
plt.scatter(x, y)
plt.grid()
plt.show()
# # 暴力穷举
# # inf 代表正无穷
# min_loss = float('inf')
# for k in np.arange(-2, 2, 0.1):
# 	for b in np.arange(-10, 10, 0.1):
# 		y_hat = [linear(xi, k, b) for xi in list(x)]
# 		current_loss = MSE_loss(y, y_hat)
# 		if current_loss < min_loss:
# 			min_loss = current_loss
# 			best_k, best_b = k, b
# 			print('best_k is {}, best_b is {}'.format(best_k, best_b))

# 		#print(current_loss)
# #best_k = 0.8
# #best_b = 0.4
# y_hat = best_k * x +best_b
# plt.plot(x, y_hat, color='red')
# plt.grid() #显示网格
# plt.show()


