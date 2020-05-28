# 使用numpy实现一个神经网络
import numpy as np

# n为样本大小，d_in为输入维度,d_out为输出维度,h为隐藏层维度
n, d_in, h, d_out = 64, 1000, 100, 10

# 随机生成数据
x = np.random.randn(n, d_in)  # 64 * 1000
y = np.random.randn(n, d_out)  # 64 * 10
# 随机初始化权重
# 输入层到隐藏层的权重（1000，100）
w1 = np.random.randn(d_in, h)  # 1000 * 100
# 隐藏层到输出层的权重（100，10）
w2 = np.random.randn(h, d_out)  # 100 * 10
# 设置学习率
learning_rate = 1e-6

# 500次迭代
for t in range(500):
    # 前向传播，输出层到隐藏层
    temp = np.dot(x, w1)  # 64 * 100
    # relu激活
    temp_relu = np.maximum(temp, 0)  # 64 * 100
    # 隐藏层到输出层
    y_pred = np.dot(temp_relu, w2)  # 64 * 10

    # 计算损失函数
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 反向传播，基于loss 计算w1和w2的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    # 计算w2梯度
    grad_w2 = np.dot(temp_relu.T, grad_y_pred)
    grad_temp_relu = np.dot(grad_y_pred, w2.T)
    grad_temp = grad_temp_relu.copy()
    # relu的反向传播
    grad_temp_relu[temp<0] = 0
    # 计算w1梯度
    grad_w1 = np.dot(x.T, grad_temp)
    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    #print(w1,w2)