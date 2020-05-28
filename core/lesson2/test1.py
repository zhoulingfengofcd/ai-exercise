# -*- encoding=utf-8 -*-
__Author__ = "stubborn vegeta"

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap

class neuralNetwork(object):
    def __init__(self, X, Y, inputLayer, outputLayer, hiddenLayer=3,learningRate=0.01, epochs=10):
        """
        learningRate:学习率
        epochs:训练次数
        inputLayer:输入层节点数
        hiddenLayer:隐藏层节点数
        outputLayer:输出层节点数
        """
        self.learningRate = learningRate
        self.epochs = epochs
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        self.X = X
        self.Y = Y
        self.lenX,_ = np.shape(self.X)
        s=np.random.seed(0)
        # W1：输入层与隐藏层之间的权重；W2：隐藏层与输出层之间的权重；B1：隐藏层各节点的偏置项；B2：输出层各节点的偏置项
        self.W1 = np.array(np.random.random([self.inputLayer, self.hiddenLayer])*0.5)        #2*3
        self.B1 = np.array(np.random.random([self.lenX,self.hiddenLayer])*0.5)               #200*3
        self.W2 = np.array(np.random.random([self.hiddenLayer, self.outputLayer])*0.5)       #3*1
        self.B2 = np.array(np.random.random([self.lenX,self.outputLayer])*0.5)               #200*1

    def activationFunction(self, funcName:str, X):
        """
        激活函数
        sigmoid: 1/1+e^(-z)
        tanh: [e^z-e^(-z)]/[e^z+e^(-z)]
        softmax: e^zi/sum(e^j)
        """
        switch = {
                "sigmoid": 1/(1+np.exp(-X)),
                "tanh": np.tanh(X),
                # "softmax": np.exp(X-np.max(X))/np.sum(np.exp(X-np.max(X)), axis=0)
                }
        return switch[funcName]

    def activationFunctionGrad(self, funcName:str, X):
        """
        激活函数的导数
        """
        switch = {
                "sigmoid": np.exp(-X)/(1+np.exp(-X))**2,
                "tanh": 1-(np.tanh(X)**2),
                # "softmax": np.exp(X-np.max(X))/np.sum(np.exp(X-np.max(X)), axis=0)
                }
        return switch[funcName]

    def train(self, funcNameH:str, funcNameO:str):
        """
        funcNameH: 隐藏层激活函数
        funcNameO: 输出层激活函数
        """
        for i in range(0,self.epochs):
            j = np.random.randint(self.lenX)
            x = np.array([self.X[j]])
            y = np.array([self.Y[j]])
            b1 = np.array([self.B1[j]])
            b2 = np.array([self.B2[j]])
            # 前向传播
            zHidden = x.dot(self.W1)+b1
            z1 = self.activationFunction(funcNameH, zHidden)  #1*3
            zOutput = z1.dot(self.W2)+b2
            z2 = self.activationFunction(funcNameO, zOutput)  #1*1

            # 反向传播
            dW2 = (z2-y)*(z1.T*self.activationFunctionGrad(funcNameO,zOutput))
            db2 = (z2-y)*self.activationFunctionGrad(funcNameO,zOutput)
            dW1 = (z2-y)*(self.activationFunctionGrad(funcNameO,zOutput)*self.W2.T)*(x.T.dot(self.activationFunctionGrad(funcNameH,zHidden)))
            db1 = (z2-y)*(self.activationFunctionGrad(funcNameO,zOutput)*self.W2.T)*self.activationFunctionGrad(funcNameH,zHidden)

            #更新参数
            self.W2 -= self.learningRate*dW2
            self.B2[j] -= self.learningRate*db2[0]
            self.W1 -= self.learningRate*dW1
            self.B1[j] -= self.learningRate*db1[0]
        return 0

    def predict(self, xNewData, funcNameH:str, funcNameO:str):
        X = xNewData                                         #200*2
        N,_ = np.shape(X)
        yPredict = []
        for j in range(0,N):
            x = np.array([X[j]])
            b1 = np.array([self.B1[j]])
            b2 = np.array([self.B2[j]])
            # 前向传播
            zHidden = x.dot(self.W1)+b1
            z1 = self.activationFunction(funcNameH, zHidden)  #1*3
            zOutput = z1.dot(self.W2)+b2
            z2 = self.activationFunction(funcNameO, zOutput)  #1*1
            z2 = 1 if z2>0.5 else 0
            yPredict.append(z2)
        return yPredict,N


if __name__ == "__main__":
    X,Y = datasets.make_moons(200, noise=0.15)
    neural_network = neuralNetwork (X=X, Y=Y, learningRate=0.2, epochs=1000, inputLayer=2, hiddenLayer=3, outputLayer=1)
    funcNameH = "sigmoid"
    funcNameO = "tanh"
    neural_network.train(funcNameH=funcNameH,funcNameO=funcNameO)
    yPredict,N = neural_network.predict(xNewData=X,funcNameH=funcNameH,funcNameO=funcNameO)
    print("错误率：", sum((Y-yPredict)**2)/N)
    colormap = ListedColormap(['royalblue','forestgreen'])              # 用colormap中的颜色表示分类结果
    plt.subplot(1,2,1)
    plt.scatter(X[:,0],X[:,1],s=40, c=Y, cmap=colormap)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Standard data")
    plt.subplot(1,2,2)
    plt.scatter(X[:,0],X[:,1],s=40, c=yPredict, cmap=colormap)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted data")
    plt.show()