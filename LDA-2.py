import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def lda(data, target, n_dim):

    #该函数是去除数组中的重复数字，并进行排序之后输出。
    clusters = np.unique(target)

    if n_dim > len(clusters)-1:
        print("K is too much")
        print("please input again")
        exit(0)

    #计算类内散度矩阵
    Sw = np.zeros((data.shape[1],data.shape[1]))
    for i in clusters:
        datai = data[target == i]
        datai = datai-datai.mean(0)
        Swi = np.mat(datai).T*np.mat(datai)
        Sw += Swi

    #计算类间散度矩阵
    SB = np.zeros((data.shape[1],data.shape[1]))
    #计算所有样本的平均值
    u = data.mean(0)
    for i in clusters:
        Ni = data[target == i].shape[0]
        ui = data[target == i].mean(0)
        SBi = Ni*np.mat(ui - u).T*np.mat(ui - u)
        SB += SBi

    # 函数是求逆矩阵
    S = np.linalg.inv(Sw)*SB

    #计算矩阵特征向量，特征值
    eigVals,eigVects = np.linalg.eig(S)
    #对特征值进行排序
    eigValInd = np.argsort(eigVals)
    # 取最大的特征向量对应的index序号
    eigValInd = eigValInd[:(-n_dim-1):-1]
    # 根据取到的特征值对特征向量进行排序
    w = eigVects[:,eigValInd]
    #将维后的样本集
    data_ndim = np.dot(data, w)

    return data_ndim

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    data_1 = lda(X, Y, 2)

    #n_components:整形，可选维数约减的维数  类别数-1
    data_2 = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, Y)


    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("LDA")
    plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)

    plt.subplot(122)
    plt.title("sklearn_LDA")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    plt.savefig("LDA.png",dpi=600)
    plt.show()
