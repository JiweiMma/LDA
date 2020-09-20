import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
# X为样本特征，y为样本输出， coef为回归系数，共1000个样本，每个样本1个特征
X, y, coef =make_regression(n_samples=1000, n_features=1,noise=10, coef=True)

# 画图
plt.scatter(X, y,  color='yellow')
plt.plot(X, X*coef, color='blue',linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
