import numpy as np
from mlearn import linear_model
from mlearn import metrics
import matplotlib.pyplot as plt

# get dataset | 获取数据集
np.random.seed(0)
X = np.random.rand(100, 1)  # shape: (100, 1)
y = 2 + 3 * X + np.random.randn(100, 1) * 0.1  # shape: (100, 1)

# create and train model | 创建并训练模型
model = linear_model.LinearRegression()
model.fit(X, y)  # X shape: (100, 1), y shape: (100, 1)
# print | 打印结果
print("Coefficients:", model.coef_)  # shape: (1,)
print("Intercept:", model.intercept_)  # shape: ()
print("**R² Score:", model.score(X, y))  # X shape: (100, 1), y shape: (100, 1)

# predict | 预测
X_test = np.array([[0.5]])  # shape: (1, 1)
print("Prediction for X=0.5:", model.predict(X_test))  # 输出 shape: (1,)

# plot | 绘图
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.savefig('linear_regression.png')
plt.show()