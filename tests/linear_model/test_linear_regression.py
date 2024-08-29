import config
import numpy as np
from mlearn import linear_model

def get_dataset():
    # 创建一些示例数据
    np.random.seed(0)
    X = np.random.rand(100, 1)  # shape: (100, 1)
    y = 2 + 3 * X + np.random.randn(100, 1) * 0.1  # shape: (100, 1)
    return X, y


def test_linear_numpy_basic():
    X, y = get_dataset()

    # 创建并训练模型
    model = linear_model.LinearRegression()
    model.fit(X, y)  # X shape: (100, 1), y shape: (100, 1)

    # 打印结果
    print("Coefficients:", model.coef_)  # shape: (1,)
    print("Intercept:", model.intercept_)  # shape: ()
    print("**R² Score:", model.score(X, y))  # X shape: (100, 1), y shape: (100, 1)

    # 进行预测
    X_test = np.array([[0.5]])  # shape: (1, 1)
    print("Prediction for X=0.5:", model.predict(X_test))  # 输出 shape: (1,)


def test_linear_torch_basic():
    X, y = get_dataset()

    # 创建并训练模型
    model = linear_model.linear_regression.torch.LinearRegression()
    optim = 'sgd'
    optim_kwargs = {'lr': 0.01}
    model.fit(X, y, epochs=2000, optim=optim, **optim_kwargs)

    # 打印结果
    print("Coefficients:", model.coef_)  # shape: (1,)
    print("Intercept:", model.intercept_)  # shape: ()
    print("**R² Score:", model.score(X, y))  # X shape: (100, 1), y shape: (100, 1)

    # 进行预测
    X_test = np.array([[0.5]])  # shape: (1, 1)
    print("Prediction for X=0.5:", model.predict(X_test))  # 输出 shape: (1,)


if __name__ == '__main__':
    if 1:
        test_linear_numpy_basic()
    if 1:
        test_linear_torch_basic()
