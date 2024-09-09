import config
import numpy as np
from mlearn import tree
from mlearn.metrics import r2_score


def get_dataset():
    np.random.seed(0)
    X = np.random.rand(100, 2)  # shape: (100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # shape: (100,)
    return X, y

def test_tree_regressor_hello():
    # 示例数据
    X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y_train = np.array([0.1, 1.1, 1.0, 0.0])

    # 创建决策树分类器实例
    clf = tree.DecisionTreeRegressor(max_depth=3, ccp_alpha=0.0)

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测
    X_test = np.array([[0, 0], [1, 1]])
    predictions = clf.predict(X_test)
    print("Predictions:", predictions)

    y_true = np.array([0.1, 1.1])  # 真实目标值

    score = r2_score(y_true, predictions)
    print("r2_score:", score)


if __name__ == '__main__':
    if 1:
        test_tree_regressor_hello()
