import config
import numpy as np
from mlearn import tree
from mlearn.metrics import accuracy_score


def get_dataset():
    np.random.seed(0)
    X = np.random.rand(100, 2)  # shape: (100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # shape: (100,)
    return X, y  # X shape: (100, 2), y shape: (100,)

def test_tree_classifier_hello():
    # 示例数据 | Example data
    X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y_train = np.array([0, 1, 1, 0])

    # 创建决策树分类器实例
    clf = tree.DecisionTreeClassifier(max_depth=3, ccp_alpha=0.1)

    # 训练模型 | Train the model
    clf.fit(X_train, y_train)

    # 预测 | Predict    
    X_test = np.array([[0, 0], [1, 1]])
    predictions = clf.predict(X_test)
    print("Predictions:", predictions)

    # 计算准确率 | Calculate accuracy
    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print("Accuracy:", accuracy)

def test_tree_classifier():
    X, y = get_dataset()
    print("X:", X)
    print("y:", y)

    # 创建并训练模型 | Create and train the model
    clf = tree.DecisionTreeClassifier(max_depth=3, ccp_alpha=0.0)
    clf.fit(X, y)  # X shape: (100, 2), y shape: (100,)

    # 进行预测 | Predict
    X_test = np.array([[0.5, 0.5], [0.8, 0.8]])  # shape: (2, 2)
    predictions = clf.predict(X_test)  # shape: (2,)
    print("Predictions:", predictions)

    # 计算准确率 | Calculate accuracy   
    y_pred = clf.predict(X)  # shape: (100,)
    accuracy = np.mean(y_pred == y)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    if 0:
        test_tree_classifier_hello()
    if 1:
        test_tree_classifier()