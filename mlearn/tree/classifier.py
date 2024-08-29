import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        """
        初始化决策树分类器。

        参数:
        - max_depth: int 或 None, 决策树的最大深度。None 表示无限制。
        """
        self.max_depth = max_depth  # 最大深度
        self.tree = None  # 存储树的结构

    def _gini(self, y):
        """
        计算基尼不纯度（Gini impurity）。

        参数:
        - y: numpy.ndarray, shape (n_samples,), 样本标签。

        返回:
        - gini: float, 基尼不纯度。
        """
        # 计算各类别的概率
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        # 计算基尼不纯度
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _best_split(self, X, y):
        """
        找到最佳的分裂点。

        参数:
        - X: numpy.ndarray, shape (n_samples, n_features), 特征矩阵。
        - y: numpy.ndarray, shape (n_samples,), 标签向量。

        返回:
        - best_idx: int, 最佳特征索引。
        - best_thr: float, 最佳分割阈值。
        """
        m, n = X.shape
        if m <= 1:
            return None, None

        # 计算整体基尼不纯度
        parent_gini = self._gini(y)
        best_gini = 0
        best_idx, best_thr = None, None

        # 遍历每一个特征
        for idx in range(n):
            # 对当前特征进行排序
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # 初始化左右分支的样本数和类别计数
            left_count = np.zeros_like(np.unique(y))
            # 可以用来计算每个标签的出现频率
            right_count = np.bincount(classes)

            # 遍历每一个阈值
            # 通过遍历特征的所有可能分裂点，计算每个分裂点的基尼不纯度，
            # 并选择能够最小化基尼不纯度的分裂点，从而构建出能够有效分类的决策树。
            for i in range(1, m):
                c = classes[i - 1]
                left_count[c] += 1
                right_count[c] -= 1

                # 如果当前值和下一个值相同，跳过
                if thresholds[i] == thresholds[i - 1]:
                    continue

                # 计算左、右分支的基尼不纯度
                left_gini = 1.0 - np.sum((left_count / i) ** 2)
                right_gini = 1.0 - np.sum((right_count / (m - i)) ** 2)

                # 计算加权平均基尼不纯度
                gini = (i * left_gini + (m - i) * right_gini) / m

                # 如果找到更好的分割点，更新最优分割点
                if gini < best_gini or best_thr is None:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        """
        递归地生长决策树。

        参数:
        - X: numpy.ndarray, shape (n_samples, n_features), 特征矩阵。
        - y: numpy.ndarray, shape (n_samples,), 标签向量。
        - depth: int, 当前树的深度。

        返回:
        - tree: dict, 决策树的结构。
        """
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)

        # 构建叶节点, 分类最多的点被选为第0级预测值
        node = {
            'predicted_class': predicted_class
        }

        # 检查是否达到最大深度或样本数量小于等于1
        if depth < self.max_depth and len(np.unique(y)) > 1:
            # 找到最佳分割
            idx, thr = self._best_split(X, y)
            if idx is not None:
                # 递归构建左右子树
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature_index'] = idx
                node['threshold'] = thr
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)

        return node

    def fit(self, X, y):
        """
        训练决策树模型。

        参数:
        - X: numpy.ndarray, shape (n_samples, n_features), 训练数据。
        - y: numpy.ndarray, shape (n_samples,), 目标值。
        """
        self.tree = self._grow_tree(X, y)

    def _predict(self, inputs):
        """
        递归预测样本的类别。

        参数:
        - inputs: numpy.ndarray, shape (n_features,), 单个样本特征。

        返回:
        - predicted_class: int, 预测的类别。
        """
        node = self.tree
        while 'left' in node:
            if inputs[node['feature_index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted_class']

    def predict(self, X):
        """
        预测样本类别。

        参数:
        - X: numpy.ndarray, shape (n_samples, n_features), 样本特征。

        返回:
        - predictions: numpy.ndarray, shape (n_samples,), 预测的类别。
        """
        return np.array([self._predict(inputs) for inputs in X])
