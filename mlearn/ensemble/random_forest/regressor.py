import numpy as np
from ...tree.regressor import DecisionTreeRegressor

class RandomForestRegressor:
    """随机森林回归器 / Random Forest Regressor
    
    计算所有树的预测结果的平均值作为最终预测结果 / Calculate the average of all tree predictions as the final prediction
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt'):
        """初始化随机森林 / Initialize the random forest"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        """训练随机森林模型 / Train the random forest model"""
        # X shape: (n_samples, n_features), y shape: (n_samples,)
        self.n_features = X.shape[1]
        
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(self.n_features))
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(self.n_features))
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * self.n_features)
        
        # 训练每棵决策树 / Train each decision tree
        for _ in range(self.n_estimators):
            # 使用自助采样选择训练数据 / Use bootstrap sampling to select training data
            idxs = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        return self

    def predict(self, X):
        """使用随机森林进行预测 / Make predictions using the random forest"""
        # X shape: (n_samples, n_features), return shape: (n_samples,)
        # 收集每棵树的预测结果 / Collect predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # 计算平均值作为最终预测 / Calculate the mean as the final prediction
        return np.mean(tree_preds, axis=0)
