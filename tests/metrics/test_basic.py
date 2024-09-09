import config
from mlearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import numpy as np


def test_metrics_basic():
    # 示例数据
    y_true_class = [0, 1, 1, 0]
    y_pred_class = [0, 1, 0, 1]

    y_true_reg = [3, -0.5, 2, 7]
    y_pred_reg = [2.5, 0.0, 2, 8]

    # 分类指标
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class)
    recall = recall_score(y_true_class, y_pred_class)
    f1 = f1_score(y_true_class, y_pred_class)

    print("Classification Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # 回归指标
    mse = mean_squared_error(y_true_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_reg, y_pred_reg)

    print("\nRegression Metrics:")
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R^2 Score:", r2)

if __name__ == '__main__':
    if 1:
        test_metrics_basic()
