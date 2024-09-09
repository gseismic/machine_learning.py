import numpy as np
from .utils import ensure_array


def accuracy_score(y_true, y_pred):
    """
    Compute the accuracy score.

    计算准确率。

    Parameters:
    - y_true: numpy.ndarray, shape为 (n_samples,) 的真实目标值。
    - y_pred: numpy.ndarray, shape为 (n_samples,) 的预测目标值。

    Returns:
    - accuracy: float，准确率。
    """
    y_true = ensure_array(y_true)
    y_pred = ensure_array(y_pred)
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    """
    Compute the precision score.

    计算精确率。

    Parameters:
    - y_true: numpy.ndarray, shape为 (n_samples,) 的真实目标值。
    - y_pred: numpy.ndarray, shape为 (n_samples,) 的预测目标值。

    Returns:
    - precision: float，精确率。
    """
    y_true = ensure_array(y_true)
    y_pred = ensure_array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall_score(y_true, y_pred):
    """
    Compute the recall score.

    计算召回率。

    Parameters:
    - y_true: numpy.ndarray, shape为 (n_samples,) 的真实目标值。
    - y_pred: numpy.ndarray, shape为 (n_samples,) 的预测目标值。

    Returns:
    - recall: float, recall score.
    - recall: float，召回率。
    """
    y_true = ensure_array(y_true)
    y_pred = ensure_array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    """
    Compute the F1 score.

    计算 F1 分数。

    Parameters:
    - y_true: numpy.ndarray, shape (n_samples,), true target values.
    - y_true: numpy.ndarray, shape为 (n_samples,) 的真实目标值。
    - y_pred: numpy.ndarray, shape (n_samples,), predicted target values.
    - y_pred: numpy.ndarray, shape为 (n_samples,) 的预测目标值。

    Returns:
    - f1: float, F1 score.
    - f1: float，F1 分数。
    """
    y_true = ensure_array(y_true)
    y_pred = ensure_array(y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def r2_score(y_true, y_pred):
    """
    Compute the R^2 score.

    计算 R^2 分数。

    Parameters 参数:
    - y_true: numpy.ndarray, shape为 (n_samples,) 的真实目标值。
    - y_pred: numpy.ndarray, shape为 (n_samples,) 的预测目标值。

    Returns 返回值:
    - r2: float，R^2 分数。
    """
    y_true = ensure_array(y_true)
    y_pred = ensure_array(y_pred)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

