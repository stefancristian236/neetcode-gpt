import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        x = y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred + 1e-7)
        n = len(y_true)
        loss = -np.sum(x) / n
        return float(np.round(loss, 4))

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = len(y_true)
        x = y_true * np.log(y_pred + 1e-7)
        loss = -np.sum(x) / n
        return float(np.round(loss, 4))
