import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x, w) + b
        y = 1 / (1 + np.exp(-z))
        dL_dz = (y - y_true) * y * (1 - y)
        
        gradient_w = dL_dz * x
        gradient_b = dL_dz * 1.0
        
        return (np.round(gradient_w, 5), np.round(gradient_b, 5))