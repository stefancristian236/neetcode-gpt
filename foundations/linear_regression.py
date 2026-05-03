import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        Y = np.dot(X, weights)
        return np.round(Y, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        n = len(model_prediction)
        MSE_ = np.pow((model_prediction - ground_truth), 2)
        MSE = np.sum(MSE_) / n
        return np.round(MSE, 5)
