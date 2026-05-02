import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.round(np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z))), 4)
        pass
