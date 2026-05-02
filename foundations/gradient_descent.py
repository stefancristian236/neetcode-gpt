class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init
        for _ in range(iterations):
            f_der = 2 * x
            x = x - learning_rate * f_der
        return round(x, 5)