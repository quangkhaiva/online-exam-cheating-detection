import numpy as np, time

class EMA:
    """Làm trơn tín hiệu"""
    def __init__(self, alpha=0.6): self.alpha, self.value = alpha, None
    def update(self, x):
        self.value = x if self.value is None else self.alpha*x + (1-self.alpha)*self.value
        return self.value
