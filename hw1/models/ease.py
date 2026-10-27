import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Recommender
class EASE(Recommender):
    def __init__(self, reg=500.0):
        self.reg = float(reg)
    def fit(self, positive):
        self.history = positive.astype(np.float64).tocsr()
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::len(gram) + 1] += self.reg
        precision = cho_solve(cho_factor(gram, lower=True), np.eye(len(gram)))
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0.0)
        self.weights = self.weights.astype(np.float32)
        return self
    def score(self, users):
        return np.asarray(self.history[np.asarray(users)] @ self.weights)
