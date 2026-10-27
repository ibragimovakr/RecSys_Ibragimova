import numpy as np
from scipy import sparse
from .base import Recommender

def target_matrix(frame, shape, targets=None):
    values = ((frame.rating.to_numpy(dtype=float) - 3.0) / 2.0
              if targets is None else np.asarray(targets, dtype=float))
    return sparse.csr_matrix((values, (frame.u, frame.i)), shape=shape)

class ExplicitALS(Recommender):
    def __init__(self, factors=32, reg=5.0, iterations=12, seed=42):
        self.factors, self.reg = int(factors), float(reg)
        self.iterations, self.seed = int(iterations), int(seed)
    def fit(self, ratings):
        ratings = ratings.astype(np.float64).tocsr()
        ratings.sort_indices()
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (ratings.shape[0], self.factors))
        self.item_factors = rng.normal(0, 0.1, (ratings.shape[1], self.factors))
        self.loss_history = []
        transposed = ratings.T.tocsr()
        for _ in range(self.iterations):
            self._update(ratings, self.user_factors, self.item_factors)
            self._update(transposed, self.item_factors, self.user_factors)
            self.loss_history.append(self.objective(ratings))
        return self
    def _update(self, matrix, current, other):
        identity = self.reg * np.eye(self.factors)
        for row in range(matrix.shape[0]):
            start, stop = matrix.indptr[row:row + 2]
            indices, targets = matrix.indices[start:stop], matrix.data[start:stop]
            if len(indices) == 0:
                current[row] = 0.0
                continue
            neighbors = other[indices]
            current[row] = np.linalg.solve(
                neighbors.T @ neighbors + identity, neighbors.T @ targets)
    def objective(self, ratings):
        coo = ratings.tocoo()
        prediction = np.einsum("ij,ij->i", self.user_factors[coo.row],
                               self.item_factors[coo.col])
        return float(np.sum((coo.data - prediction) ** 2) + self.reg * (
            np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)))
    def score(self, users):
        return self.user_factors[np.asarray(users)] @ self.item_factors.T
