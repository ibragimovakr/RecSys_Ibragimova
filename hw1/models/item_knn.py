import numpy as np
from scipy import sparse
from .base import Recommender
class ItemKNN(Recommender):
    def __init__(self, k=100, shrink=10.0):
        self.k = int(k)
        self.shrink = float(shrink)
    def fit(self, positive):
        self.history = positive.astype(np.float64).tocsr()
        overlap = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.maximum(np.diag(overlap), 1e-12))
        similarity = overlap / np.outer(norms, norms)
        if self.shrink:
            similarity *= overlap / (overlap + self.shrink)
        np.fill_diagonal(similarity, 0.0)
        rows, cols, values = [], [], []
        for j in range(similarity.shape[1]):
            neighbors = np.argsort(-similarity[:, j])[:self.k]
            neighbors = neighbors[similarity[neighbors, j] > 0]
            rows.extend(neighbors.tolist())
            cols.extend([j] * len(neighbors))
            values.extend(similarity[neighbors, j].tolist())
        self.weights = sparse.csr_matrix((values, (rows, cols)), shape=similarity.shape, dtype=np.float32)
        return self
    def score(self, users):
        return (self.history[np.asarray(users)] @ self.weights).toarray()
