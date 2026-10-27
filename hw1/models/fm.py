import numpy as np
from scipy import sparse
from .base import Recommender
class FactorizationMachine(Recommender):
    def __init__(self, factors=16, reg=0.001, learning_rate=0.01, epochs=12, batch_size=4096, seed=42):
        self.factors, self.reg = int(factors), float(reg)
        self.learning_rate, self.epochs = float(learning_rate), int(epochs)
        self.batch_size, self.seed = int(batch_size), int(seed)
    def _predict(self, features):
        summed = features @ self.v
        interaction = 0.5 * np.sum(
            summed ** 2 - features.multiply(features) @ (self.v ** 2), axis=1)
        return self.bias + np.asarray(features @ self.w).ravel() + interaction
    def _loss_gradients(self, features, targets):
        n = len(targets)
        error = self._predict(features) - targets
        summed = features @ self.v
        squared_features = features.multiply(features)
        grad_w = np.asarray(features.T @ error).ravel() / n + self.reg * self.w
        grad_v = (np.asarray(features.T @ (error[:, None] * summed)) / n
                  - np.asarray(squared_features.T @ error)[:, None] * self.v / n
                  + self.reg * self.v)
        loss = 0.5 * np.mean(error ** 2) + 0.5 * self.reg * (
            np.sum(self.w ** 2) + np.sum(self.v ** 2))
        return float(loss), float(error.mean()), grad_w, grad_v
    def fit(self, frame, item_features, n_users):
        self.n_users = int(n_users)
        self.item_features = item_features.astype(np.float64).tocsr()
        user_features = sparse.csr_matrix(
            (np.ones(len(frame)), (np.arange(len(frame)), frame.u)),
            shape=(len(frame), n_users))
        features = sparse.hstack(
            [user_features, self.item_features[frame.i.to_numpy()]], format="csr")
        targets = (frame.rating.to_numpy(dtype=float) - 3.0) / 2.0
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(features.shape[1])
        self.v = rng.normal(0, 0.03, (features.shape[1], self.factors))
        self.bias = np.array(0.0)
        params = [self.bias, self.w, self.v]
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        step, self.loss_history = 0, []
        for _ in range(self.epochs):
            order = rng.permutation(len(frame))
            epoch_loss, count = 0.0, 0
            for start in range(0, len(frame), self.batch_size):
                rows = order[start:start + self.batch_size]
                loss, gb, gw, gv = self._loss_gradients(features[rows], targets[rows])
                step += 1
                for p, m, v, grad in zip(params, first, second, [gb, gw, gv]):
                    m *= 0.9
                    m += 0.1 * grad
                    v *= 0.999
                    v += 0.001 * np.asarray(grad) ** 2
                    p -= self.learning_rate * (m / (1 - 0.9 ** step)) / (
                        np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
                epoch_loss += loss * len(rows)
                count += len(rows)
            self.loss_history.append(epoch_loss / count)
        item_v = self.v[n_users:]
        self.item_embedding_sum = self.item_features @ item_v
        self.item_offset = np.asarray(self.item_features @ self.w[n_users:]).ravel()
        self.item_offset += 0.5 * np.sum(self.item_embedding_sum ** 2 - self.item_features.multiply(self.item_features) @ (item_v ** 2), axis=1)
        return self
    def score(self, users):
        users = np.asarray(users)
        return (self.v[users] @ self.item_embedding_sum.T + self.item_offset[None, :]
                + self.w[users, None] + self.bias)
