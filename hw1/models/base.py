import numpy as np
def top_k(scores, k=10):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :min(k, scores.shape[1])]
    out = np.full((len(scores), k), -1, dtype=np.int64)
    values = np.take_along_axis(scores, order, axis=1)
    out[:, :order.shape[1]] = np.where(np.isfinite(values), order, -1)
    return out
class Recommender:
    def recommend(self, users, seen, k=10):
        scores = np.array(self.score(users), dtype=float, copy=True)
        history = seen[np.asarray(users)].tocoo()
        scores[history.row, history.col] = -np.inf
        return top_k(scores, k)
