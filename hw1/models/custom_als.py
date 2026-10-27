import numpy as np
import pandas as pd
from .als import ExplicitALS, target_matrix
class CustomTargetALS(ExplicitALS):
    def __init__(self, beta=0.5, gamma=0.5, smoothing=5.0, **kwargs):
        super().__init__(**kwargs)
        self.beta, self.gamma, self.smoothing = beta, gamma, smoothing
    def make_targets(self, frame, primary_categories):
        work = frame[["u", "i", "rating"]].copy()
        work["category"] = np.asarray(primary_categories, dtype=object)[work.i]
        work["positive"] = (work.rating >= 4).astype(float)
        mu = float(work.rating.mean())
        user_stats = work.groupby("u").rating.agg(["sum", "count"])
        user_mean = (user_stats["sum"] + self.smoothing * mu) / (
            user_stats["count"] + self.smoothing)
        category_rate = work.groupby("category").positive.mean()
        uc = work.groupby(["u", "category"]).positive.agg(["sum", "count"])
        prior = uc.index.get_level_values("category").map(category_rate).to_numpy()
        preference = (uc["sum"].to_numpy() + self.smoothing * prior) / (uc["count"].to_numpy() + self.smoothing)
        lookup = pd.Series(preference, index=uc.index)
        keys = pd.MultiIndex.from_frame(work[["u", "category"]])
        category_delta = lookup.reindex(keys).to_numpy() - work.category.map(category_rate).to_numpy()
        targets = ((work.rating.to_numpy() - 3.0) / 2.0 + self.beta * (work.rating.to_numpy() - work.u.map(user_mean).to_numpy()) / 2.0 + self.gamma * category_delta)
        self.target_summary = {
            "mean": float(np.mean(targets)), "std": float(np.std(targets)),
            "min": float(np.min(targets)), "max": float(np.max(targets)),
            "beta": self.beta, "gamma": self.gamma, "smoothing": self.smoothing}
        return targets
    def fit(self, frame, shape, primary_categories):
        targets = self.make_targets(frame, primary_categories)
        return super().fit(target_matrix(frame, shape, targets))
