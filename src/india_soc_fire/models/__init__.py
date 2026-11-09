"""Model factory and wrappers for RF, XGBoost, and Quantile Regression Forest."""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


class ForestQuantileRegressor:
    """Conditional quantile forest estimator supporting `predict(X, quantiles=[0.05, 0.50, 0.95])`.

    Wraps `quantile_forest.RandomForestQuantileRegressor` if installed; otherwise
    computes empirical conditional quantiles across the fitted trees of a
    `scikit-learn` `RandomForestRegressor` (Meinshausen, 2006).
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42, n_jobs: int = 1, **kwargs: Any) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self._backend: Any = None
        self._uses_native_qrf: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForestQuantileRegressor":
        try:
            from quantile_forest import RandomForestQuantileRegressor

            self._backend = RandomForestQuantileRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs,
            )
            self._uses_native_qrf = True
        except ImportError:
            self._backend = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs,
            )
            self._uses_native_qrf = False
        self._backend.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(
        self,
        X: np.ndarray,
        quantiles: Sequence[float] | float | None = None,
    ) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if quantiles is None:
            if self._uses_native_qrf:
                return np.asarray(self._backend.predict(X_arr, quantiles=0.5), dtype=float)
            return np.asarray(self._backend.predict(X_arr), dtype=float)

        if self._uses_native_qrf:
            return np.asarray(self._backend.predict(X_arr, quantiles=list(quantiles) if not isinstance(quantiles, float) else quantiles), dtype=float)

        # Empirical distribution across individual decision trees (Meinshausen, 2006)
        tree_preds = np.stack([est.predict(X_arr) for est in self._backend.estimators_], axis=0)
        if isinstance(quantiles, (int, float)):
            return np.quantile(tree_preds, float(quantiles), axis=0)
        q_arr = np.quantile(tree_preds, list(quantiles), axis=0)  # shape (n_q, n_samples)
        return q_arr.T  # shape (n_samples, n_q)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self._backend.feature_importances_, dtype=float)


def build_model(name: str, **params: Any) -> Any:
    """Instantiate a regression estimator (`rf`, `xgb`, or `qrf`)."""
    name = name.lower()
    if name == "rf":
        defaults: dict[str, Any] = dict(n_estimators=200, n_jobs=1, random_state=42)
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    if name == "xgb":
        try:
            from xgboost import XGBRegressor

            defaults = dict(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1,
            )
            defaults.update(params)
            return XGBRegressor(**defaults)
        except ImportError:
            # Fallback to sklearn GradientBoostingRegressor if xgboost is not installed
            gb_params = {
                "n_estimators": params.get("n_estimators", 200),
                "max_depth": params.get("max_depth", 5),
                "learning_rate": params.get("learning_rate", 0.05),
                "subsample": params.get("subsample", 0.8),
                "random_state": params.get("random_state", 42),
            }
            return GradientBoostingRegressor(**gb_params)
    if name == "qrf":
        defaults = dict(n_estimators=200, n_jobs=1, random_state=42)
        defaults.update(params)
        return ForestQuantileRegressor(**defaults)
    raise ValueError(f"Unknown model '{name}'. Choose from: rf, xgb, qrf.")


from .metrics import compute_all_metrics, interval_metrics, lins_ccc, regression_metrics

__all__ = [
    "ForestQuantileRegressor",
    "build_model",
    "compute_all_metrics",
    "interval_metrics",
    "lins_ccc",
    "regression_metrics",
]
