"""Model factories (STUB). RF, XGBoost (GPU optional), Quantile RF.

Planned: build_model(name, params) returning sklearn-compatible estimators;
nested CV runner using cv.hex_block_folds for outer folds; SHAP + PDP in
models/interpret.py. Not implemented yet.
"""
def build_model(name: str, **params):
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    if name == "quantile_rf":
        from quantile_forest import RandomForestQuantileRegressor
        return RandomForestQuantileRegressor(**params)
    raise ValueError(f"unknown model {name}")
