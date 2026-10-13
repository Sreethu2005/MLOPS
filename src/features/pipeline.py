"""
Feature engineering pipeline module using scikit-learn ColumnTransformer.
Fits strictly on training data only to prevent train/serve skew and data leakage.
Implements robust tabular imputers compatible with sklearn BaseEstimator.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class NumericalImputer(BaseEstimator, TransformerMixin):
    """Computes and imputes column-wise median values from training data."""
    __module__ = "src.features.pipeline"
    def __init__(self):
        self.medians_ = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.medians_ = X.median().to_dict()
        else:
            X_df = pd.DataFrame(X)
            self.medians_ = X_df.median().to_dict()
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, med in self.medians_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(med)
            elif isinstance(col, int) and col < X_df.shape[1]:
                X_df.iloc[:, col] = X_df.iloc[:, col].fillna(med)
        return X_df.to_numpy(dtype=np.float64)

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Computes and imputes column-wise mode values from training data."""
    __module__ = "src.features.pipeline"
    def __init__(self):
        self.modes_ = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.modes_ = {
                col: (X[col].mode().iloc[0] if not X[col].mode().empty else "Missing")
                for col in X.columns
            }
        else:
            X_df = pd.DataFrame(X)
            self.modes_ = {
                col: (X_df[col].mode().iloc[0] if not X_df[col].mode().empty else "Missing")
                for col in X_df.columns
            }
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, mode_val in self.modes_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(mode_val).astype(str)
            elif isinstance(col, int) and col < X_df.shape[1]:
                X_df.iloc[:, col] = X_df.iloc[:, col].fillna(mode_val).astype(str)
        return X_df.to_numpy(dtype=object)

# Provide fallback binding for unpickling artifacts originally saved when executed as __main__
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    if not hasattr(main_mod, "NumericalImputer"):
        setattr(main_mod, "NumericalImputer", NumericalImputer)
    if not hasattr(main_mod, "CategoricalImputer"):
        setattr(main_mod, "CategoricalImputer", CategoricalImputer)

def build_preprocessor(config: dict) -> ColumnTransformer:
    """
    Creates an unfitted ColumnTransformer with numeric scaling and categorical one-hot encoding.
    """
    numerical_cols = config["features"]["numerical_cols"]
    categorical_cols = config["features"]["categorical_cols"]

    num_pipeline = Pipeline([
        ("imputer", NumericalImputer()),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", CategoricalImputer()),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor

def get_transformed_feature_names(preprocessor: ColumnTransformer, config: dict = None) -> list:
    """
    Extracts the feature names output by the fitted ColumnTransformer.
    """
    if config is None:
        config = load_config("configs/config.yaml")

    numerical_cols = config["features"]["numerical_cols"]
    categorical_cols = config["features"]["categorical_cols"]

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()

    return numerical_cols + encoded_cat_names

def fit_and_save_preprocessor(train_df: pd.DataFrame, config: dict, output_path: str = "models/preprocessor.joblib") -> ColumnTransformer:
    """
    Fits the preprocessor strictly on the training dataset and serializes it.
    """
    preprocessor = build_preprocessor(config)
    logger.info("Fitting feature preprocessor exclusively on training data...")
    preprocessor.fit(train_df)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_file)
    logger.info(f"Fitted preprocessor serialized to: {out_file}")

    return preprocessor

def transform_data(df: pd.DataFrame, preprocessor: ColumnTransformer, config: dict):
    """
    Transforms dataframe features using the fitted preprocessor and extracts target array.
    """
    target_col = config["data"]["target_col"]
    X_transformed = preprocessor.transform(df)
    
    y = None
    if target_col in df.columns:
        y = df[target_col].values

    return X_transformed, y

def run_feature_pipeline(config_path: str = "configs/config.yaml"):
    """
    End-to-end execution of feature pipeline on saved training split.
    """
    config = load_config(config_path)
    train_path = config["data"]["train_data_path"]
    val_path = config["data"]["val_data_path"]
    test_path = config["data"]["test_data_path"]

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    preprocessor = fit_and_save_preprocessor(train_df, config)

    feature_names = get_transformed_feature_names(preprocessor, config)
    logger.info(f"Extracted {len(feature_names)} engineered features.")

    return preprocessor, feature_names

if __name__ == "__main__":
    run_feature_pipeline()
