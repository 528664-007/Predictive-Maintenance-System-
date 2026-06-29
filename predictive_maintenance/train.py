"""
Model training module for Predictive Maintenance System

FIX SUMMARY vs original:
  1. Removed sklearn Pipeline — XGBoost early stopping requires passing a
     pre-scaled eval_set, which is impossible when scaler lives inside the
     pipeline (scaler not fitted yet at that point).
  2. Scaler is now fitted separately before XGBoost training.
  3. early_stopping_rounds / eval_metric come from ModelConfig (constructor),
     not patched via set_params() after construction.
  4. Feature names are stored explicitly and saved alongside the model so
     app.py can retrieve them without relying on pipeline internals.
  5. Saved artefact is a dict {'scaler': scaler, 'model': model,
     'feature_names': [...]} for clean loading in app.py.
"""
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any, List, Optional
import logging
from pathlib import Path
import shap
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

from config import ModelConfig, FeatureConfig
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler: RobustScaler = None
        self.model: xgb.XGBRegressor = None
        self.feature_names: List[str] = []
        logger.info("ModelTrainer initialized")

    # ------------------------------------------------------------------ #
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train scaler + XGBoost model."""
        logger.info("Starting model training...")
        self.feature_names = X_train.columns.tolist()

        # FIX: fit scaler FIRST, then build eval_set from already-scaled data
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # FIX: early_stopping_rounds & eval_metric live in ModelConfig constructor
        self.model = xgb.XGBRegressor(**self.config.xgboost_params)

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
        else:
            # Remove early stopping params if no validation data
            params = {k: v for k, v in self.config.xgboost_params.items()
                      if k not in ('early_stopping_rounds', 'eval_metric')}
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train_scaled, y_train)

        train_pred = self.model.predict(X_train_scaled)
        results: Dict[str, Any] = {
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'train_r2':   float(r2_score(y_train, train_pred)),
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(self.scaler.transform(X_val))
            results['val_rmse'] = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            results['val_r2']   = float(r2_score(y_val, val_pred))
            logger.info(f"Val  RMSE={results['val_rmse']:.4f}  R²={results['val_r2']:.4f}")

        logger.info(f"Train RMSE={results['train_rmse']:.4f}  R²={results['train_r2']:.4f}")
        return results

    # ------------------------------------------------------------------ #
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        X_scaled = self.scaler.transform(X_test)
        test_pred = self.model.predict(X_scaled)
        metrics = {
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
            'test_r2':   float(r2_score(y_test, test_pred)),
        }
        logger.info(f"Test RMSE={metrics['test_rmse']:.4f}  R²={metrics['test_r2']:.4f}")
        return metrics

    # ------------------------------------------------------------------ #
    def explain_prediction(self, X_instance: pd.DataFrame,
                           n_features: int = 10) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(X_instance)
        explainer  = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)

        explanation: Dict[str, Any] = {
            'shap_values':   shap_values[0],
            'base_value':    float(explainer.expected_value),
            'feature_names': self.feature_names,
            'prediction':    float(self.model.predict(X_scaled)[0]),
            'top_features':  [],
        }

        ranked = sorted(zip(self.feature_names, shap_values[0]),
                        key=lambda t: abs(t[1]), reverse=True)
        for feat, imp in ranked[:n_features]:
            explanation['top_features'].append({
                'feature':   feat,
                'importance': float(imp),
                'direction':  'positive' if imp > 0 else 'negative',
            })
        return explanation

    # ------------------------------------------------------------------ #
    def save_model(self, model_path: Optional[Path] = None) -> None:
        """Save scaler + model + feature names as a single dict."""
        if model_path is None:
            model_path = self.config.model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler':        self.scaler,
            'model':         self.model,
            'feature_names': self.feature_names,
        }, model_path)
        logger.info(f"Model artefact saved to {model_path}")

    def load_model(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load the saved dict and restore scaler/model/feature_names."""
        if model_path is None:
            model_path = self.config.model_path
        artefact = joblib.load(model_path)
        self.scaler        = artefact['scaler']
        self.model         = artefact['model']
        self.feature_names = artefact['feature_names']
        logger.info(f"Model artefact loaded from {model_path}")
        return artefact


# ---------------------------------------------------------------------- #
def run_training_pipeline(data_path: Path) -> Tuple['ModelTrainer', Dict[str, Any]]:
    logger.info("Starting training pipeline...")

    preprocessor   = DataPreprocessor()
    feature_config = FeatureConfig()
    model_config   = ModelConfig()

    data = preprocessor.load_data(data_path)
    data = preprocessor.validate_data(data)

    fe   = FeatureEngineer(feature_config)
    data = fe.engineer_features(data)

    train_data, val_data, test_data = preprocessor.split_data(
        data,
        test_size=model_config.test_size,
        validation_size=model_config.validation_size,
    )

    X_train, y_train = preprocessor.prepare_features_target(train_data, 'rul')
    X_val,   y_val   = preprocessor.prepare_features_target(val_data,   'rul')
    X_test,  y_test  = preprocessor.prepare_features_target(test_data,  'rul')

    trainer = ModelTrainer(model_config)
    results = trainer.train(X_train, y_train, X_val, y_val)
    test_metrics = trainer.evaluate(X_test, y_test)
    results.update(test_metrics)

    trainer.save_model()
    logger.info("Training pipeline completed")
    return trainer, results


if __name__ == "__main__":
    from config import setup_logging
    setup_logging()
    run_training_pipeline(Path("data/synthetic_data.csv"))
