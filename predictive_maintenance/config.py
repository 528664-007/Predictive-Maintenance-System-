"""
Configuration module for Predictive Maintenance System
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import logging


@dataclass
class DataConfig:
    """Data generation and processing configuration"""
    n_machines: int = 10
    n_cycles: int = 300
    failure_window: int = 30
    random_seed: int = 42
    data_path: Path = Path("data/synthetic_data.csv")


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    rolling_windows: List[int] = field(default_factory=lambda: [5, 20])
    lag_features: List[int] = field(default_factory=lambda: [1, 2])
    sensor_columns: List[str] = field(default_factory=lambda: [
        'temperature', 'vibration', 'pressure', 'rotation_speed'
    ])


@dataclass
class ModelConfig:
    """Model training configuration"""
    test_size: float = 0.2
    validation_size: float = 0.1
    model_path: Path = Path("models/xgboost_rul_model.pkl")
    scaler_path: Path = Path("models/robust_scaler.pkl")

    # XGBoost hyperparameters
    # FIX: early_stopping_rounds and eval_metric must live here
    #      (not patched in via set_params after construction)
    xgboost_params: Dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'early_stopping_rounds': 10,
        'eval_metric': 'rmse',
    })


@dataclass
class DashboardConfig:
    """Streamlit dashboard configuration"""
    rul_warning_threshold: int = 50
    rul_critical_threshold: int = 15
    page_title: str = "Predictive Maintenance Control Center"


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/predictive_maintenance.log'),
        ]
    )
    return logging.getLogger(__name__)
