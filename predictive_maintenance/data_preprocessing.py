"""
Data preprocessing module for Predictive Maintenance System
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data cleaning, validation, and preprocessing"""

    def __init__(self):
        logger.info("DataPreprocessor initialized")

    def load_data(self, filepath: Path) -> pd.DataFrame:
        logger.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        logger.info(f"Loaded {len(data)} records, {data['machine_id'].nunique()} machines")
        return data

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Validating data quality...")
        required_columns = [
            'timestamp', 'machine_id', 'cycle',
            'temperature', 'vibration', 'pressure', 'rotation_speed', 'failure',
        ]

        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
            # FIX: fillna(method=...) is deprecated in pandas 2.x
            #      Use .ffill() / .bfill() directly
            data = data.ffill().bfill()

        duplicates = data.duplicated(['machine_id', 'cycle']).sum()
        if duplicates:
            logger.warning(f"Removing {duplicates} duplicate records")
            data = data.drop_duplicates(['machine_id', 'cycle'], keep='last')

        self._validate_value_ranges(data)
        logger.info("Data validation completed")
        return data

    def _validate_value_ranges(self, data: pd.DataFrame) -> None:
        clamp_rules = {
            'temperature':    (20,   150),
            'vibration':      (0,    10),
            'pressure':       (20,   150),
            'rotation_speed': (1000, 2000),
        }
        for col, (lo, hi) in clamp_rules.items():
            outliers = (data[col] < lo) | (data[col] > hi)
            if outliers.any():
                logger.warning(f"{outliers.sum()} outliers in '{col}' — clamping")
                data[col] = data[col].clip(lo, hi)

    def split_data(self, data: pd.DataFrame,
                   test_size: float = 0.2,
                   validation_size: float = 0.1,
                   random_state: int = 42
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Splitting data by machine_id (no leakage)...")

        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss_test.split(data, groups=data['machine_id']))

        train_val_data = data.iloc[train_val_idx]
        test_data      = data.iloc[test_idx]

        # validation fraction relative to what remains after removing test
        validation_ratio = validation_size / (1 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(train_val_data,
                                                  groups=train_val_data['machine_id']))

        train_data = train_val_data.iloc[train_idx]
        val_data   = train_val_data.iloc[val_idx]

        logger.info(f"Train:      {len(train_data)} samples, {train_data['machine_id'].nunique()} machines")
        logger.info(f"Validation: {len(val_data)} samples, {val_data['machine_id'].nunique()} machines")
        logger.info(f"Test:       {len(test_data)} samples, {test_data['machine_id'].nunique()} machines")

        assert set(train_data['machine_id'].unique()).isdisjoint(set(val_data['machine_id'].unique()))
        assert set(train_data['machine_id'].unique()).isdisjoint(set(test_data['machine_id'].unique()))
        assert set(val_data['machine_id'].unique()).isdisjoint(set(test_data['machine_id'].unique()))

        return train_data, val_data, test_data

    def prepare_features_target(self, data: pd.DataFrame,
                                target_col: str = 'rul',
                                feature_cols: Optional[list] = None
                                ) -> Tuple[pd.DataFrame, pd.Series]:
        if feature_cols is None:
            exclude = {'timestamp', 'machine_id', 'cycle', 'failure', 'rul', 'machine_status'}
            feature_cols = [c for c in data.columns if c not in exclude]

        X = data[feature_cols].copy()
        y = data[target_col].copy() if target_col in data.columns else None
        logger.info(f"Prepared {len(feature_cols)} features, target='{target_col}'")
        return X, y
