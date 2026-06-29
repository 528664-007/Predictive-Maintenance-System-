"""
Feature engineering module for Predictive Maintenance System
"""
import pandas as pd
import numpy as np
from typing import List
import logging
from config import FeatureConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced time-series feature engineering for predictive maintenance"""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_columns: List[str] = []
        logger.info(f"FeatureEngineer init: windows={config.rolling_windows}, "
                    f"lags={config.lag_features}")

    # ------------------------------------------------------------------ #
    def calculate_rul(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating Remaining Useful Life (RUL)...")
        data = data.copy()
        data['rul'] = 0

        for machine_id in data['machine_id'].unique():
            mask = data['machine_id'] == machine_id
            machine_data = data.loc[mask].sort_values('cycle')
            failure_rows = machine_data[machine_data['failure'] == 1]

            if not failure_rows.empty:
                failure_cycle = failure_rows['cycle'].min()
                data.loc[mask, 'rul'] = failure_cycle - machine_data['cycle'].values
            else:
                max_cycle = machine_data['cycle'].max()
                data.loc[mask, 'rul'] = max_cycle - machine_data['cycle'].values + 1

            data.loc[mask & (data['rul'] < 0), 'rul'] = 0

        rul_stats = data['rul'].describe()
        logger.info(f"RUL — mean: {rul_stats['mean']:.2f}, "
                    f"max: {rul_stats['max']:.2f}, min: {rul_stats['min']:.2f}")
        return data

    # ------------------------------------------------------------------ #
    def create_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating rolling window features...")
        data = data.copy()

        for machine_id in data['machine_id'].unique():
            mask = data['machine_id'] == machine_id
            for sensor in self.config.sensor_columns:
                for window in self.config.rolling_windows:
                    s = data.loc[mask, sensor]
                    data.loc[mask, f'{sensor}_rolling_mean_{window}'] = (
                        s.rolling(window, min_periods=1).mean())
                    data.loc[mask, f'{sensor}_rolling_std_{window}'] = (
                        s.rolling(window, min_periods=1).std())
                    data.loc[mask, f'{sensor}_rolling_min_{window}'] = (
                        s.rolling(window, min_periods=1).min())
                    data.loc[mask, f'{sensor}_rolling_max_{window}'] = (
                        s.rolling(window, min_periods=1).max())

        logger.info(f"Created rolling features for {len(self.config.rolling_windows)} windows")
        return data

    # ------------------------------------------------------------------ #
    def create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating lag features...")
        data = data.copy()

        for machine_id in data['machine_id'].unique():
            mask = data['machine_id'] == machine_id
            for sensor in self.config.sensor_columns:
                for lag in self.config.lag_features:
                    data.loc[mask, f'{sensor}_lag_{lag}'] = (
                        data.loc[mask, sensor].shift(lag))

        # FIX: fillna(method=...) deprecated in pandas 2.x — use .bfill() directly
        for sensor in self.config.sensor_columns:
            for lag in self.config.lag_features:
                col = f'{sensor}_lag_{lag}'
                data[col] = data.groupby('machine_id')[col].transform(
                    lambda s: s.bfill()
                )

        logger.info(f"Created lag features for lags: {self.config.lag_features}")
        return data

    # ------------------------------------------------------------------ #
    def create_cumulative_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating cumulative features...")
        data = data.copy()

        data['cumulative_hours'] = (
            data.groupby('machine_id')['cycle'].cumcount() * 8
        )

        for sensor in self.config.sensor_columns:
            data[f'{sensor}_cumulative_mean'] = (
                data.groupby('machine_id')[sensor]
                .transform(lambda x: x.ewm(span=20, adjust=False).mean())
            )
            data[f'{sensor}_rate_of_change'] = (
                data.groupby('machine_id')[sensor]
                .transform(lambda x: x.diff() / x.shift(1))
            )

        # FIX: fillna(method=...) deprecated — use .bfill() directly
        data = data.bfill()
        logger.info("Created cumulative features")
        return data

    # ------------------------------------------------------------------ #
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating interaction features...")
        data = data.copy()
        data['temp_vib_ratio']      = data['temperature'] / (data['vibration'] + 1e-6)
        data['temp_press_product']  = data['temperature'] * data['pressure']
        data['vib_per_rpm']         = data['vibration'] / (data['rotation_speed'] + 1e-6)
        data['press_per_rpm']       = data['pressure']  / (data['rotation_speed'] + 1e-6)
        logger.info("Created interaction features")
        return data

    # ------------------------------------------------------------------ #
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering pipeline...")
        data = self.calculate_rul(data)
        data = self.create_rolling_features(data)
        data = self.create_lag_features(data)
        data = self.create_cumulative_features(data)
        data = self.create_interaction_features(data)

        self.feature_columns = [
            c for c in data.columns
            if c not in {'timestamp', 'machine_id', 'cycle', 'failure', 'rul'}
        ]
        logger.info(f"Feature engineering done — {len(self.feature_columns)} features")
        return data
