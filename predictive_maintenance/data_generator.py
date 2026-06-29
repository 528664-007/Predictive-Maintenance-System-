"""
Synthetic data generation for Predictive Maintenance System
"""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class IndustrialDataGenerator:
    """Generates synthetic industrial sensor data with realistic failure patterns"""

    def __init__(self, n_machines: int = 10, n_cycles: int = 300,
                 failure_window: int = 30, random_seed: int = 42):
        self.n_machines = n_machines
        self.n_cycles = n_cycles
        self.failure_window = failure_window
        self.random_seed = random_seed
        np.random.seed(random_seed)
        logger.info(f"Initialized DataGenerator: {n_machines} machines, "
                    f"{n_cycles} cycles, {failure_window} failure window")

    def _generate_base_sensor_data(self, n_samples: int) -> pd.DataFrame:
        return pd.DataFrame({
            'temperature':    np.random.normal(70,   5,  n_samples),
            'vibration':      np.random.normal(0.5,  0.1, n_samples),
            'pressure':       np.random.normal(100,  8,  n_samples),
            'rotation_speed': np.random.normal(1500, 50, n_samples),
        })

    def _apply_degradation_pattern(self, data: pd.DataFrame,
                                   failure_start_idx: int) -> pd.DataFrame:
        n_degradation = len(data) - failure_start_idx
        if n_degradation <= 0:
            return data

        degradation_multiplier = np.exp(np.linspace(0, 2, n_degradation)) - 1

        data = data.copy()
        data.loc[failure_start_idx:, 'temperature'] += degradation_multiplier * 15
        data.loc[failure_start_idx:, 'vibration']   += degradation_multiplier * 2
        data.loc[failure_start_idx:, 'pressure']    -= degradation_multiplier * 10
        data.loc[failure_start_idx:, 'rotation_speed'] += np.random.normal(0, 20, n_degradation)

        data['temperature']    = np.clip(data['temperature'],    20,   150)
        data['vibration']      = np.clip(data['vibration'],      0,    10)
        data['pressure']       = np.clip(data['pressure'],       20,   150)
        data['rotation_speed'] = np.clip(data['rotation_speed'], 1000, 2000)

        return data

    def generate_dataset(self) -> pd.DataFrame:
        logger.info("Starting synthetic data generation...")
        all_machines_data = []
        base_timestamp = datetime(2024, 1, 1)

        for machine_id in range(1, self.n_machines + 1):
            logger.debug(f"Generating data for machine {machine_id}")
            machine_data = self._generate_base_sensor_data(self.n_cycles)
            will_fail = np.random.random() < 0.7

            if will_fail:
                failure_cycle = np.random.randint(100, self.n_cycles)
                failure_start = max(0, failure_cycle - self.failure_window)
                machine_data = self._apply_degradation_pattern(machine_data, failure_start)
                machine_data['failure'] = 0
                machine_data.loc[failure_cycle:, 'failure'] = 1
            else:
                machine_data['failure'] = 0
                if np.random.random() < 0.3:
                    wear_start = np.random.randint(200, self.n_cycles - 20)
                    machine_data.loc[wear_start:, 'temperature'] *= 1.05
                    machine_data.loc[wear_start:, 'vibration']   *= 1.1

            machine_data['machine_id'] = machine_id
            machine_data['cycle'] = range(1, self.n_cycles + 1)
            machine_data['timestamp'] = [
                base_timestamp + timedelta(hours=i * 8)
                for i in range(self.n_cycles)
            ]
            all_machines_data.append(machine_data)

        final_dataset = pd.concat(all_machines_data, ignore_index=True)
        column_order = [
            'timestamp', 'machine_id', 'cycle',
            'temperature', 'vibration', 'pressure', 'rotation_speed', 'failure',
        ]
        final_dataset = final_dataset[column_order]
        final_dataset = final_dataset.sort_values(['machine_id', 'cycle']).reset_index(drop=True)

        logger.info(f"Generated dataset with {len(final_dataset)} records")
        logger.info(f"Failure rate: {final_dataset['failure'].mean():.2%}")
        return final_dataset

    def save_dataset(self, data: pd.DataFrame, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")


def generate_and_save_data(config) -> pd.DataFrame:
    generator = IndustrialDataGenerator(
        n_machines=config.n_machines,
        n_cycles=config.n_cycles,
        failure_window=config.failure_window,
        random_seed=config.random_seed,
    )
    data = generator.generate_dataset()
    generator.save_dataset(data, config.data_path)
    return data
