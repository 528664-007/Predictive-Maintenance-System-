"""
Main execution script for Predictive Maintenance System
"""
import argparse
import logging
from pathlib import Path
import sys

from config import DataConfig, ModelConfig, DashboardConfig, setup_logging
from data_generator import generate_and_save_data
from train import run_training_pipeline

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Predictive Maintenance Pipeline")
    parser.add_argument("--mode", choices=["generate", "train", "all"], default="all")
    parser.add_argument("--data-path",  default="data/synthetic_data.csv")
    parser.add_argument("--model-path", default="models/xgboost_rul_model.pkl")
    parser.add_argument("--log-level",  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    data_path  = Path(args.data_path)
    model_path = Path(args.model_path)

    try:
        if args.mode in ("generate", "all"):
            logger.info("=" * 50 + " GENERATING DATA")
            data_config = DataConfig(data_path=data_path)
            data = generate_and_save_data(data_config)
            logger.info(f"Generated {len(data)} records")

        if args.mode in ("train", "all"):
            logger.info("=" * 50 + " TRAINING MODEL")
            trainer, results = run_training_pipeline(data_path)
            logger.info("Results:")
            for k, v in results.items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")

        logger.info("=" * 50)
        logger.info("Pipeline complete. Launch the dashboard with:")
        logger.info("  streamlit run app.py")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
