# 🔧 Predictive Maintenance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29%2B-red.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.43%2B-purple.svg?style=for-the-badge)

**Production-Grade Industrial Predictive Maintenance with Multivariate Time-Series Analysis**

[![GitHub stars](https://img.shields.io/github/stars/528664-007/Predictive-Maintenance-System-?style=social)](https://github.com/528664-007/Predictive-Maintenance-System-/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/528664-007/Predictive-Maintenance-System-?style=social)](https://github.com/528664-007/Predictive-Maintenance-System-/network/members)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/528664-007/Predictive-Maintenance-System-/pulls)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technical Deep Dive](#-technical-deep-dive)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dashboard Features](#-dashboard-features)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [Roadmap](#-roadmap)
- [References](#-references)
- [License](#-license)
- [Output](#-output)
- [Contact](#-contact)

---

## 🎯 Overview

A **Predictive Maintenance System** designed for Industry 4.0 environments. This solution uses multivariate time-series sensor data to predict equipment failures before they occur, enabling proactive maintenance and minimizing costly downtime.

### 🏭 Problem Statement
Industrial equipment failures result in:
- **Unplanned downtime**: industry estimates put the cost in manufacturing at around $260,000 per hour
- **Production loss**: cascading effects across supply chains
- **Safety risks**: potential hazards for operators
- **Reactive maintenance**: commonly cited as several times more expensive than preventive approaches

### 💡 Solution
This system provides:
- **RUL prediction**: XGBoost-based regression for Remaining Useful Life estimation
- **Multi-sensor fusion**: feature engineering from temperature, vibration, pressure, and rotation data
- **Explainable AI**: SHAP-based interpretability for actionable insights
- **Interactive dashboard**: Streamlit-powered control center for operators and analysts

---

## ✨ Key Features

### 🔬 Advanced Analytics
- **Multivariate time-series processing**: handles complex sensor relationships
- **Exponential degradation modeling**: realistic failure pattern simulation
- **Rolling window features**: 5-cycle and 20-cycle windows for trend detection
- **Lag features**: temporal dependencies at t-1 and t-2
- **Cumulative operating metrics**: EWM-based cumulative statistics and rate of change

### 🤖 Machine Learning
- **XGBoost regression**: gradient boosting tuned for RUL prediction
- **Group-aware splitting**: prevents data leakage across machine IDs
- **Robust scaling**: outlier-resistant feature normalization
- **Early stopping**: overfitting prevention using a validation set
- **Model persistence**: Joblib serialization for deployment

### 📊 Interpretability
- **SHAP force plots**: individual prediction explanations
- **Feature importance ranking**: top 15 most influential features
- **Interactive visualizations**: Plotly-based dynamic charts
- **Sensor correlation analysis**: cross-sensor relationship heatmap

### 🎛️ Production Features
- **Modular architecture**: clean separation of concerns across modules
- **Logging**: structured logging with configurable levels
- **Error handling**: graceful degradation with informative messages
- **Type hints**: full static type checking support
- **Configuration management**: centralized dataclass-based configuration

---

## 🧱 System Architecture

```mermaid
flowchart TD
    subgraph DP["Data Pipeline"]
        A["Synthetic Generator"] --> B["Validation and Cleaning"]
    end
    B --> C["Feature Engineering<br/>RUL · Rolling · Lags · Cumulative"]
    C --> D["ML Pipeline<br/>RobustScaler → XGBoost → SHAP"]
    D --> E["Streamlit Dashboard<br/>RUL Monitor · SHAP Charts · Alerts"]
```

### 📁 Project Structure

```text
predictive_maintenance/
├── config.py                  # Centralized configuration dataclasses
├── data_generator.py          # Synthetic industrial sensor data generation
├── data_preprocessing.py      # Data validation, cleaning & group-safe splitting
├── feature_engineering.py     # Rolling, lag, cumulative & interaction features
├── train.py                   # XGBoost training pipeline with SHAP explanations
├── app.py                     # Streamlit interactive dashboard
├── main.py                    # CLI entry point (generate | train | all)
├── requirements.txt           # Python dependencies
├── models/                    # Saved model artefacts
│   └── xgboost_rul_model.pkl  # Serialized scaler + model + feature names
├── data/                      # Data storage
│   └── synthetic_data.csv     # Generated sensor dataset
└── logs/                      # Application logs
    └── predictive_maintenance.log
```

---

## 🔬 Technical Deep Dive

### 1. Synthetic Data Generation

`IndustrialDataGenerator` creates realistic sensor data with engineered failure patterns:

```python
# Exponential degradation pattern before failure
degradation_multiplier = np.exp(np.linspace(0, 2, n_degradation)) - 1

temperature += degradation_multiplier * 15   # +15°C max increase
vibration   += degradation_multiplier * 2    # +2 mm/s max increase
pressure    -= degradation_multiplier * 10   # -10 PSI max decrease
```

**Failure characteristics:**
- 70% of machines experience failure within the observation window
- 30-cycle degradation window with exponential progression
- Realistic noise and variation across all sensor channels

### 2. Feature Engineering Pipeline

| Feature Category | Count | Description |
|-----------------|-------|-------------|
| **Raw sensors** | 4 | temperature, vibration, pressure, rotation_speed |
| **Rolling windows** | 32 | mean, std, min, max over 5-cycle and 20-cycle windows |
| **Lag features** | 8 | t-1 and t-2 for each sensor |
| **Cumulative** | 9 | EWM mean and rate of change per sensor, plus cumulative hours |
| **Interactions** | 4 | temp/vib ratio, temp×pressure, vib/rpm, pressure/rpm |
| **Total** | **57** | Comprehensive feature set |

### 3. Model Architecture

The model is saved as a single artefact dict `{scaler, model, feature_names}` rather than a sklearn Pipeline. This is required because XGBoost's early stopping needs a pre-scaled `eval_set`, which is impossible when the scaler lives inside a Pipeline (the scaler is not fitted at that point).

```python
# Step 1: Fit scaler independently
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# Step 2: Train XGBoost with a correctly scaled eval_set
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    early_stopping_rounds=10,
    eval_metric="rmse",
)
model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)])

# Step 3: Save all three components together
joblib.dump({"scaler": scaler, "model": model, "feature_names": feature_names}, path)
```

### 4. Data Splitting Strategy

**Group-aware splitting** prevents data leakage, so machines are never split across train, validation, and test sets:
- **70% of machines** for training
- **10% of machines** for validation (early stopping)
- **20% of machines** for final test evaluation
- Validated with assertion checks after each split

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup with a Virtual Environment

```bash
# Clone the repository
git clone https://github.com/528664-007/Predictive-Maintenance-System-.git
cd Predictive-Maintenance-System-

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Generate data, then train
python main.py --mode generate
python main.py --mode train
```

Or run both steps in one command:

```bash
python main.py --mode all
```

---

## 🚀 Quick Start

### Step 1: Generate Data & Train

```bash
python main.py --mode all
```

**Expected output:**

```text
INFO - ===== GENERATING DATA
INFO - Generated dataset with 3000 records
INFO - Failure rate: 27.00%

INFO - ===== TRAINING MODEL
INFO - Feature engineering done — 57 features
INFO - Train:      2100 samples, 7 machines
INFO - Validation: 300 samples,  1 machine
INFO - Test:       600 samples,  2 machines
INFO - Model artefact saved to models/xgboost_rul_model.pkl

INFO - Pipeline complete. Launch the dashboard with:
INFO -   streamlit run app.py
```

### Step 2: Launch the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Dashboard Features

### 📈 RUL Monitoring
- **Real-time predictions** as you scrub through cycles
- **Actual vs predicted RUL** overlay chart
- **Threshold alerts**: Warning (RUL < 50) and Critical (RUL < 15)
- **Full lifecycle** degradation visualization

### 📉 Sensor Analytics
- **Multi-sensor plot**: all 4 sensors on a single interactive chart
- **Current cycle indicator**: vertical reference line
- **Correlation matrix**: cross-sensor relationship heatmap
- **Statistical metrics**: live averages and deltas per sensor

### 🔍 Explainability
- **SHAP force plot**: visual explanation for each individual prediction
- **Feature importance bar chart**: top 15 most influential features
- **Directional impact**: green (positive RUL effect) and red (negative)
- **Key findings**: plain-language summary of top drivers

### 🎨 Health Status Indicators

| Status | RUL Range | Color | Action |
|--------|-----------|-------|--------|
| **HEALTHY** | > 50 cycles | 🟢 Green | Routine monitoring |
| **WARNING** | 15–50 cycles | 🟠 Orange | Schedule maintenance |
| **CRITICAL** | < 15 cycles | 🔴 Red | Immediate action required |

---

## 📝 Configuration

All parameters are centralized in `config.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    n_machines: int = 10          # Machines to simulate
    n_cycles: int = 300           # Operational cycles per machine
    failure_window: int = 30      # Degradation window before failure
    random_seed: int = 42         # Reproducibility seed


@dataclass
class FeatureConfig:
    rolling_windows: List[int] = field(default_factory=lambda: [5, 20])
    lag_features: List[int] = field(default_factory=lambda: [1, 2])
    sensor_columns: List[str] = field(
        default_factory=lambda: ["temperature", "vibration", "pressure", "rotation_speed"]
    )


@dataclass
class ModelConfig:
    test_size: float = 0.2         # Test split ratio
    validation_size: float = 0.1   # Validation split ratio
    model_path: Path = Path("models/xgboost_rul_model.pkl")
    xgboost_params: Dict = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": 7,
            "learning_rate": 0.05,
            "early_stopping_rounds": 10,
            "eval_metric": "rmse",
            # ... full params in config.py
        }
    )


@dataclass
class DashboardConfig:
    rul_warning_threshold: int = 50    # Warning alert level
    rul_critical_threshold: int = 15   # Critical alert level
```

> Python dataclasses do not allow a plain list or dict as a default value, so those fields use `field(default_factory=...)`.

---

## 🔧 API Reference

### Data Generator

```python
from pathlib import Path
from data_generator import IndustrialDataGenerator

gen = IndustrialDataGenerator(
    n_machines=10,
    n_cycles=300,
    failure_window=30,
    random_seed=42,
)
data = gen.generate_dataset()       # returns pd.DataFrame
gen.save_dataset(data, Path("data/synthetic_data.csv"))
```

### Feature Engineering

```python
from feature_engineering import FeatureEngineer
from config import FeatureConfig

engineer = FeatureEngineer(FeatureConfig())
featured_data = engineer.engineer_features(raw_data)  # adds 57 features + RUL column
```

### Model Training

```python
from train import ModelTrainer
from config import ModelConfig

trainer = ModelTrainer(ModelConfig())
results = trainer.train(X_train, y_train, X_val, y_val)

# SHAP explanation for a single instance
explanation = trainer.explain_prediction(X_instance)

# Save / load artefact (scaler + model + feature_names)
trainer.save_model()
trainer.load_model()
```

### CLI (`main.py`)

```bash
python main.py --mode generate               # generate data only
python main.py --mode train                  # train only (data must exist)
python main.py --mode all                    # generate + train
python main.py --mode all --log-level DEBUG  # verbose output
```

---

## 🤝 Contributing

Contributions are welcome. Here's how:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m "Add amazing feature"`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Guidelines
- Follow PEP 8 style
- Add type hints to all functions
- Include docstrings for public APIs
- Update this README if behavior changes

---

## 📍 Roadmap

### v1.1
- [ ] Docker containerization
- [ ] REST API with FastAPI
- [ ] PostgreSQL integration for persistent storage
- [ ] CI/CD pipeline (GitHub Actions)

### v1.2
- [ ] Real-time data streaming with Kafka
- [ ] Automated retraining pipeline
- [ ] Email / Slack alert notifications

### v2.0
- [ ] Deep learning models (LSTM / Transformer)
- [ ] Transfer learning for new equipment types
- [ ] Edge deployment optimization

---

## 📚 References

- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754): Chen & Guestrin, 2016
- [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874): Lundberg & Lee, 2017
- [Predictive Maintenance in Industry 4.0](https://doi.org/10.1016/j.procir.2020.04.102): Procedia CIRP, 2020

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📸 Output

<div align="center">
  <img width="49%" alt="Dashboard output 1" src="https://github.com/user-attachments/assets/92513da5-0ffb-4715-a039-5e1ac72c5669" />
  <img width="49%" alt="Dashboard output 2" src="https://github.com/user-attachments/assets/6e14fa2e-836a-4f38-8f0d-579895ba8f09" />
  <img width="49%" alt="Dashboard output 3" src="https://github.com/user-attachments/assets/3522b00b-9e72-4040-a5e0-5a127dd59627" />
  <img width="49%" alt="Dashboard output 4" src="https://github.com/user-attachments/assets/a8ab64dc-d1c8-44ad-b461-cee577b00047" />
  <img width="49%" alt="Dashboard output 5" src="https://github.com/user-attachments/assets/58e60988-8338-488d-9451-2d2ec34dd9bf" />
</div>

---

## 📞 Contact

**Sameer Yaseen A S**

[![GitHub](https://img.shields.io/badge/GitHub-528664--007-black?style=flat&logo=github)](https://github.com/528664-007)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green?style=flat)](https://sameeryaseen-a-s-portfolio.netlify.app)

**Project Link**: [https://github.com/528664-007/Predictive-Maintenance-System-](https://github.com/528664-007/Predictive-Maintenance-System-)

---

<div align="center">

⭐ **Star this repo if you found it useful!**

*Built with ❤️ by [Sameer Yaseen A S](https://github.com/528664-007)*

</div>
