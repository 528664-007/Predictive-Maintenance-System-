"""
Streamlit dashboard for Predictive Maintenance System

FIX SUMMARY vs original:
  1. Loads the new artefact dict (scaler + model + feature_names) instead
     of a raw pipeline object.
  2. predict_rul / SHAP helpers use self.scaler + self.model directly.
  3. feature_names sourced from artefact dict, not from pipeline attr.
  4. argparse removed — conflicts with streamlit's own CLI argument parser.
     Paths are configured via st.secrets or sensible defaults.
  5. SHAP force_plot: result is a real Figure object (verified); kept as-is.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
import shap
import logging
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from config import DashboardConfig, setup_logging

matplotlib.use("Agg")
logger = setup_logging()

MODEL_PATH = Path("models/xgboost_rul_model.pkl")
DATA_PATH  = Path("data/synthetic_data.csv")


class PredictiveMaintenanceDashboard:

    def __init__(self, model_path: Path, data_path: Path, config: DashboardConfig):
        self.config     = config
        self.model_path = model_path
        self.data_path  = data_path
        # FIX: separate artefacts instead of one pipeline object
        self.scaler        = None
        self.model         = None
        self.feature_names = None
        self.data          = None

    # ------------------------------------------------------------------ #
    def load_resources(self) -> None:
        artefact = joblib.load(self.model_path)
        self.scaler        = artefact["scaler"]
        self.model         = artefact["model"]
        # FIX: feature_names from the saved dict, not pipeline.feature_names_in_
        self.feature_names = artefact["feature_names"]

        self.data = pd.read_csv(self.data_path)
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        logger.info("Resources loaded")

    # ------------------------------------------------------------------ #
    def get_machine_data(self, machine_id: int) -> pd.DataFrame:
        return self.data[self.data["machine_id"] == machine_id].sort_values("cycle")

    def predict_rul(self, instance: pd.DataFrame) -> float:
        X = instance[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])

    def get_machine_status(self, rul: float) -> Tuple[str, str]:
        if rul < self.config.rul_critical_threshold:
            return "CRITICAL ALERT", "red"
        elif rul < self.config.rul_warning_threshold:
            return "WARNING", "orange"
        return "HEALTHY", "green"

    # ------------------------------------------------------------------ #
    def create_rul_plot(self, machine_data: pd.DataFrame,
                        predictions: Optional[np.ndarray] = None) -> go.Figure:
        fig = go.Figure()
        if "rul" in machine_data.columns:
            fig.add_trace(go.Scatter(
                x=machine_data["cycle"], y=machine_data["rul"],
                mode="lines+markers", name="Actual RUL",
                line=dict(color="blue", width=2)))
        if predictions is not None:
            fig.add_trace(go.Scatter(
                x=machine_data["cycle"], y=predictions,
                mode="lines+markers", name="Predicted RUL",
                line=dict(color="red", width=2, dash="dash")))

        fig.add_hline(y=self.config.rul_warning_threshold,
                      line_dash="dash", line_color="orange",
                      annotation_text="Warning Threshold")
        fig.add_hline(y=self.config.rul_critical_threshold,
                      line_dash="dash", line_color="red",
                      annotation_text="Critical Threshold")
        fig.update_layout(
            title="RUL Degradation Over Time",
            xaxis_title="Cycle", yaxis_title="Remaining Useful Life",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig

    def create_sensor_plots(self, machine_data: pd.DataFrame,
                            selected_cycle: int) -> go.Figure:
        sensors = ["temperature", "vibration", "pressure", "rotation_speed"]
        fig = go.Figure()
        for sensor in sensors:
            fig.add_trace(go.Scatter(
                x=machine_data["cycle"], y=machine_data[sensor],
                mode="lines", name=sensor.replace("_", " ").title()))
        fig.add_vline(x=selected_cycle, line_dash="dash", line_color="gray",
                      annotation_text=f"Current Cycle: {selected_cycle}")
        fig.update_layout(title="Sensor Readings Over Time",
                          xaxis_title="Cycle", hovermode="x unified")
        return fig

    # ------------------------------------------------------------------ #
    def _shap_values_for_instance(self, instance: pd.DataFrame):
        X = instance[self.feature_names]
        X_scaled = self.scaler.transform(X)
        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)
        return explainer, shap_values, X, X_scaled

    def create_shap_explanation(self, instance: pd.DataFrame):
        explainer, shap_values, X, _ = self._shap_values_for_instance(instance)
        # shap.force_plot with matplotlib=True returns a Figure (verified)
        fig = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X.iloc[0],
            matplotlib=True,
            show=False,
        )
        return fig

    # ------------------------------------------------------------------ #
    def render(self) -> None:
        st.set_page_config(
            page_title=self.config.page_title,
            page_icon="🔧",
            layout="wide",
        )
        st.title("🔧 Predictive Maintenance Control Center")

        if self.model is None:
            with st.spinner("Loading resources..."):
                self.load_resources()

        # Sidebar
        st.sidebar.header("Machine Selection")
        machine_ids     = sorted(self.data["machine_id"].unique())
        selected_machine = st.sidebar.selectbox("Select Machine ID", machine_ids)

        machine_data = self.get_machine_data(selected_machine)
        max_cycle    = int(machine_data["cycle"].max())
        selected_cycle = st.sidebar.slider("Operational Cycle", 1, max_cycle, max_cycle)

        instance = machine_data[machine_data["cycle"] == selected_cycle]
        if instance.empty:
            st.error(f"No data for cycle {selected_cycle}")
            return

        # Predict
        try:
            predicted_rul = self.predict_rul(instance)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            predicted_rul = 0.0

        # KPI row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted RUL", f"{predicted_rul:.1f} cycles",
                      f"Cycle {selected_cycle}")
        with col2:
            status, color = self.get_machine_status(predicted_rul)
            st.markdown(
                f'<div style="padding:20px;background:{color};border-radius:10px;">'                f'<h2 style="color:white;text-align:center;">{status}</h2></div>',
                unsafe_allow_html=True)
        with col3:
            failure_rows = machine_data[machine_data["failure"] == 1]
            if not failure_rows.empty:
                failure_cycle = failure_rows["cycle"].min()
                st.metric("Actual Cycles to Failure",
                          str(int(failure_cycle - selected_cycle)))
            else:
                st.metric("Failure Status", "No failure recorded")

        # RUL plot
        st.header("RUL Degradation Analysis")
        all_preds = []
        for _, row in machine_data.iterrows():
            try:
                all_preds.append(self.predict_rul(pd.DataFrame([row])))
            except Exception:
                all_preds.append(np.nan)
        st.plotly_chart(self.create_rul_plot(machine_data, np.array(all_preds)),
                        use_container_width=True)

        # Sensor plots
        st.header("Sensor Readings")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(self.create_sensor_plots(machine_data, selected_cycle),
                            use_container_width=True)
        with col2:
            sensors = ["temperature", "vibration", "pressure", "rotation_speed"]
            corr_fig = px.imshow(machine_data[sensors].corr(), text_auto=True,
                                 title="Sensor Correlation Matrix",
                                 color_continuous_scale="RdBu")
            st.plotly_chart(corr_fig, use_container_width=True)

        # SHAP
        st.header("SHAP Feature Importance Analysis")
        try:
            shap_fig = self.create_shap_explanation(instance)
            st.pyplot(shap_fig)
            plt.clf()

            explainer, shap_values, X, _ = self._shap_values_for_instance(instance)
            importance_df = pd.DataFrame({
                "feature":   self.feature_names,
                "importance": np.abs(shap_values[0]),
                "direction":  ["positive" if v > 0 else "negative"
                               for v in shap_values[0]],
            }).sort_values("importance").tail(15)

            bar_fig = go.Figure(go.Bar(
                y=importance_df["feature"],
                x=importance_df["importance"],
                orientation="h",
                marker_color=["red" if d == "negative" else "green"
                              for d in importance_df["direction"]],
                text=importance_df["importance"].round(4),
                textposition="outside",
            ))
            bar_fig.update_layout(title="Top 15 Feature Importances (SHAP)",
                                  xaxis_title="|SHAP Value|", height=500)
            st.plotly_chart(bar_fig, use_container_width=True)

            st.subheader("Key Findings:")
            for _, row in importance_df.nlargest(3, "importance").iterrows():
                fname = row["feature"].replace("_", " ").title()
                direction = "increasing" if row["direction"] == "positive" else "decreasing"
                st.write(f"• **{fname}** is {direction} the predicted RUL "
                         f"(SHAP: {row['importance']:.4f})")
        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")

        # Historical stats
        st.header("Machine Historical Statistics")
        c1, c2, c3, c4 = st.columns(4)
        last = machine_data.iloc[-1]
        means = machine_data[["temperature", "vibration", "pressure"]].mean()
        with c1:
            st.metric("Avg Temperature", f"{means['temperature']:.1f}°C",
                      f"{last['temperature'] - means['temperature']:.1f}°C")
        with c2:
            st.metric("Avg Vibration", f"{means['vibration']:.2f} mm/s",
                      f"{last['vibration'] - means['vibration']:.2f}")
        with c3:
            st.metric("Avg Pressure", f"{means['pressure']:.1f} PSI",
                      f"{last['pressure'] - means['pressure']:.1f}")
        with c4:
            st.metric("Operating Hours", f"{selected_cycle * 8:,}",
                      f"Cycle {selected_cycle}")


# ------------------------------------------------------------------ #
def main():
    # FIX: argparse removed — use constants or st.secrets for path config
    config    = DashboardConfig()
    dashboard = PredictiveMaintenanceDashboard(MODEL_PATH, DATA_PATH, config)
    dashboard.render()


if __name__ == "__main__":
    main()
