import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import json

from gaspriceforecast.entity.config_entity import ProphetBaselineConfig
from gaspriceforecast.utils.common import create_directories
from gaspriceforecast.utils.logger import get_logger

logger = get_logger("prophet_baseline.log")


class ProphetBaseline:
    def __init__(self, config: ProphetBaselineConfig):
        self.config = config

    def run(self):
        df = pd.read_csv("artifacts/prepare_data/processed_data.csv", parse_dates=["Date"])
        prophet_df = df.rename(columns={"Date": "ds", "Close": "y"})[["ds", "y"]]
        prophet_df["y"] = prophet_df["y"].astype(float)

        logger.info("Data loaded and formatted for Prophet.")

        test_size = self.config.horizon
        train_df = prophet_df[:-test_size]
        test_df = prophet_df[-test_size:]

        logger.info("Initializing Prophet with hyperparameters from params.yaml")
        model = Prophet(
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            changepoint_range=self.config.changepoint_range,
            seasonality_mode=self.config.seasonality_mode,
            daily_seasonality=self.config.daily_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            yearly_seasonality=self.config.yearly_seasonality,
        )

        if self.config.monthly_seasonality:
            model.add_seasonality(
                name="monthly",
                period=30.5,
                fourier_order=self.config.fourier_order
            )

        logger.info("Fitting Prophet model...")
        model.fit(train_df)

        joblib.dump(model, self.config.model_file)
        logger.info("Model saved.")

        logger.info("Generating forecast...")
        future = model.make_future_dataframe(periods=self.config.horizon)
        forecast = model.predict(future)

        forecast_df = forecast[["ds", "yhat"]].merge(test_df, on="ds", how="inner")
        forecast_df.to_csv(self.config.forecast_file, index=False)
        logger.info(f"Forecast CSV saved to {self.config.forecast_file}")

        # Metrics
        rmse = np.sqrt(mean_squared_error(forecast_df["y"], forecast_df["yhat"]))
        mae = mean_absolute_error(forecast_df["y"], forecast_df["yhat"])
        mape = mean_absolute_percentage_error(forecast_df["y"], forecast_df["yhat"])
        logger.info(f"Prophet Test RMSE: {rmse:.4f}")
        logger.info(f"Prophet Test MAE : {mae:.4f}")
        logger.info(f"Prophet Test MAPE: {mape:.4f}")

        metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
        with open(self.config.metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to {self.config.metrics_file}")

        # Forecast vs Actual Plot
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df["ds"], forecast_df["y"], label="Actual")
        plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast", linestyle="--")
        plt.title("Prophet Forecast vs Actual (Test Period)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.config.forecast_plot)
        logger.info(f"Forecast plot saved to {self.config.forecast_plot}")

        # Residual Plot
        residuals = forecast_df["y"] - forecast_df["yhat"]
        plt.figure(figsize=(10, 4))
        plt.plot(forecast_df["ds"], residuals)
        plt.title("Residuals (Actual - Predicted)")
        plt.xlabel("Date")
        plt.ylabel("Residual")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.config.residual_plot)
        logger.info(f"Residuals plot saved to {self.config.residual_plot}")

        # Component Plot
        fig2 = model.plot_components(forecast)
        fig2.savefig(self.config.component_plot)
        logger.info(f"Prophet components plot saved to {self.config.component_plot}")

