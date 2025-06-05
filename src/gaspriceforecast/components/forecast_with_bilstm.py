import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from gaspriceforecast.entity.config_entity import ForecastWithBiLSTMConfig
from gaspriceforecast.utils.logger import get_logger

logger = get_logger("forecast_with_bilstm.log")


class ForecastWithBiLSTM:
    def __init__(self, config: ForecastWithBiLSTMConfig):
        self.config = config

    def load_dependencies(self):
        logger.info("Loading scalers and model...")
        self.feature_scaler = joblib.load(self.config.feature_scaler_path)
        self.target_scaler = joblib.load(self.config.target_scaler_path)
        self.model = tf.keras.models.load_model(self.config.model_path)
        self.feature_dtypes = self.config.feature_dtypes

    def prepare_data(self):
        logger.info("Loading and preparing future features...")

        df = pd.read_csv(self.config.future_feature_path, parse_dates=["Date"])

        # Save future dates before dropping
        self.future_dates = df["Date"]

        # Drop any columns not in feature_order
        df = df[self.config.feature_order]

        # Round and cast to correct dtypes
        df = df.round(2).astype(self.feature_dtypes)

        self.future_features = df
        print(df)
        return df


    def forecast(self, features_df):
        logger.info("Running BiLSTM forecast on future data...")

        last_known = pd.read_csv(self.config.last_known_data_path)
        last_known = last_known[self.config.feature_order].round(2).astype(self.feature_dtypes)

        full_input = pd.concat([last_known, features_df]).reset_index(drop=True)
        scaled_full = self.feature_scaler.transform(full_input)

        time_step = self.config.time_step
        current_input = scaled_full[:time_step].copy()

        preds_scaled = []

        for i in range(30):
            seq_input = current_input[-time_step:].reshape(1, time_step, scaled_full.shape[1])
            next_pred_scaled = self.model.predict(seq_input, verbose=0)[0][0]
            preds_scaled.append([next_pred_scaled]) 

            next_row = scaled_full[time_step + i].copy()
            next_row[0] = next_pred_scaled
            current_input = np.vstack([current_input, next_row])

        preds = self.target_scaler.inverse_transform(np.array(preds_scaled))
        return preds


    def plot_and_save(self, preds):
        logger.info("Plotting and saving forecast...")
        Path(self.config.forecast_plot_path).parent.mkdir(parents=True, exist_ok=True)

        actual = pd.read_csv(self.config.processed_data_path, parse_dates=["Date"])
        actual = actual[-600:]

        plt.figure(figsize=(14, 6))
        plt.plot(actual["Date"], actual["Close"], label="Actual Close (last 600)", color="blue")
        plt.plot(self.future_dates, preds[:,0], label="Forecasted Close (next 30)", color="orange")
        plt.title("Next 30-Day BiLSTM Forecast")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.forecast_plot_path)
        plt.close()
        logger.info(f"Saved forecast plot to: {self.config.forecast_plot_path}")

    def run(self):
        self.load_dependencies()
        future_df = self.prepare_data()
        predictions = self.forecast(future_df)
        self.plot_and_save(predictions)

        forecast_df = pd.DataFrame({
            "Date": self.future_dates,
            "Forecasted_Close": predictions.ravel()
        })

        forecast_df.to_csv(self.config.output_forecast_path, index=False)
        logger.info(f"Saved forecasted prices to: {self.config.output_forecast_path}")