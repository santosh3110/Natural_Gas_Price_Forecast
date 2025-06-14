import os
import json
import joblib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import dagshub
from gaspriceforecast.entity.config_entity import BiLSTMConfig
from gaspriceforecast.utils.logger import get_logger

# Set all seeds for reproducibility
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'

set_seed(42)

logger = get_logger("bilstm_model.log")


class BiLSTMTrainer:
    def __init__(self, config: BiLSTMConfig):
        self.config = config

    def load_data(self):
        logger.info("Loading and formatting data...")
        df = pd.read_csv(self.config.processed_data_path, parse_dates=["Date"])
        df = df.drop(columns=["Date", "Return", "Volume", "Hdd"], errors="ignore")

        df = df.round(2).astype({
            'Close': 'float32',
            'Technical_Strength': 'float32',
            'Technical_Strength_Signal': 'float32',
            'Hist_Vol': 'float32',
            'Inventory_Bcf': 'float32',
            'Inventory_Bcf_lag3': 'float32',
            'Hdd_ma30': 'float32',
            'Inventory_ma30': 'float32',
            'Hdd_cumsum': 'float32',
            'Inventory_cumsum': 'float64',
            'Volume_ma30': 'float32',
            'Volume_cumsum': 'float64'
        })

        return df

    def scale_data(self, df_train, df_test):
        logger.info("Scaling features and target...")
        X_train = df_train
        y_train = df_train[["Close"]]

        X_test = df_test
        y_test = df_test[["Close"]]

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)

        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)

        joblib.dump(feature_scaler, self.config.scaler_path.replace(".pkl", "_feature.pkl"))
        joblib.dump(target_scaler, self.config.scaler_path)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, target_scaler

    def split_sequences(self, X, y, time_step):
        X_seq, y_seq = [], []
        for i in range(time_step, len(X)):
            X_seq.append(X[i - time_step:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape, params):
        logger.info("Building BiLSTM model...")
        model = Sequential()
        for i in range(params['layers']):
            return_seq = i < params['layers'] - 1
            layer = Bidirectional(LSTM(params['units'], return_sequences=return_seq), merge_mode='concat')
            model.add(layer)
            model.add(Dropout(params['dropout']))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                      loss='mean_squared_error')
        return model

    def train(self):
        df = self.load_data()
        test_size = self.config.params.get("test_size", 0.25)
        df_train, df_test = train_test_split(df, test_size=test_size, shuffle=False, random_state=42)

        X_train, X_test, y_train, y_test, target_scaler = self.scale_data(df_train, df_test)

        time_step = self.config.params['time_step']
        X_train_seq, y_train_seq = self.split_sequences(X_train, y_train, time_step)
        X_test_seq, y_test_seq = self.split_sequences(X_test, y_test, time_step)

        model = self.build_model((X_train_seq.shape[1], X_train_seq.shape[2]), self.config.params)

        set_seed(42)
        early_stop = EarlyStopping(monitor="val_loss", patience=self.config.params["patience"], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=10, min_lr=1e-6)

        dagshub.init(
        repo_owner="santoshkumarguntupalli",
        repo_name="Natural_Gas_Price_Forecast",
        mlflow=True
        )

        logger.info("Training BiLSTM model...")
        with mlflow.start_run(run_name="BiLSTM_Model"):
            history = model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_test_seq, y_test_seq),
                epochs=self.config.params["epochs"],
                batch_size=self.config.params["batch_size"],
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )

            model.save(self.config.model_path)
            logger.info(f"Model saved at: {self.config.model_path}")
            mlflow.keras.log_model(model, "model")

            # Training curve
            plt.plot(history.history["loss"], label="Train Loss")
            plt.plot(history.history["val_loss"], label="Val Loss")
            plt.title("BiLSTM Loss Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.config.history_plot)
            plt.close()
            logger.info(f"Loss plot saved at: {self.config.history_plot}")
            mlflow.log_artifact(self.config.history_plot)

            # Forecast
            set_seed(42)
            y_pred = model.predict(X_test_seq)
            y_pred_inv = target_scaler.inverse_transform(y_pred)
            y_test_inv = target_scaler.inverse_transform(y_test_seq)

            plt.figure(figsize=(14, 6))
            plt.plot(y_test_inv, label="Actual")
            plt.plot(y_pred_inv, label="Predicted")
            plt.title("BiLSTM Forecast vs Actual")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.config.prediction_plot)
            plt.close()
            logger.info(f"Prediction plot saved at: {self.config.prediction_plot}")
            mlflow.log_artifact(self.config.prediction_plot)

            # Evaluation
            rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
            mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
            mape = float(mean_absolute_percentage_error(y_test_inv, y_pred_inv))

            metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
            with open(self.config.metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"BiLSTM Metrics: {metrics}")
            logger.info(f"Metrics saved at: {self.config.metrics_file}")
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(self.config.metrics_file)

            for k, v in self.config.params.items():
                mlflow.log_param(k, v)

    def run(self):
        self.train()
        
