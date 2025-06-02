import json
import mlflow
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from gaspriceforecast.entity.config_entity import ModelEvaluationConfig
from gaspriceforecast.utils.common import create_directories
from gaspriceforecast.utils.logger import get_logger

logger = get_logger("mlflow_model_evaluation.log")


class ModelEvaluatorWithMLflow:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def log_metrics_and_artifacts(self):
        logger.info("Loading metric files...")
        with open(self.config.lstm_metrics_path) as f:
            lstm_metrics = json.load(f)
        with open(self.config.bilstm_model_metrics_path) as f:
            bilstm_metrics = json.load(f)
        with open(self.config.prophet_metrics_path) as f:
            prophet_metrics = json.load(f)

        comparison = {
            "RMSE": {
                "LSTM": lstm_metrics["RMSE"],
                "BiLSTM": bilstm_metrics["RMSE"],
                "Prophet": prophet_metrics["RMSE"]
            },
            "MAE": {
                "LSTM": lstm_metrics["MAE"],
                "BiLSTM": bilstm_metrics["MAE"],
                "Prophet": prophet_metrics["MAE"]
            },
            "MAPE": {
                "LSTM": lstm_metrics["MAPE"],
                "BiLSTM": bilstm_metrics["MAPE"],
                "Prophet": prophet_metrics["MAPE"]
            }
        }

        create_directories([self.config.root_dir])
        with open(self.config.evaluation_report, 'w') as f:
            json.dump(comparison, f, indent=4)
        logger.info(f"Comparison report saved to: {self.config.evaluation_report}")

        # Plot and save
        self._plot_metrics(comparison)

        # MLflow logging
        mlflow.set_tracking_uri("mlruns")
        with mlflow.start_run(run_name="Model Evaluation"):
            mlflow.log_metrics({
                "LSTM_RMSE": comparison["RMSE"]["LSTM"],
                "LSTM_MAE": comparison["MAE"]["LSTM"],
                "LSTM_MAPE": comparison["MAPE"]["LSTM"],
                "BiLSTM_RMSE": comparison["RMSE"]["BiLSTM"],
                "BiLSTM_MAE": comparison["MAE"]["BiLSTM"],
                "BiLSTM_MAPE": comparison["MAPE"]["BiLSTM"],
                "Prophet_RMSE": comparison["RMSE"]["Prophet"],
                "Prophet_MAE": comparison["MAE"]["Prophet"],
                "Prophet_MAPE": comparison["MAPE"]["Prophet"],
            })

            mlflow.log_artifact(str(self.config.evaluation_report))
            mlflow.log_artifact(str(self.config.evaluation_plot))

            # Log LSTM model
            lstm_model_path = "artifacts/lstm_model/lstm_model.h5"
            if Path(lstm_model_path).exists():
                mlflow.keras.log_model(tf.keras.models.load_model(lstm_model_path), "LSTM_Model")
                logger.info("Logged LSTM model to MLflow")

            # Log BiLSTM model
            bilstm_model_path = "artifacts/bilstm_model/bilstm_model.h5"
            if Path(bilstm_model_path).exists():
                mlflow.keras.log_model(tf.keras.models.load_model(bilstm_model_path), "BiLSTM_Model")
                logger.info("Logged BiLSTM model to MLflow")

    def _plot_metrics(self, comparison):
        metrics = ["RMSE", "MAE", "MAPE"]
        lstm_vals = [comparison[m]["LSTM"] for m in metrics]
        bilstm_vals = [comparison[m]["BiLSTM"] for m in metrics]
        prophet_vals = [comparison[m]["Prophet"] for m in metrics]

        x = range(len(metrics))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar([i - width for i in x], lstm_vals, width=width, label="LSTM", color="indianred")
        plt.bar(x, bilstm_vals, width=width, label="BiLSTM", color="seagreen")
        plt.bar([i + width for i in x], prophet_vals, width=width, label="Prophet", color="steelblue")

        plt.xticks(ticks=x, labels=metrics)
        plt.ylabel("Score")
        plt.title("Model Comparison: LSTM vs BiLSTM vs Prophet")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.evaluation_plot)
        plt.close()
        logger.info(f"Saved evaluation plot to: {self.config.evaluation_plot}")
