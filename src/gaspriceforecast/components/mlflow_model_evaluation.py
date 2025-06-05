# import json
# import mlflow
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from pathlib import Path
# import dagshub
# from gaspriceforecast.entity.config_entity import ModelEvaluationConfig
# from gaspriceforecast.utils.common import create_directories
# from gaspriceforecast.utils.logger import get_logger

# logger = get_logger("mlflow_model_evaluation.log")


# class ModelEvaluatorWithMLflow:
#     def __init__(self, config: ModelEvaluationConfig):
#         self.config = config

#     def log_metrics_and_artifacts(self):
#         logger.info("Loading metric files...")
#         with open(self.config.lstm_metrics_path) as f:
#             lstm_metrics = json.load(f)
#         with open(self.config.bilstm_model_metrics_path) as f:
#             bilstm_metrics = json.load(f)
#         with open(self.config.prophet_metrics_path) as f:
#             prophet_metrics = json.load(f)

#         comparison = {
#             "RMSE": {
#                 "LSTM": lstm_metrics["RMSE"],
#                 "BiLSTM": bilstm_metrics["RMSE"],
#                 "Prophet": prophet_metrics["RMSE"]
#             },
#             "MAE": {
#                 "LSTM": lstm_metrics["MAE"],
#                 "BiLSTM": bilstm_metrics["MAE"],
#                 "Prophet": prophet_metrics["MAE"]
#             },
#             "MAPE": {
#                 "LSTM": lstm_metrics["MAPE"],
#                 "BiLSTM": bilstm_metrics["MAPE"],
#                 "Prophet": prophet_metrics["MAPE"]
#             }
#         }

#         create_directories([self.config.root_dir])
#         with open(self.config.evaluation_report, 'w') as f:
#             json.dump(comparison, f, indent=4)
#         logger.info(f"Comparison report saved to: {self.config.evaluation_report}")

#         self._plot_metrics(comparison)

#         # Set Dagshub MLflow tracking
#         dagshub.init(
#             repo_owner="santoshkumarguntupalli",
#             repo_name="Natural_Gas_Price_Forecast",
#             mlflow=True
#         )

#         with mlflow.start_run(run_name="Model Evaluation"):
#             # Log Metrics
#             mlflow.log_metrics({
#                 "LSTM_RMSE": comparison["RMSE"]["LSTM"],
#                 "LSTM_MAE": comparison["MAE"]["LSTM"],
#                 "LSTM_MAPE": comparison["MAPE"]["LSTM"],
#                 "BiLSTM_RMSE": comparison["RMSE"]["BiLSTM"],
#                 "BiLSTM_MAE": comparison["MAE"]["BiLSTM"],
#                 "BiLSTM_MAPE": comparison["MAPE"]["BiLSTM"],
#                 "Prophet_RMSE": comparison["RMSE"]["Prophet"],
#                 "Prophet_MAE": comparison["MAE"]["Prophet"],
#                 "Prophet_MAPE": comparison["MAPE"]["Prophet"],
#             })

#             # Log Artifacts
#             mlflow.log_artifact(str(self.config.evaluation_report))
#             mlflow.log_artifact(str(self.config.evaluation_plot))

#             # Log LSTM model and params
#             lstm_model_path = "artifacts/lstm_model/lstm_model.h5"
#             lstm_params_path = "artifacts/lstm_model/metrics.json"
#             if Path(lstm_model_path).exists():
#                 mlflow.keras.log_model(tf.keras.models.load_model(lstm_model_path), "LSTM_Model")
#                 with open(lstm_params_path) as f:
#                     lstm_params = json.load(f)
#                     for k, v in lstm_params.items():
#                         mlflow.log_param(f"LSTM_{k}", v)
#                 logger.info("Logged LSTM model and params to MLflow")

#             # Log BiLSTM model and params
#             bilstm_model_path = "artifacts/bilstm_model/bilstm_model.h5"
#             bilstm_params_path = "artifacts/bilstm_model/metrics.json"
#             if Path(bilstm_model_path).exists():
#                 mlflow.keras.log_model(tf.keras.models.load_model(bilstm_model_path), "BiLSTM_Model")
#                 with open(bilstm_params_path) as f:
#                     bilstm_params = json.load(f)
#                     for k, v in bilstm_params.items():
#                         mlflow.log_param(f"BiLSTM_{k}", v)
#                 logger.info("Logged BiLSTM model and params to MLflow")

#     def _plot_metrics(self, comparison):
#         metrics = ["RMSE", "MAE", "MAPE"]
#         lstm_vals = [comparison[m]["LSTM"] for m in metrics]
#         bilstm_vals = [comparison[m]["BiLSTM"] for m in metrics]
#         prophet_vals = [comparison[m]["Prophet"] for m in metrics]

#         x = range(len(metrics))
#         width = 0.25

#         plt.figure(figsize=(10, 6))
#         plt.bar([i - width for i in x], lstm_vals, width=width, label="LSTM", color="indianred")
#         plt.bar(x, bilstm_vals, width=width, label="BiLSTM", color="seagreen")
#         plt.bar([i + width for i in x], prophet_vals, width=width, label="Prophet", color="steelblue")

#         plt.xticks(ticks=x, labels=metrics)
#         plt.ylabel("Score")
#         plt.title("Model Comparison: LSTM vs BiLSTM vs Prophet")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(self.config.evaluation_plot)
#         plt.close()
#         logger.info(f"Saved evaluation plot to: {self.config.evaluation_plot}")

# mlflow_model_evaluation.py

import json
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import dagshub

from gaspriceforecast.entity.config_entity import ModelEvaluationConfig
from gaspriceforecast.utils.common import create_directories
from gaspriceforecast.utils.logger import get_logger

logger = get_logger("mlflow_model_evaluation.log")


class ModelEvaluatorWithMLflow:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _load_metrics(self):
        logger.info("Loading metric files...")
        with open(self.config.lstm_metrics_path) as f:
            lstm = json.load(f)
        with open(self.config.bilstm_model_metrics_path) as f:
            bilstm = json.load(f)
        with open(self.config.prophet_metrics_path) as f:
            prophet = json.load(f)
        return lstm, bilstm, prophet

    def _plot_comparison(self, comparison):
        logger.info("Generating comparison plot...")
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

    def log_metrics_and_artifacts(self):
        # Load metrics
        lstm, bilstm, prophet = self._load_metrics()

        # Construct comparison dict
        comparison = {
            "RMSE": {"LSTM": lstm["RMSE"], "BiLSTM": bilstm["RMSE"], "Prophet": prophet["RMSE"]},
            "MAE":  {"LSTM": lstm["MAE"], "BiLSTM": bilstm["MAE"], "Prophet": prophet["MAE"]},
            "MAPE": {"LSTM": lstm["MAPE"], "BiLSTM": bilstm["MAPE"], "Prophet": prophet["MAPE"]}
        }

        # Ensure directory exists before writing JSON
        Path(self.config.evaluation_report).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.evaluation_report, 'w') as f:
            json.dump(comparison, f, indent=4)
        logger.info(f"Comparison report saved to: {self.config.evaluation_report}")

        # Generate comparison plot
        self._plot_comparison(comparison)

        # Dagshub MLflow connection
        logger.info("Initializing Dagshub tracking...")
        dagshub.init(
            repo_owner="santoshkumarguntupalli",
            repo_name="Natural_Gas_Price_Forecast",
            mlflow=True
        )

        # Main MLflow run
        with mlflow.start_run(run_name="Model Comparison"):
            logger.info("Logging LSTM model, metrics and params...")
            self._log_model_run(
                name="LSTM_Model_Evaluation",
                metrics=lstm,
                model_path="artifacts/lstm_model/lstm_model.h5",
                metrics_file=self.config.lstm_metrics_path
            )

            logger.info("Logging BiLSTM model, metrics and params...")
            self._log_model_run(
                name="BiLSTM_Model_Evaluation",
                metrics=bilstm,
                model_path="artifacts/bilstm_model/bilstm_model.h5",
                metrics_file=self.config.bilstm_model_metrics_path
            )

            logger.info("Logging Prophet model, metrics and params...")
            self._log_model_run(
                name="Prophet_Model_Evaluation",
                metrics=prophet,
                model_path="artifacts/prophet_model/prophet_model.pkl",
                metrics_file=self.config.prophet_metrics_path
            )

            # Log comparison artifacts
            logger.info("Logging comparison report and plot to MLflow...")
            mlflow.log_artifact(self.config.evaluation_report)
            mlflow.log_artifact(self.config.evaluation_plot)

    def _log_model_run(self, name, metrics, model_path, metrics_file):
        with mlflow.start_run(run_name=name, nested=True):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(metrics_file)

            if Path(model_path).exists():
                if name == "Prophet_Model":
                    mlflow.log_artifact(model_path)
                    logger.info(f"Logged Prophet model file to MLflow: {model_path}")
                else:
                    model = tf.keras.models.load_model(model_path)
                    mlflow.keras.log_model(model, artifact_path="model")
                    logger.info(f"Logged {name} model to MLflow: {model_path}")
