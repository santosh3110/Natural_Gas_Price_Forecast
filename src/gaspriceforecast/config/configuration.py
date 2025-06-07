from gaspriceforecast.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, EIA_API_KEY
from gaspriceforecast.utils.common import read_yaml, create_directories
from gaspriceforecast.entity.config_entity import (
    DataIngestionConfig, PrepareDataConfig,
    ProphetBaselineConfig, LSTMConfig, BiLSTMConfig,
    LoggingConfig, ModelEvaluationConfig, FutureFeatureEngineeringConfig,
    ForecastWithLSTMConfig, ForecastWithBiLSTMConfig
)
from pathlib import Path


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            ticker=config.yahoo_finance.ticker,
            start_date=config.yahoo_finance.start_date,
            end_date=config.yahoo_finance.end_date,
            yahoo_data_file=Path(config.yahoo_finance.data_file),
            eia_api_key=EIA_API_KEY,
            eia_start_date=config.eia_api.start_date,
            eia_end_date=config.eia_api.end_date,
            eia_inventory_file=Path(config.eia_api.data_file),
            hdd_base_url=config.hdd_data.base_url,
            hdd_raw_file=Path(config.hdd_data.local_raw_path),
            hdd_processed_file=Path(config.hdd_data.processed_file),
            hdd_year_start=config.hdd_data.year_start
        )
    
    def get_prepare_data_config(self) -> PrepareDataConfig:
        config = self.config.prepare_data
        data_config = self.config.data_ingestion

        return PrepareDataConfig(
            root_dir=Path(config.root_dir),
            yahoo_data_path=Path(data_config.yahoo_finance.data_file),
            hdd_path=Path(data_config.hdd_data.processed_file),
            inventory_path=Path(data_config.eia_api.data_file),
            processed_data_path=Path(config.processed_data_path)
        )

    def get_prophet_baseline_config(self) -> ProphetBaselineConfig:
        config = self.config.prophet_baseline
        params = self.params.prophet_baseline

        create_directories([config.root_dir])

        return ProphetBaselineConfig(
            root_dir=Path(config.root_dir),
            model_file=Path(config.model_file),
            forecast_file=Path(config.forecast_file),
            forecast_plot=Path(config.forecast_plot),
            residual_plot=Path(config.residual_plot),
            component_plot=Path(config.component_plot),
            metrics_file=Path(config.metrics_file),
            horizon=params.horizon,
            seasonality_mode=params.seasonality_mode,
            daily_seasonality=params.daily_seasonality,
            weekly_seasonality=params.weekly_seasonality,
            yearly_seasonality=params.yearly_seasonality,
            monthly_seasonality=params.monthly_seasonality,
            changepoint_prior_scale=params.changepoint_prior_scale,
            changepoint_range=params.changepoint_range,
            fourier_order=params.fourier_order,
            future_days=params.future_days
        )


    def get_lstm_model_config(self) -> LSTMConfig:
        config = self.config.lstm_model
        params = self.params.lstm

        create_directories([config.root_dir])

        return LSTMConfig(
            root_dir=config.root_dir,
            processed_data_path=config.processed_data_path,
            model_path=config.model_path,
            history_plot=config.history_plot,
            prediction_plot=config.prediction_plot,
            metrics_file=config.metrics_file,
            scaler_path=config.scaler_path,
            params=params
        )
    
    
    def get_bilstm_model_config(self) -> BiLSTMConfig:
        config = self.config.bilstm_model
        params = self.params.BiLSTM_model

        # Create directory from the path object directly
        create_directories([config.root_dir])

        return BiLSTMConfig(
            root_dir=config.root_dir,
            processed_data_path=config.processed_data_path,
            model_path=config.model_path,
            history_plot=config.history_plot,
            prediction_plot=config.prediction_plot,
            metrics_file=config.metrics_file,
            scaler_path=config.scaler_path,
            params=params
        )

  
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        eval_cfg = self.config.model_evaluation

        return ModelEvaluationConfig(
            root_dir=Path(eval_cfg.root_dir),
            lstm_metrics_path=Path(eval_cfg.lstm_metrics_path),
            bilstm_model_metrics_path=(eval_cfg.bilstm_model_metrics_path),
            prophet_metrics_path=Path(eval_cfg.prophet_metrics_path),
            evaluation_report=Path(eval_cfg.evaluation_report),
            evaluation_plot=Path(eval_cfg.evaluation_plot)
        )
    
    def get_future_feature_engineering_config(self) -> FutureFeatureEngineeringConfig:
        config = self.config.future_feature_engineering
        params = self.params.future_feature_engineering

        create_directories([config.root_dir])

        return FutureFeatureEngineeringConfig(
            root_dir=config.root_dir,
            processed_data_path=config.processed_data_path,
            last_known_data_path=config.last_known_data_path,
            output_path=config.output_path,
            garch_plot_path = config.garch_plot_path,
            params=params
        )
    
    def get_forecast_with_lstm_config(self) -> ForecastWithLSTMConfig:
        config = self.config.forecast_with_lstm
        params = self.params.forecast_lstm

        feature_order = [
            "Close",
            "Technical_Strength",
            "Technical_Strength_Signal",
            "Inventory_Bcf",
            "Hist_Vol",
            "Inventory_Bcf_lag3",
            "Hdd_ma30",
            "Inventory_ma30",
            "Hdd_cumsum",
            "Inventory_cumsum",
            "Volume_ma30",
            "Volume_cumsum"
        ]

        feature_dtypes = {
            "Close": "float32",
            "Technical_Strength": "float32",
            "Technical_Strength_Signal": "float32",
            "Inventory_Bcf": "float32",
            "Hist_Vol": "float32",
            "Inventory_Bcf_lag3": "float32",
            "Hdd_ma30": "float32",
            "Inventory_ma30": "float32",
            "Hdd_cumsum": "float64",
            "Inventory_cumsum": "float64",
            "Volume_ma30": "float32",
            "Volume_cumsum": "float64"
        }

        return ForecastWithLSTMConfig(
            model_path=Path(config.model_path),
            feature_scaler_path=Path(config.feature_scaler_path),
            target_scaler_path=Path(config.target_scaler_path),
            future_feature_path=Path(config.future_feature_path),
            last_known_data_path=Path(config.last_known_data_path),
            processed_data_path=Path(config.processed_data_path),
            forecast_plot_path=Path(config.forecast_plot_path),
            output_forecast_path=Path(config.output_forecast_path),
            time_step=params.time_step,
            feature_order=feature_order,
            feature_dtypes=feature_dtypes
        )
    def get_forecast_with_bilstm_config(self) -> ForecastWithBiLSTMConfig:
        config = self.config.forecast_with_bilstm
        params = self.params.forecast_bilstm

        feature_order = [
            "Close",
            "Technical_Strength",
            "Technical_Strength_Signal",
            "Inventory_Bcf",
            "Hist_Vol",
            "Inventory_Bcf_lag3",
            "Hdd_ma30",
            "Inventory_ma30",
            "Hdd_cumsum",
            "Inventory_cumsum",
            "Volume_ma30",
            "Volume_cumsum"
        ]

        feature_dtypes = {
            "Close": "float32",
            "Technical_Strength": "float32",
            "Technical_Strength_Signal": "float32",
            "Inventory_Bcf": "float32",
            "Hist_Vol": "float32",
            "Inventory_Bcf_lag3": "float32",
            "Hdd_ma30": "float32",
            "Inventory_ma30": "float32",
            "Hdd_cumsum": "float64",
            "Inventory_cumsum": "float64",
            "Volume_ma30": "float32",
            "Volume_cumsum": "float64"
        }

        return ForecastWithBiLSTMConfig(
            model_path=Path(config.model_path),
            feature_scaler_path=Path(config.feature_scaler_path),
            target_scaler_path=Path(config.target_scaler_path),
            future_feature_path=Path(config.future_feature_path),
            last_known_data_path=Path(config.last_known_data_path),
            processed_data_path=Path(config.processed_data_path),
            forecast_plot_path=Path(config.forecast_plot_path),
            output_forecast_path=Path(config.output_forecast_path),
            time_step=params.time_step,
            feature_order=feature_order,
            feature_dtypes=feature_dtypes
        )
    
    def get_logging_config(self) -> LoggingConfig:
        config = self.config.logging
        return LoggingConfig(
            log_file=Path(config.log_file),
            level=config.level
        )