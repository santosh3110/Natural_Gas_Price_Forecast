from gaspriceforecast.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, EIA_API_KEY
from gaspriceforecast.utils.common import read_yaml, create_directories
from gaspriceforecast.entity.config_entity import (
    DataIngestionConfig, PrepareDataConfig,
    ProphetBaselineConfig, LSTMModelConfig, ForecastConfig,
    LoggingConfig, ProphetHyperParams, LSTMHyperParams, GARCHHyperParams,
    HyperParametersConfig
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


    def get_lstm_model_config(self) -> LSTMModelConfig:
        config = self.config.lstm_model
        return LSTMModelConfig(
            model_path=Path(config.model_path),
            checkpoint_path=Path(config.checkpoint_path),
            tensorboard_log_dir=Path(config.tensorboard_log_dir)
        )

    def get_forecast_config(self) -> ForecastConfig:
        config = self.config.forecast
        create_directories([config.root_dir])
        return ForecastConfig(
            root_dir=Path(config.root_dir),
            forecast_results_path=Path(config.forecast_results_path)
        )

    def get_logging_config(self) -> LoggingConfig:
        config = self.config.logging
        return LoggingConfig(
            log_file=Path(config.log_file),
            level=config.level
        )

    def get_hyperparams_config(self) -> HyperParametersConfig:
        p = self.params

        return HyperParametersConfig(
            forecast_horizon_days=p.forecast_horizon_days,
            prophet=ProphetHyperParams(
                seasonality_mode=p.prophet.seasonality_mode,
                daily_seasonality=p.prophet.daily_seasonality,
                yearly_seasonality=p.prophet.yearly_seasonality,
                weekly_seasonality=p.prophet.weekly_seasonality,
                monthly_seasonality=p.prophet.monthly_seasonality,
                changepoint_range=p.prophet.changepoint_range,
                changepoint_prior_scale=p.prophet.changepoint_prior_scale,
                Fourier_order=p.prophet.Fourier_order
            ),
            lstm=LSTMHyperParams(
                time_step=p.lstm.time_step,
                epochs=p.lstm.epochs,
                batch_size=p.lstm.batch_size,
                learning_rate=p.lstm.learning_rate,
                layers=p.lstm.layers,
                units=p.lstm.units,
                dropout=p.lstm.dropout
            ),
            garch=GARCHHyperParams(
                p=p.garch.p,
                q=p.garch.q,
                dist=p.garch.dist
            )
        )
