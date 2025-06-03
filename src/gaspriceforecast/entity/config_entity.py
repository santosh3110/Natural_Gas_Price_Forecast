from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataIngestionConfig:
    root_dir: Path
    ticker: str
    start_date: str
    end_date: str
    yahoo_data_file: Path
    eia_api_key: str
    eia_start_date: str
    eia_end_date: str
    eia_inventory_file: Path
    hdd_raw_file: Path
    hdd_processed_file: Path
    hdd_base_url: str
    hdd_year_start: int


@dataclass
class PrepareDataConfig:
    root_dir: Path
    yahoo_data_path: Path
    hdd_path: Path
    inventory_path: Path
    processed_data_path: Path


@dataclass
class ProphetBaselineConfig:
    root_dir: Path
    model_file: Path
    forecast_file: Path
    forecast_plot: Path
    residual_plot: Path
    component_plot: Path
    metrics_file: Path
    horizon: int
    seasonality_mode: str
    daily_seasonality: bool
    weekly_seasonality: bool
    yearly_seasonality: bool
    monthly_seasonality: bool
    changepoint_prior_scale: float
    changepoint_range: float
    fourier_order: int
    future_days: int

@dataclass
class LSTMConfig:
    root_dir: Path
    processed_data_path: Path
    model_path: Path
    history_plot: Path
    prediction_plot: Path
    metrics_file: Path
    scaler_path: Path
    params: dict

@dataclass
class BiLSTMConfig:
    root_dir: Path
    processed_data_path: Path
    model_path: Path
    history_plot: Path
    prediction_plot: Path
    metrics_file: Path
    scaler_path: Path
    params: dict

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    lstm_metrics_path: Path
    bilstm_model_metrics_path: Path
    prophet_metrics_path: Path
    evaluation_report: Path
    evaluation_plot: Path

@dataclass
class FutureFeatureEngineeringConfig:
    root_dir: Path
    processed_data_path: Path
    output_path: Path
    params: dict

@dataclass
class LoggingConfig:
    level: str
    log_file: Path