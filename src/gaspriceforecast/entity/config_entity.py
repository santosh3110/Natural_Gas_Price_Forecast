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
class BaselineModelConfig:
    model_path: Path


@dataclass
class LSTMModelConfig:
    model_path: Path
    checkpoint_path: Path
    tensorboard_log_dir: Path


@dataclass
class ForecastConfig:
    root_dir: Path
    forecast_results_path: Path


@dataclass
class LoggingConfig:
    level: str
    log_file: Path


@dataclass(frozen=True)
class ProphetHyperParams:
    seasonality_mode: str
    daily_seasonality: bool
    yearly_seasonality: bool
    weekly_seasonality: bool
    monthly_seasonality: bool
    changepoint_range: float
    changepoint_prior_scale: float
    Fourier_order: int


@dataclass(frozen=True)
class LSTMHyperParams:
    time_step: int
    epochs: int
    batch_size: int
    learning_rate: float
    layers: int
    units: int
    dropout: float


@dataclass(frozen=True)
class GARCHHyperParams:
    p: int
    q: int
    dist: str


@dataclass(frozen=True)
class HyperParametersConfig:
    forecast_horizon_days: int
    prophet: ProphetHyperParams
    lstm: LSTMHyperParams
    garch: GARCHHyperParams
