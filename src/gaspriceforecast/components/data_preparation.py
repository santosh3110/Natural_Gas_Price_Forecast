import pandas as pd
import numpy as np
from pathlib import Path
from gaspriceforecast.entity.config_entity import PrepareDataConfig
from gaspriceforecast.utils.logger import get_logger
from gaspriceforecast.utils.common import create_directories
from gunsan_strength.core import get_gunsan_strength

logger = get_logger(log_file="prepare_data_component.log")


class PrepareData:
    def __init__(self, config: PrepareDataConfig):
        self.config = config

    def load_and_merge(self) -> pd.DataFrame:
        try:
            logger.info("Loading Yahoo Finance, HDD, and Inventory data...")
            yf_df = pd.read_csv(self.config.yahoo_data_path, parse_dates=["Date"])
            hdd_df = pd.read_csv(self.config.hdd_path, parse_dates=["Date"])
            inv_df = pd.read_csv(self.config.inventory_path, parse_dates=["Date"])

            logger.info(f"Yahoo data shape: {yf_df.shape}")
            logger.info(f"HDD data shape: {hdd_df.shape}")
            logger.info(f"Inventory data shape: {inv_df.shape}")

            logger.info("Computing GunSan Technical Strength index...")
            gunsan_df = get_gunsan_strength(yf_df, signal_window=50)

            # Merge back 'Volume' into GunSan output
            if 'Volume' in yf_df.columns:
                gunsan_df = pd.merge(gunsan_df, yf_df[['Date', 'Volume']], on='Date', how='left')

            logger.info("Merging datasets on 'Date'...")
            df = gunsan_df.copy()
            df = pd.merge(df, hdd_df, on="Date", how="left")
            df = pd.merge(df, inv_df, on="Date", how="left")

            logger.info("Creating return & historical volatility features...")
            df["Return"] = np.log(df['Close'] / df['Close'].shift(1))
            df["Hist_Vol"] = df["Return"].rolling(window=252).std() * np.sqrt(252)

            logger.info("Interpolating missing values...")
            df["Inventory_Bcf"] = df["Inventory_Bcf"].interpolate(method='linear', limit_direction='both')
            df["Hdd"] = df["Hdd"].interpolate(method='linear', limit_direction='both')

            df = df[df['Date'] >= "2010-01-01"]
            df = df.dropna()

            logger.info(f"Final merged data shape: {df.shape}")
            logger.info(f"Columns:\n{df.dtypes}")
            logger.info(f"Nulls:\n{df.isnull().sum()}")
            logger.info(f"Duplicates: {df.duplicated().sum()}")

            self.final_df = df
            return df

        except Exception as e:
            logger.error(f"Error during merging/cleaning: {e}")
            raise

    def feature_engineering(self):
        try:
            logger.info("Starting feature engineering...")
            df = self.final_df

            df['Inventory_Bcf_lag3'] = df['Inventory_Bcf'].shift(3)
            df['Hdd_ma30'] = df['Hdd'].rolling(30).mean()
            df['Inventory_ma30'] = df['Inventory_Bcf'].rolling(30).mean()
            df['Hdd_cumsum'] = df['Hdd'].cumsum()
            df['Inventory_cumsum'] = df['Inventory_Bcf'].cumsum()
            df['Volume_ma30'] = df['Volume'].rolling(30).mean()
            df['Volume_cumsum'] = df['Volume'].cumsum()

            df = df.dropna()
            self.final_df = df

            logger.info("Feature engineering completed.")
            logger.info(f"Feature engineered data info: {df.info()}")
            logger.info(f"Columns:\n{df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

    def save(self):
        try:
            logger.info("Saving processed data to disk...")
            create_directories([self.config.root_dir])
            self.final_df.to_csv(self.config.processed_data_path, index=False)
            logger.info(f"Processed data saved at: {self.config.processed_data_path}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise
