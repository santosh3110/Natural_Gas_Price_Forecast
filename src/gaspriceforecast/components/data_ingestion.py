import os
import sys
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date
from gaspriceforecast.entity.config_entity import DataIngestionConfig
from gaspriceforecast.utils.logger import get_logger

logger = get_logger(log_file="data_ingestion.log")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_yahoo_data(self) -> pd.DataFrame:
        try:
            logger.info(" Downloading Natural Gas data from Yahoo Finance...")
            data = yf.download(
                self.config.ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False
            )
            data.reset_index(inplace=True)
            data.to_csv(self.config.yahoo_data_file, index=False)
            logger.info(f"Yahoo data saved to: {self.config.yahoo_data_file}")
            return data
        except Exception as e:
            logger.error(f" Failed to download Yahoo Finance data: {e}")
            raise

    def download_hdd_data(self) -> pd.DataFrame:
        logger.info("Downloading HDD data from NOAA FTP...")
        dfs = []
        current_year = date.today().year
        for year in range(self.config.hdd_year_start, current_year + 1):
            url = f"{self.config.hdd_base_url}/{year}/UtilityGas.Heating.txt"
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    content = res.text.splitlines()
                    header = content[3].split("|")
                    dates = [d for d in header if d.strip().isdigit()]
                    data_line = next((l for l in content[4:] if l.startswith("CONUS|")), None)
                    if data_line:
                        hdds = data_line.strip().split("|")[1:]
                        df = pd.DataFrame({
                            "Date": pd.to_datetime(dates, format="%Y%m%d"),
                            "Hdd": list(map(int, hdds))
                        })
                        dfs.append(df)
            except Exception as e:
                logger.warning(f" Skipped HDD year {year}: {e}")

        if dfs:
            full_df = pd.concat(dfs)
            full_df.to_csv(self.config.hdd_processed_file, index=False)
            logger.info(f"HDD data saved to: {self.config.hdd_processed_file}")
            return full_df
        else:
            logger.error("No HDD data downloaded.")
            raise ValueError("HDD data is empty.")

    def download_inventory_data(self) -> pd.DataFrame:
        logger.info("Downloading EIA Inventory data...")
        url = (
            "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
            f"?api_key={self.config.eia_api_key}"
            "&frequency=weekly"
            "&data[0]=value"
            "&facets[series][]=NW2_EPG0_SWO_R48_BCF"
            f"&start={self.config.eia_start_date}"
            f"&end={self.config.eia_end_date}"
            "&sort[0][column]=period"
            "&sort[0][direction]=asc"
            "&offset=0"
            "&length=5000"
        )

        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            data_json = res.json()
            records = data_json.get("response", {}).get("data", [])
            if not records:
                raise ValueError("Empty inventory response.")

            df = pd.DataFrame(records)
            df["Date"] = pd.to_datetime(df["period"])
            df.rename(columns={"value": "Inventory_Bcf"}, inplace=True)
            df["Inventory_Bcf"] = df["Inventory_Bcf"].astype(int)
            df = df[["Date", "Inventory_Bcf"]].sort_values("Date")
            df.to_csv(self.config.eia_inventory_file, index=False)
            logger.info(f"EIA Inventory saved to: {self.config.eia_inventory_file}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch inventory data: {e}")
            raise
