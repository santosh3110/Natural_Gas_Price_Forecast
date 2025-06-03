import pandas as pd
import numpy as np
from prophet import Prophet
from arch import arch_model
from gaspriceforecast.entity.config_entity import FutureFeatureEngineeringConfig
from gaspriceforecast.utils.common import create_directories
from gaspriceforecast.utils.logger import get_logger

logger = get_logger("future_feature_engineering.log")


class FutureFeatureEngineer:
    def __init__(self, config: FutureFeatureEngineeringConfig):
        self.config = config
        self.params = config.params

    def simulate_with_prophet(self, df, col_name):
        df = df[['Date', col_name]].dropna().rename(columns={"Date": "ds", col_name: "y"})
        p_params = self.params["prophet_params"]

        model = Prophet(
            seasonality_mode=p_params["seasonality_mode"],
            daily_seasonality=p_params["daily_seasonality"],
            weekly_seasonality=p_params["weekly_seasonality"],
            yearly_seasonality=p_params["yearly_seasonality"],
            changepoint_prior_scale=p_params["changepoint_prior_scale"],
            changepoint_range=p_params["changepoint_range"]
        )

        model.add_seasonality(
            name="monthly",
            period=30.5,
            fourier_order=p_params["monthly_fourier_order"]
        )

        model.fit(df)
        future = model.make_future_dataframe(periods=self.params["forecast_horizon"])
        forecast = model.predict(future)

        return forecast[["ds", "yhat"]].tail(self.params["forecast_horizon"]).set_index("ds")["yhat"]

    def simulate_volatility_with_garch(self, df: pd.DataFrame):
        series = df[['Date', 'Return']].dropna()
        series['Return'] = series['Return'] * 100
        series.set_index("Date", inplace=True)
        g_params = self.params["garch_params"]
        forecast_horizon = self.params["forecast_horizon"]

        predictions = []

        for i in range(forecast_horizon):
            train_data = series[:-(forecast_horizon - i)]
            model = arch_model(
                train_data,
                vol=g_params["vol"],
                p=g_params["p"],
                q=g_params["q"],
                dist=g_params["dist"]
            )
            fitted_model = model.fit(disp="off")
            forecast = fitted_model.forecast(horizon=1, reindex=False)
            next_vol = np.sqrt(forecast.variance.values[-1][0]) * np.sqrt(252)
            predictions.append(next_vol / 100)

        future_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
        return pd.Series(
            predictions,
            index=future_index
            )

    def run(self):
        logger.info("Starting future feature engineering...")

        df = pd.read_csv(self.config.processed_data_path, parse_dates=["Date"])
        df_model = df.copy()
        horizon = self.params["forecast_horizon"]

        # Prophet Forecasts
        future_strength = self.simulate_with_prophet(df, "Technical_Strength")
        future_bcf = self.simulate_with_prophet(df, "Inventory_Bcf")
        future_hdd = self.simulate_with_prophet(df, "Hdd")
        future_volume = self.simulate_with_prophet(df, "Volume")
        future_volatility = self.simulate_volatility_with_garch(df)

        future_dates = future_strength.index
        last_known = df_model.iloc[-60:].copy()

        # Feature Engineering
        signal = pd.concat([df["Technical_Strength"].iloc[-self.params["rolling_signal_window"]:], future_strength])
        future_signal = signal.rolling(self.params["rolling_signal_window"]).mean().loc[future_dates]

        # Inventory
        full_bcf = pd.concat([df["Inventory_Bcf"].iloc[-40:], future_bcf])
        inv_cumsum_last = df_model["Inventory_cumsum"].iloc[-1]
        future_inventory_lag3 = full_bcf.shift(self.params["lag_inventory"]).loc[future_dates]
        future_inventory_cumsum = full_bcf.cumsum().loc[future_dates] + inv_cumsum_last
        future_inventory_ma30 = full_bcf.rolling(self.params["rolling_window"]).mean().loc[future_dates]

        # HDD
        full_hdd = pd.concat([df["Hdd"].iloc[-40:], future_hdd])
        hdd_cumsum_last = df_model["Hdd_cumsum"].iloc[-1]
        future_hdd_cumsum = full_hdd.cumsum().loc[future_dates] + hdd_cumsum_last
        future_hdd_ma30 = full_hdd.rolling(self.params["rolling_window"]).mean().loc[future_dates]

        # Volume
        full_vol = pd.concat([df["Volume"].iloc[-40:], future_volume])
        vol_cumsum_last = df_model["Volume_cumsum"].iloc[-1]
        future_volume_cumsum = full_vol.cumsum().loc[future_dates] + vol_cumsum_last
        future_volume_ma30 = full_vol.rolling(self.params["rolling_window"]).mean().loc[future_dates]

        # Final DF
        logger.info("Creating engineered feature DataFrame...")
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Close": 0,  # Placeholder
            "Technical_Strength": future_strength.values,
            "Technical_Strength_Signal": future_signal.values,
            "Hist_Vol": future_volatility.values,
            "Inventory_Bcf": future_bcf.values,
            "Inventory_Bcf_lag3": future_inventory_lag3.values,
            "Hdd_ma30": future_hdd_ma30.values,
            "Inventory_ma30": future_inventory_ma30.values,
            "Hdd_cumsum": future_hdd_cumsum.values,
            "Inventory_cumsum": future_inventory_cumsum.values,
            "Volume_ma30": future_volume_ma30.values,
            "Volume_cumsum": future_volume_cumsum.values
        })

        future_df.to_csv(self.config.output_path, index=False)
        logger.info(f"Saved future features to {self.config.output_path}")
