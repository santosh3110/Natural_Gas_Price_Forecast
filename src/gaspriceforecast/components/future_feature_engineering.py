import pandas as pd
import numpy as np
from prophet import Prophet
from arch import arch_model
from gaspriceforecast.entity.config_entity import FutureFeatureEngineeringConfig
from gaspriceforecast.utils.common import create_directories
from gaspriceforecast.utils.logger import get_logger
import plotly.graph_objs as go

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
        model.add_seasonality("monthly", period=30.5, fourier_order=p_params["monthly_fourier_order"])
        model.fit(df)

        future = model.make_future_dataframe(periods=self.params["forecast_horizon"])
        forecast = model.predict(future)

        return forecast[["ds", "yhat"]].tail(self.params["forecast_horizon"]).set_index("ds")["yhat"]
 
    def simulate_volatility_with_garch(self, df: pd.DataFrame):
        """Simulate future volatility using GARCH model"""
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


    def save_garch_plot(self, df: pd.DataFrame, forecast_series: pd.Series):
        logger.info("Saving GARCH forecast plot using Plotly...")
        hist_vol = df[['Date', 'Hist_Vol']].dropna().iloc[-600:]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist_vol['Date'],
            y=hist_vol['Hist_Vol'],
            mode='lines',
            name='Historical Volatility',
            line=dict(color='royalblue')
        ))

        fig.add_trace(go.Scatter(
            x=forecast_series.index,
            y=forecast_series.values,
            mode='lines+markers',
            name='Forecasted Volatility',
            line=dict(color='indianred')
        ))

        fig.update_layout(
            title="GARCH Forecasted Volatility vs Past 600 Days",
            xaxis_title="Date",
            yaxis_title="Volatility",
            height=600,
            width=1000,
            plot_bgcolor="white",
            margin=dict(t=50, b=50)
        )

        fig.write_html(self.config.garch_plot_path)
        logger.info(f"GARCH plot saved to {self.config.garch_plot_path}")

    def run(self):
        logger.info("Starting future feature engineering...")

        df = pd.read_csv(self.config.processed_data_path, parse_dates=["Date"])
        df_model = df.copy()
        horizon = self.params["forecast_horizon"]

        future_strength = self.simulate_with_prophet(df, "Technical_Strength")
        future_bcf = self.simulate_with_prophet(df, "Inventory_Bcf")
        future_hdd = self.simulate_with_prophet(df, "Hdd")
        future_volume = self.simulate_with_prophet(df, "Volume")
        future_volatility = self.simulate_volatility_with_garch(df)

        future_dates = future_strength.index
        last_known = df_model.iloc[-60:].copy()
        feature_order = [
            'Date', 'Close', 'Technical_Strength', 'Technical_Strength_Signal', 
            'Inventory_Bcf', 'Hist_Vol', 'Inventory_Bcf_lag3', 
            'Hdd_ma30', 'Inventory_ma30', 'Hdd_cumsum', 
            'Inventory_cumsum', 'Volume_ma30', 'Volume_cumsum'
        ]
        last_known = last_known[feature_order]
        last_known.to_csv(self.config.last_known_data_path, index=False)

        logger.info(f"Saved last known data to {self.config.last_known_data_path}")

        signal_window = self.params["rolling_signal_window"]
        signal = pd.concat([df["Technical_Strength"].iloc[-signal_window:], future_strength])
        future_signal = signal.rolling(signal_window).mean().loc[future_dates]

        inv_window = self.params["lag_inventory"]
        inv_bcf = pd.concat([df["Inventory_Bcf"].iloc[-40:], future_bcf])
        inv_cumsum_last = df_model["Inventory_cumsum"].iloc[-1]
        future_inventory_lag3 = inv_bcf.shift(inv_window).loc[future_dates]
        future_inventory_cumsum = inv_bcf.cumsum().loc[future_dates] + inv_cumsum_last
        future_inventory_ma30 = inv_bcf.rolling(self.params["rolling_window"]).mean().loc[future_dates]

        hdd = pd.concat([df["Hdd"].iloc[-40:], future_hdd])
        hdd_cumsum_last = df_model["Hdd_cumsum"].iloc[-1]
        future_hdd_cumsum = hdd.cumsum().loc[future_dates] + hdd_cumsum_last
        future_hdd_ma30 = hdd.rolling(self.params["rolling_window"]).mean().loc[future_dates]

        vol = pd.concat([df["Volume"].iloc[-40:], future_volume])
        vol_cumsum_last = df_model["Volume_cumsum"].iloc[-1]
        future_volume_cumsum = vol.cumsum().loc[future_dates] + vol_cumsum_last
        future_volume_ma30 = vol.rolling(self.params["rolling_window"]).mean().loc[future_dates]

        future_df = pd.DataFrame({
            "Date": future_dates,
            "Close": 0,
            "Technical_Strength": future_strength.values,
            "Technical_Strength_Signal": future_signal.values,
            "Inventory_Bcf": future_bcf.values,
            "Hist_Vol": future_volatility.values,
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

        #  Save the GARCH volatility plot
        self.save_garch_plot(df, future_volatility)
