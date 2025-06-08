from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import io
import os

app = Flask(__name__)

# Load data
actual_df = pd.read_csv("artifacts/prepare_data/processed_data.csv", parse_dates=["Date"])
lstm_df = pd.read_csv("artifacts/forecast_with_lstm/forecasted_prices.csv", parse_dates=["Date"])
bilstm_df = pd.read_csv("artifacts/forecast_with_bilstm/forecasted_prices.csv", parse_dates=["Date"])
garch_path = "artifacts/future_feature_engineering/garch_forecast.png"

# GunSan Strength Plot
def plot_gunsan_strength(df):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1
    )
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='royalblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Technical_Strength'], name='Technical Strength', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Technical_Strength_Signal'], name='Signal Line', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=[0]*len(df), line=dict(color='black', dash='dash'), name='Zero Line', showlegend=False), row=2, col=1)
    fig.update_layout(height=800, width=1200, title="GunSan Strength & Close Price", showlegend=True)
    return fig.to_html(full_html=False)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/page1")
def page1():
    gunsan_html = plot_gunsan_strength(actual_df[-600:])
    return render_template("page1.html", gunsan_plot=gunsan_html)

@app.route("/page2")
def garch_forecast_plot():
    with open("artifacts/future_feature_engineering/garch_forecast.html", "r") as f:
        plot_html = f.read()
    return render_template("page2.html", plot_html=plot_html)


@app.route("/page3")
def page3():
    model = request.args.get("model", "lstm")
    forecast_df = lstm_df if model == "lstm" else bilstm_df

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_df["Date"].iloc[-600:], y=actual_df["Close"].iloc[-600:], name="Actual Close"))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecasted_Close"], name=f"{model.upper()} Forecast"))
    fig.update_layout(title=f"Natural Gas Price Forecast ({model.upper()})", width=1200, height=500)
    forecast_plot = fig.to_html(full_html=False)

    return render_template("page3.html", forecast_plot=forecast_plot, model=model)

@app.route("/download")
def download():
    model = request.args.get("model", "lstm")
    df = lstm_df if model == "lstm" else bilstm_df
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(io.BytesIO(buffer.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name=f"{model}_forecast.csv")

@app.route("/api/forecast")
def api():
    model = request.args.get("model", "lstm")
    df = lstm_df if model == "lstm" else bilstm_df
    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
