from gaspriceforecast.config.configuration import ConfigurationManager
from gaspriceforecast.components.forecast_with_lstm import ForecastWithLSTM
from gaspriceforecast.utils.logger import get_logger

logger = get_logger("stage_07_forecast_with_lstm.log")

STAGE_NAME = "Forecast With LSTM"

class ForecastWithLSTMPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        forecast_config = config.get_forecast_with_lstm_config()
        forecast = ForecastWithLSTM(config=forecast_config)
        forecast.run()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>> {STAGE_NAME} started <<<<<\n")
        pipeline = ForecastWithLSTMPipeline()
        pipeline.main()
        logger.info(f"\n\n>>>>> {STAGE_NAME} completed <<<<<\n")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed due to: {e}")
        raise e
