from gaspriceforecast.config.configuration import ConfigurationManager
from gaspriceforecast.components.lstm_model import LSTMTrainer
from gaspriceforecast.utils.logger import get_logger

logger = get_logger(log_file="pipeline_stage_03_lstm_model.log")

STAGE_NAME = "LSTM Model Stage"

class LSTMModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        lstm_config = config.get_lstm_model_config()
        model_runner = LSTMTrainer(config=lstm_config)
        model_runner.run()

if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n")
        pipeline = LSTMModelPipeline()
        pipeline.main()
        logger.info(f"\n\n>>>>>> {STAGE_NAME} completed successfully <<<<<<\n")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed due to: {e}")
        raise e