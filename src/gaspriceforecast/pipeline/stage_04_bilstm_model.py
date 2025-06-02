from gaspriceforecast.config.configuration import ConfigurationManager
from gaspriceforecast.components.bilstm_model import BiLSTMTrainer
from gaspriceforecast.utils.logger import get_logger

logger = get_logger(log_file="pipeline_stage_04_bilstm_model.log")

STAGE_NAME = "BiLSTM Model Training"

class BiLSTMPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        bilstm_config = config.get_bilstm_model_config()
        model_runner = BiLSTMTrainer(config=bilstm_config)
        model_runner.run()

if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>> {STAGE_NAME} started <<<<<\n")
        pipeline = BiLSTMPipeline()
        pipeline.main()
        logger.info(f"\n\n>>>>> {STAGE_NAME} completed successfully <<<<<\n")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed due to: {e}")
        raise e
