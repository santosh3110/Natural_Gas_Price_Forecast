from gaspriceforecast.config.configuration import ConfigurationManager
from gaspriceforecast.components.prophet_baseline import ProphetBaseline
from gaspriceforecast.utils.logger import get_logger

logger = get_logger(log_file="pipeline_stage_03_prophet_baseline.log")

STAGE_NAME = "Prophet Baseline Stage"

class ProphetBaselinePipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prophet_config = config.get_prophet_baseline_config()
        model_runner = ProphetBaseline(config=prophet_config)
        model_runner.run()

if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n")
        pipeline = ProphetBaselinePipeline()
        pipeline.main()
        logger.info(f"\n\n>>>>>> {STAGE_NAME} completed successfully <<<<<<\n")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed due to: {e}")
        raise e
