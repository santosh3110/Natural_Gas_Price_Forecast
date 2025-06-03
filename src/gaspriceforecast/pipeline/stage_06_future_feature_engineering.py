from gaspriceforecast.config.configuration import ConfigurationManager
from gaspriceforecast.components.future_feature_engineering import FutureFeatureEngineer
from gaspriceforecast.utils.logger import get_logger

logger = get_logger("pipeline_stage_06_future_feature_engineering.log")

STAGE_NAME = "Future Feature Engineering"


class FutureFeatureEngineeringPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feat_config = config.get_future_feature_engineering_config()
        engineer = FutureFeatureEngineer(feat_config)
        engineer.run()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>> {STAGE_NAME} started <<<<<\n")
        pipeline = FutureFeatureEngineeringPipeline()
        pipeline.main()
        logger.info(f"\n\n>>>>> {STAGE_NAME} completed successfully <<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e
