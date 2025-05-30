from gaspriceforecast.config.configuration import ConfigurationManager
from gaspriceforecast.components.data_preparation import PrepareData
from gaspriceforecast.utils.logger import get_logger

logger = get_logger(log_file="pipeline_stage_02_prepare_data.log")

STAGE_NAME = "Prepare Data Stage"


class PrepareDataTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_data_config = config.get_prepare_data_config()
        prepare_data = PrepareData(config=prepare_data_config)

        # Pipeline steps
        prepare_data.load_and_merge()
        prepare_data.feature_engineering()
        prepare_data.save()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n{STAGE_NAME} started ")
        pipeline = PrepareDataTrainingPipeline()
        pipeline.main()
        logger.info(f"{STAGE_NAME} completed successfully \n\n")

    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed due to: {e}")
        raise e
