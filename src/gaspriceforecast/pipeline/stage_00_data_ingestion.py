from gaspriceforecast.config.configuration import ConfigurationManager
from gaspriceforecast.components.data_ingestion import DataIngestion
from gaspriceforecast.utils.logger import get_logger

logger = get_logger(log_file="pipeline_stage_00_data_ingestion.log")

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # Ingestion Steps
        data_ingestion.download_yahoo_data()
        data_ingestion.download_hdd_data()
        data_ingestion.download_inventory_data()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n{STAGE_NAME} started ")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f"{STAGE_NAME} completed successfully \n\n")

    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed due to: {e}")
        raise e
