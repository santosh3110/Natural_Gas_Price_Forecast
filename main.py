from gaspriceforecast.pipeline.stage_00_data_ingestion import DataIngestionTrainingPipeline

from gaspriceforecast.utils.logger import get_logger

logger = get_logger(log_file="main_pipeline.log")

if __name__ == "__main__":

    STAGE_NAME = "Data Ingestion stage"
    try:
        logger.info(f"\n\n {STAGE_NAME} started ")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f" {STAGE_NAME} completed \n\nx==========x\n")

    except Exception as e:
        logger.exception(f" {STAGE_NAME} failed due to: {e}")
        raise e