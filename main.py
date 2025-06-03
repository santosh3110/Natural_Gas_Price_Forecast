from gaspriceforecast.pipeline.stage_00_data_ingestion import DataIngestionTrainingPipeline
from gaspriceforecast.pipeline.stage_01_data_preparation import PrepareDataTrainingPipeline
from gaspriceforecast.pipeline.stage_02_prophet_baseline import ProphetBaselinePipeline
from gaspriceforecast.pipeline.stage_03_lstm_model import LSTMModelPipeline
from gaspriceforecast.pipeline.stage_04_bilstm_model import BiLSTMPipeline
from gaspriceforecast.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from gaspriceforecast.pipeline.stage_06_future_feature_engineering import FutureFeatureEngineeringPipeline
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
    
    STAGE_NAME = "Data Preparation"
    try:
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} started <<<<<\n")
        data_preparation = PrepareDataTrainingPipeline()
        data_preparation.main()
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} completed <<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e
     
    STAGE_NAME = "Prophet Baseline"
    try:
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} started <<<<<\n")
        prophet_pipeline = ProphetBaselinePipeline()
        prophet_pipeline.main()
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} completed <<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e
    
    STAGE_NAME = "LSTM Model"
    try:
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} started <<<<<\n")
        lstm_pipeline = LSTMModelPipeline()
        lstm_pipeline.main()
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} completed <<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e
    
    STAGE_NAME = "BiLSTM Model"
    try:
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} started <<<<<\n")
        bilstm_pipeline = BiLSTMPipeline()
        bilstm_pipeline.main()
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} completed <<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e

    STAGE_NAME = "Model Evaluation"
    try:
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} started <<<<<\n")
        eval_pipeline = ModelEvaluationPipeline()
        eval_pipeline.main()
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} completed <<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e
    
    STAGE_NAME = "Future Feature Engineering"
    try:
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} started <<<<<\n")
        future_feat_pipeline = FutureFeatureEngineeringPipeline()
        future_feat_pipeline.main()
        logger.info(f"\n\n>>>>> stage {STAGE_NAME} completed <<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e


