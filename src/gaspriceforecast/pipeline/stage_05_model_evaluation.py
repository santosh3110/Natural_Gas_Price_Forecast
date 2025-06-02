from gaspriceforecast.config.configuration import ConfigurationManager
from gaspriceforecast.components.mlflow_model_evaluation import ModelEvaluatorWithMLflow
from gaspriceforecast.utils.logger import get_logger

logger = get_logger(log_file="pipeline_stage_05_model_evaluation.log")

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        evaluator = ModelEvaluatorWithMLflow(eval_config)
        evaluator.log_metrics_and_artifacts()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n")
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
        logger.info(f"\n\n>>>>>> {STAGE_NAME} completed successfully <<<<<<\n")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed due to: {e}")
        raise e
