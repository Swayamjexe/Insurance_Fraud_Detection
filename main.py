from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
import sys

def train_pipeline():
    try:
        logging.info("Training pipeline started")
        
        # Data Ingestion
        data_ingestion = DataIngestion()
        raw_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(raw_data_path)
        logging.info("Data transformation completed")

        # Model Training
        model_trainer = ModelTrainer()
        metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info("Model training completed")

        # Print metrics
        logging.info("Final model performance metrics:")
        for metric_name, score in metrics.items():
            logging.info(f"{metric_name}: {score:.3f}")

        return metrics

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        metrics = train_pipeline()
        print("\nTraining Pipeline Completed Successfully!")
        print("\nModel Performance Metrics:")
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.3f}")
    except Exception as e:
        print(f"Error occurred: {e}")