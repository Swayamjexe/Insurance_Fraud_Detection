import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from imblearn.over_sampling import SMOTE

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load the dataset
            df = pd.read_csv('notebook\data\transformed_insurance_claims.csv')
            logging.info('Read the dataset as dataframe')

            # Separate features and target variable
            X = df.drop('fraud_reported', axis=1)  
            y = df['fraud_reported']  

            # Apply SMOTE to balance the dataset
            smote = SMOTE(random_state=42)
            X_resample, y_resample = smote.fit_resample(X, y)

            logging.info("Resampling done using SMOTE")

            # Combine the resampled data back into a DataFrame
            resampled_df = pd.concat([X_resample, y_resample], axis=1)

            # Create the directory to save the data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw resampled data
            resampled_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Train-test split (after resampling)
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(resampled_df, test_size=0.2, random_state=42)

            # Save the train and test sets to CSV
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
