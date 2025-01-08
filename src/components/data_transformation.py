import sys
import pandas as pd
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.exception import CustomException
from src.logger import logging
import sys

import os
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        pass

    def preprocess_data(self, data: pd.DataFrame):
        try:
            logging.info("Replacing missing values")

            # Replace missing values



            data['collision_type'].fillna(data['collision_type'].mode()[0], inplace=True)
            data['property_damage'].fillna('NO', inplace=True)
            data['police_report_available'].fillna('NO', inplace=True)
            data['authorities_contacted'].fillna('NONE', inplace=True)

            # Convert target variable to numerical
            data['fraud_reported'] = data['fraud_reported'].replace(('Y', 'N'), (1, 0)).infer_objects(copy=False)

            # Create incident period bins
            bins = [-1, 3, 6, 9, 12, 17, 20, 24]
            category = [
                "past_midnight",
                "early_morning",
                "morning",
                'fore-noon',
                'afternoon',
                'evening',
                'night'
            ]
            data['incident_period'] = pd.cut(data.incident_hour_of_the_day, bins, labels=category).astype(object)

            # Apply transformations to categorical variables
            data['number_of_vehicles_involved'] = data['number_of_vehicles_involved'].apply(str)
            data['witnesses'] = data['witnesses'].apply(str)
            data['bodily_injuries'] = data['bodily_injuries'].apply(str)

            # Create dummies for categorical variables
            dummies = pd.get_dummies(data[['policy_state', 'insured_sex', 'insured_education_level',
                                           'insured_occupation', 'insured_hobbies', 'insured_relationship',
                                           'incident_type', 'collision_type', 'incident_severity',
                                           'authorities_contacted', 'incident_state', 'incident_city',
                                           'number_of_vehicles_involved', 'property_damage', 'bodily_injuries',
                                           'witnesses', 'police_report_available', 'auto_make', 'auto_model',
                                           'incident_period']])

            # Drop unnecessary columns
            data = data.drop(columns=['policy_bind_date', 'policy_number', 'policy_csl', 'insured_zip',
                                       'incident_date', 'incident_location', 'auto_year',
                                       'incident_hour_of_the_day', 'policy_state', 'insured_sex',
                                       'insured_education_level', 'insured_occupation', 'insured_hobbies',
                                       'insured_relationship', 'incident_type', 'collision_type',
                                       'incident_severity', 'authorities_contacted', 'incident_state',
                                       'incident_city', 'number_of_vehicles_involved', 'property_damage',
                                       'bodily_injuries', 'witnesses', 'police_report_available',
                                       'auto_make', 'auto_model', 'incident_period'])

            # Concatenate transformed columns
            data = pd.concat([dummies, data], axis=1)

            logging.info("Data transformation completed successfully")
            return data

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, data_path: str):
        try:
            logging.info(f"Reading data from {data_path}")
            data = pd.read_csv(data_path)

            transformed_data = self.preprocess_data(data)

            # Split features and target
            X = transformed_data.drop(columns=['fraud_reported'])
            y = transformed_data['fraud_reported']

            # Handle imbalanced data using SMOTE
            logging.info("Applying SMOTE to handle imbalanced data")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Standardize features
            logging.info("Scaling features")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_resampled)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_resampled, test_size=0.2, random_state=42)

            train_array = np.c_[X_train, y_train]
            test_array = np.c_[X_test, y_test]

            return train_array, test_array

        except Exception as e:
            raise CustomException(e, sys)
