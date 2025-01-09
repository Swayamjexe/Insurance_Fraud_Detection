import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, data_path: str):
        try:
            # Read the data
            data = pd.read_csv(data_path)
            
            # Replace '?' with np.nan
            data = data.replace('?', np.nan)

            # Handle missing values
            data['collision_type'].fillna(data['collision_type'].mode()[0], inplace=True)
            data['property_damage'].fillna('NO', inplace=True)
            data['police_report_available'].fillna('NO', inplace=True)
            data['authorities_contacted'].fillna('NONE', inplace=True)

            # Encode fraud_reported first
            data['fraud_reported'] = data['fraud_reported'].replace(('Y','N'),(1,0)).infer_objects(copy=False)

            # Create incident period
            bins = [-1, 3, 6, 9, 12, 17, 20, 24]
            category = ["past_midnight", "early_morning", "morning", 'fore-noon', 'afternoon', 'evening', 'night']
            data['incident_period'] = pd.cut(data['incident_hour_of_the_day'], bins, labels=category).astype(object)

            # Transform numerical variables to categorical
            data['number_of_vehicles_involved'] = data['number_of_vehicles_involved'].apply(str)
            data['witnesses'] = data['witnesses'].apply(str)
            data['bodily_injuries'] = data['bodily_injuries'].apply(str)

            # Create dummies
            categorical_columns = [
                'policy_state', 'insured_sex', 'insured_education_level', 'insured_occupation',
                'insured_hobbies', 'insured_relationship', 'incident_type', 'collision_type',
                'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city',
                'number_of_vehicles_involved', 'property_damage', 'bodily_injuries', 'witnesses',
                'police_report_available', 'auto_make', 'auto_model', 'incident_period'
            ]
            
            dummies = pd.get_dummies(data[categorical_columns])

            # First drop policy_bind_date
            data = data.drop(columns=['policy_bind_date'])

            # Then drop other unnecessary columns
            columns_to_drop = [
                'policy_number', 'policy_csl', 'insured_zip', 'incident_date',
                'incident_location', 'auto_year', 'incident_hour_of_the_day'
            ] + categorical_columns

            data = data.drop(columns=columns_to_drop)

            # Combine features exactly as in notebook
            x = pd.concat([dummies, data], axis=1)
            x_unscaled = x.iloc[:, 0:-1]  # predictor features
            y = x.iloc[:, -1]  # target variable

            # Apply SMOTE on unscaled data first
            smote = SMOTE()
            x_resample, y_resample = smote.fit_resample(x_unscaled, y.values.ravel())

            # Split the resampled data
            X_train, X_test, y_train, y_test = train_test_split(
                x_resample, y_resample,
                test_size=0.2,
                random_state=0
            )

            # Scale after splitting
            scaler = StandardScaler(with_mean=False)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Save preprocessing object
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=scaler
            )

            # Combine features and target
            train_arr = np.c_[X_train_scaled, y_train]
            test_arr = np.c_[X_test_scaled, y_test]

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)