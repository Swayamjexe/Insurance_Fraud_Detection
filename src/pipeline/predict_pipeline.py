# predict_pipeline.py
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PredictionPipeline:
    def __init__(self, model_path, preprocessor_path):
        """
        Initialize the PredictionPipeline with paths to the saved model and preprocessor.
        """
        self.model = self.load_pickle(model_path)
        self.scaler = self.load_pickle(preprocessor_path)
        
        # Store the expected dummy columns from training
        self.dummy_columns = None  # Will be set in the first run
        
    @staticmethod
    def load_pickle(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)
    
    def create_consistent_dummies(self, dummies_df, reference_columns):
        """
        Ensure dummy DataFrame has all expected columns in correct order
        """
        # Add missing columns with 0s
        for col in reference_columns:
            if col not in dummies_df.columns:
                dummies_df[col] = 0
                
        # Remove extra columns that weren't in training
        dummies_df = dummies_df[reference_columns]
        
        return dummies_df

    def preprocess_input(self, input_data):
        """
        Preprocess the input data to match the training pipeline exactly.
        """
        try:
            # Replace '?' with np.nan
            data = input_data.replace('?', np.nan)
            
            # Handle missing values
            data['collision_type'].fillna(data['collision_type'].mode()[0], inplace=True)
            data['property_damage'].fillna('NO', inplace=True)
            data['police_report_available'].fillna('NO', inplace=True)
            data['authorities_contacted'].fillna('NONE', inplace=True)

            # Create incident period
            bins = [-1, 3, 6, 9, 12, 17, 20, 24]
            category = ["past_midnight", "early_morning", "morning", 'fore-noon', 'afternoon', 'evening', 'night']
            data['incident_period'] = pd.cut(data['incident_hour_of_the_day'], bins, labels=category).astype(object)

            # Transform numerical variables to categorical
            data['number_of_vehicles_involved'] = data['number_of_vehicles_involved'].apply(str)
            data['witnesses'] = data['witnesses'].apply(str)
            data['bodily_injuries'] = data['bodily_injuries'].apply(str)

            # Define categorical columns
            categorical_columns = [
                'policy_state', 'insured_sex', 'insured_education_level', 'insured_occupation',
                'insured_hobbies', 'insured_relationship', 'incident_type', 'collision_type',
                'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city',
                'number_of_vehicles_involved', 'property_damage', 'bodily_injuries', 'witnesses',
                'police_report_available', 'auto_make', 'auto_model', 'incident_period'
            ]
            
            # Create dummies with drop_first=True as in training
            dummies = pd.get_dummies(data[categorical_columns], drop_first=True)
            
            # If this is the first run, store the dummy columns
            if self.dummy_columns is None:
                # Get the feature names from the scaler
                self.dummy_columns = [col for col in self.scaler.feature_names_in_ 
                                    if any(col.startswith(f"{cat}_") for cat in categorical_columns)]
            
            # Ensure consistent dummy columns
            dummies = self.create_consistent_dummies(dummies, self.dummy_columns)

            # Drop unnecessary columns
            columns_to_drop = [
                'policy_bind_date', 'policy_number', 'policy_csl', 'insured_zip', 
                'incident_date', 'incident_location', 'auto_year', 'incident_hour_of_the_day'
            ] + categorical_columns

            # Get numerical columns (updated to include claim columns)
            numerical_columns = [
                'months_as_customer', 'age', 'policy_deductable', 
                'policy_annual_premium', 'umbrella_limit',
                'capital-gains', 'capital-loss', 'injury_claim',
                'property_claim', 'vehicle_claim', 'total_claim_amount'
            ]
            
            # Keep only numerical columns in data
            numerical_data = data[numerical_columns]

            # Combine features in the correct order
            x = pd.concat([numerical_data, dummies], axis=1)

            # Ensure column order matches training
            x = x[self.scaler.feature_names_in_]

            # Scale the features
            scaled_features = self.scaler.transform(x)
            
            return scaled_features

        except Exception as e:
            raise Exception(f"Error in preprocessing: {str(e)}")

    def predict(self, input_data):
        """
        Make predictions using the trained model.
        """
        try:
            # Preprocess the input data
            processed_data = self.preprocess_input(input_data)

            # Make predictions
            predictions = self.model.predict(processed_data)
            probabilities = self.model.predict_proba(processed_data)[:, 1]

            # Create results DataFrame
            results = pd.DataFrame({
                'Prediction': ['Fraudulent' if pred == 1 else 'Legitimate' for pred in predictions],
                'Fraud_Probability': probabilities.round(3)
            })

            return results

        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")