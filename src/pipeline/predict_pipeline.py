import pickle
import numpy as np
import pandas as pd


class PredictionPipeline:
    def __init__(self, model_path, preprocessor_path):
        """
        Initialize the PredictionPipeline with paths to the saved model and preprocessor.

        Args:
            model_path (str): Path to the trained model (.pkl file).
            preprocessor_path (str): Path to the preprocessor (.pkl file).
        """
        self.model = self.load_pickle(model_path)
        self.preprocessor = self.load_pickle(preprocessor_path)

    @staticmethod
    def load_pickle(file_path):
        """
        Load a pickle file.

        Args:
            file_path (str): Path to the pickle file.

        Returns:
            object: The loaded object.
        """
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def preprocess_input(self, input_data):
        """
        Preprocess the input data to match the format expected by the model.

        Args:
            input_data (pd.DataFrame or dict): Input data to preprocess.

        Returns:
            np.ndarray: Preprocessed input data.
        """
        # Ensure the input is a DataFrame
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        # Get the feature names from the preprocessor
        required_features = self.preprocessor.feature_names_in_

        # Add missing columns (set to 0) to match the required features
        for col in required_features:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match the preprocessor's expected order
        input_data = input_data[required_features]

        # Transform data using the preprocessor
        return self.preprocessor.transform(input_data)

    def predict(self, input_data):
        """
        Make predictions using the trained model.

        Args:
            input_data (pd.DataFrame or dict): Input data for prediction.

        Returns:
            dict: A dictionary containing the prediction and fraud probability.
        """
        # Preprocess the input data
        processed_data = self.preprocess_input(input_data)

        # Predict the fraud probability
        fraud_probability = self.model.predict_proba(processed_data)[:, 1]  # Probability of class 1 (fraud)
        prediction = (fraud_probability >= 0.5).astype(int)  # Convert probabilities to binary predictions

        return {
            "Prediction": prediction[0],
            "Fraud_Probability": fraud_probability[0],
        }
