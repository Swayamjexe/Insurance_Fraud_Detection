import os
import sys
from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define SVC parameters as in notebook
            svc_param_grid = {
                'kernel': ['rbf'],
                'gamma': [0.001, 0.01, 0.1, 1],
                'C': [1, 10, 50, 100, 200, 300, 1000]
            }

            # Create KFold object
            kfold = StratifiedKFold(n_splits=10)

            # Create and train SVC model with GridSearch
            svc = SVC(probability=True)
            grid_search = GridSearchCV(
                svc,
                param_grid=svc_param_grid,
                cv=kfold,
                scoring="accuracy",
                n_jobs=4,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_

            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate metrics
            logging.info(f"Training Accuracy: {grid_search.score(X_train, y_train)}")
            logging.info(f"Testing Accuracy: {grid_search.score(X_test, y_test)}")
            
            logging.info("\nClassification Report:")
            logging.info(f"\n{classification_report(y_test, y_test_pred)}")

            # Save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return {
                'Accuracy': accuracy_score(y_test, y_test_pred),
                'Precision': precision_score(y_test, y_test_pred),
                'Recall': recall_score(y_test, y_test_pred),
                'F1 Score': f1_score(y_test, y_test_pred)
            }

        except Exception as e:
            raise CustomException(e, sys)