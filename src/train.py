import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import logging
import os
from preprocessing import DataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.1, 1, 10]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
        }
        self.best_models = {}

    def train_and_tune(self, X_train, y_train):
        for name, config in self.models.items():
            logger.info(f"Training and tuning {name}...")
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.best_models[name] = grid_search.best_estimator_
            logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
            logger.info(f"Best ROC-AUC for {name}: {grid_search.best_score_:.4f}")

    def save_models(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        for name, model in self.best_models.items():
            model_path = os.path.join(path, f"{name.lower()}_model.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")

if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv('data/loan_data.csv')
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    preprocessor.save_preprocessor()
    
    # Save test data for evaluation
    test_data = {
        'X_test': X_test,
        'y_test': y_test
    }
    joblib.dump(test_data, 'data/test_data.joblib')
    
    # Train and save models
    trainer = ModelTrainer()
    trainer.train_and_tune(X_train, y_train)
    trainer.save_models()
