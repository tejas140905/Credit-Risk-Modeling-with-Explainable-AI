import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, target_column='Default'):
        self.target_column = target_column
        self.preprocessor = None
        self.feature_names = None

    def handle_outliers(self, df, columns):
        """Simple IQR-based outlier clipping."""
        df_cleaned = df.copy()
        for col in columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
        return df_cleaned

    def prepare_pipeline(self, numeric_features, categorical_features):
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        return self.preprocessor

    def fit_transform(self, df):
        logger.info("Starting data preprocessing...")
        
        # Identify numeric and categorical columns
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if self.target_column in numeric_features:
            numeric_features.remove(self.target_column)
            
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle outliers for numeric columns
        df = self.handle_outliers(df, numeric_features)
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Initialize and fit pipeline
        self.prepare_pipeline(numeric_features, categorical_features)
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        # Get feature names after one-hot encoding
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        self.feature_names = numeric_features + cat_feature_names
        
        # Handle class imbalance with SMOTE on training data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
        
        logger.info(f"Preprocessing complete. Training shape: {X_train_resampled.shape}")
        
        return X_train_resampled, X_test_transformed, y_train_resampled, y_test

    def save_preprocessor(self, path='models/preprocessor.joblib'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names
        }, path)
        logger.info(f"Preprocessor saved to {path}")

if __name__ == "__main__":
    # Test the preprocessor
    df = pd.read_csv('data/loan_data.csv')
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    preprocessor.save_preprocessor()
