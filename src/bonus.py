import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.metrics import accuracy_score
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BonusAnalysis:
    def __init__(self, model_path='models/xgboost_model.joblib', test_data_path='data/test_data.joblib', preprocessor_path='models/preprocessor.joblib'):
        self.model = joblib.load(model_path)
        self.test_data = joblib.load(test_data_path)
        self.prep_data = joblib.load(preprocessor_path)
        self.X_test = self.test_data['X_test']
        self.y_test = self.test_data['y_test']
        self.feature_names = self.prep_data['feature_names']

    def fairness_check(self):
        """Check for bias against age groups (as a proxy for protected class)."""
        logger.info("Performing fairness check...")
        
        # We need original 'Age' feature from test set. 
        # Since X_test is transformed, we need to map back or use the original df.
        # For simplicity, let's assume 'Age' is the first numeric feature.
        # X_test columns: ['Age', 'Income', 'LoanAmount', 'CreditScore', 'EmploymentLength', 'DTI', 'cat_...']
        
        # Extract Age (it was the first column in numeric_features)
        age_values = self.X_test[:, 0]
        
        # Create age groups: Young (<30), Middle (30-50), Senior (>50)
        # Note: Age was scaled, so we need to use the scaler to get back original values or just use percentiles.
        # For this bonus script, we'll just use percentiles.
        q1, q2 = np.percentile(age_values, [33, 66])
        sensitive_feature = np.where(age_values < q1, 'Young', np.where(age_values < q2, 'Middle', 'Senior'))
        
        y_pred = self.model.predict(self.X_test)
        
        metrics = {
            'accuracy': accuracy_score,
            'selection_rate': selection_rate
        }
        
        mf = MetricFrame(metrics=metrics, y_true=self.y_test, y_pred=y_pred, sensitive_features=sensitive_feature)
        
        print("\nFairness Metrics by Age Group:")
        print(mf.by_group)
        
        dp_diff = demographic_parity_difference(self.y_test, y_pred, sensitive_features=sensitive_feature)
        print(f"\nDemographic Parity Difference: {dp_diff:.4f}")
        
        os.makedirs('notebooks/plots/bonus', exist_ok=True)
        mf.by_group.plot.bar(subplots=True, figsize=(10, 8), title="Fairness Metrics by Age Group")
        plt.tight_layout()
        plt.savefig('notebooks/plots/bonus/fairness_metrics.png')
        plt.close()

    def compare_feature_importance(self):
        """Compare feature importance across different models."""
        logger.info("Comparing feature importance across models...")
        
        importance_data = {}
        
        # 1. XGBoost
        importance_data['XGBoost'] = self.model.feature_importances_
        
        # 2. Random Forest
        rf_model = joblib.load('models/randomforest_model.joblib')
        importance_data['RandomForest'] = rf_model.feature_importances_
        
        # 3. Logistic Regression (Coefficients)
        lr_model = joblib.load('models/logisticregression_model.joblib')
        importance_data['LogisticRegression'] = np.abs(lr_model.coef_[0])
        
        importance_df = pd.DataFrame(importance_data, index=self.feature_names)
        
        # Normalize for comparison
        importance_df = importance_df.div(importance_df.sum(axis=0), axis=1)
        
        plt.figure(figsize=(12, 8))
        importance_df.plot(kind='bar', figsize=(15, 8))
        plt.title('Normalized Feature Importance Comparison')
        plt.ylabel('Relative Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('notebooks/plots/bonus/feature_importance_comparison.png')
        plt.close()
        logger.info("Saved feature importance comparison plot.")

if __name__ == "__main__":
    bonus = BonusAnalysis()
    bonus.fairness_check()
    bonus.compare_feature_importance()
