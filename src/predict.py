import joblib
import pandas as pd
import numpy as np
import shap
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CreditPredictor:
    def __init__(self, model_path='models/xgboost_model.joblib', preprocessor_path='models/preprocessor.joblib'):
        self.model = joblib.load(model_path)
        prep_data = joblib.load(preprocessor_path)
        self.preprocessor = prep_data['preprocessor']
        self.feature_names = prep_data['feature_names']
        self.explainer = shap.TreeExplainer(self.model)

    def get_risk_level(self, prob):
        if prob < 0.3:
            return "Low"
        elif prob < 0.7:
            return "Medium"
        else:
            return "High"

    def predict(self, raw_data):
        """
        raw_data: dict containing keys like Age, Income, LoanAmount, etc.
        """
        df = pd.DataFrame([raw_data])
        
        # Transform data using the same preprocessor
        # Note: Preprocessor expects specific order and types
        X_transformed = self.preprocessor.transform(df)
        
        # Predict probability
        prob = self.model.predict_proba(X_transformed)[0, 1]
        risk_level = self.get_risk_level(prob)
        
        # SHAP explanation
        shap_values = self.explainer.shap_values(X_transformed)
        
        # Get top 3 contributing features
        # For a single sample, shap_values is (1, n_features)
        sample_shap = shap_values[0]
        feature_importance = list(zip(self.feature_names, sample_shap))
        
        # Sort by absolute value to find most impactful features
        top_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:3]
        top_features_list = [{"feature": str(f), "impact": round(float(i), 4)} for f, i in top_features]
        
        return {
            "probability": round(float(prob), 4),
            "risk_level": risk_level,
            "top_contributing_features": top_features_list
        }

if __name__ == "__main__":
    # Test prediction
    predictor = CreditPredictor()
    sample_input = {
        'Age': 45,
        'Income': 80000,
        'LoanAmount': 20000,
        'CreditScore': 720,
        'EmploymentLength': 10,
        'LoanPurpose': 'Debt Consolidation',
        'PreviousDefaults': 0,
        'DTI': 3.0
    }
    result = predictor.predict(sample_input)
    print("Test Prediction Result:")
    print(result)
