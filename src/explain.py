import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExplainableAI:
    def __init__(self, model_path='models/xgboost_model.joblib', preprocessor_path='models/preprocessor.joblib'):
        self.model = joblib.load(model_path)
        prep_data = joblib.load(preprocessor_path)
        self.feature_names = prep_data['feature_names']
        self.explainer = None

    def generate_explanations(self, X_test):
        os.makedirs('notebooks/plots/explanations', exist_ok=True)
        
        # Initialize SHAP explainer
        logger.info("Initializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_test)
        
        # 1. Global Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, show=False)
        plt.title('Global Feature Importance (SHAP)')
        plt.tight_layout()
        plt.savefig('notebooks/plots/explanations/shap_summary.png')
        plt.close()
        logger.info("Saved global summary plot.")
        
        # 2. Local Explanation for a sample (index 0)
        plt.figure(figsize=(15, 5))
        shap.force_plot(
            self.explainer.expected_value, 
            shap_values[0, :], 
            X_test[0, :], 
            feature_names=self.feature_names, 
            matplotlib=True, 
            show=False
        )
        plt.title('Local Explanation for Sample 0')
        plt.tight_layout()
        plt.savefig('notebooks/plots/explanations/shap_force_plot_sample_0.png')
        plt.close()
        logger.info("Saved local force plot for sample 0.")

    def explain_regulatory_importance(self):
        explanation = """
        **Why Explainability is Critical in Banking (Regulatory Compliance)**
        
        1. **Fair Lending (ECOA/Reg B):** Banks must prove their models don't discriminate based on protected classes (race, gender, age).
        2. **Adverse Action Notices:** When a loan is denied, the bank is legally required to provide specific "reason codes" explaining why (e.g., "debt-to-income ratio too high").
        3. **Model Risk Management (SR 11-7):** Regulators require deep understanding of how models work, their limitations, and their impact on business decisions.
        4. **Transparency & Trust:** Customers and internal stakeholders need to trust the model's output before it can be deployed in production.
        """
        print(explanation)
        with open('notebooks/regulatory_explanation.md', 'w') as f:
            f.write(explanation)

if __name__ == "__main__":
    # Load test data
    test_data = joblib.load('data/test_data.joblib')
    X_test = test_data['X_test']
    
    # Generate explanations
    xai = ExplainableAI()
    xai.generate_explanations(X_test[:100]) # Use subset for speed
    xai.explain_regulatory_importance()
