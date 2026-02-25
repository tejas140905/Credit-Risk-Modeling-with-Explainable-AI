import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
)
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, models_dir='models/', test_data_path='data/test_data.joblib'):
        self.models_dir = models_dir
        self.test_data = joblib.load(test_data_path)
        self.X_test = self.test_data['X_test']
        self.y_test = self.test_data['y_test']
        self.results = {}

    def evaluate_all(self):
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.joblib', '').capitalize()
            logger.info(f"Evaluating {model_name}...")
            
            model = joblib.load(os.path.join(self.models_dir, model_file))
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            metrics = {
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_prob)
            }
            self.results[model_name] = metrics
            
            self.plot_metrics(model_name, y_pred, y_prob)
            
        self.save_summary()

    def plot_metrics(self, model_name, y_pred, y_prob):
        os.makedirs('notebooks/plots/evaluation', exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'notebooks/plots/evaluation/{model_name}_cm.png')
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(self.y_test, y_prob):.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve - {model_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(f'notebooks/plots/evaluation/{model_name}_roc.png')
        plt.close()

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision)
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(f'notebooks/plots/evaluation/{model_name}_pr.png')
        plt.close()

    def save_summary(self):
        summary_df = pd.DataFrame(self.results).T
        summary_df.to_csv('models/model_comparison.csv')
        logger.info("Evaluation summary saved to models/model_comparison.csv")
        print("\nModel Comparison Table:")
        print(summary_df)

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate_all()
