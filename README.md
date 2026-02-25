# Credit Risk Modeling with Explainable AI (XAI)

## 📌 Problem Statement
Predicting loan default probability is a critical task for financial institutions. However, traditional "black-box" models (like XGBoost or Neural Networks) are often insufficient in banking due to regulatory requirements. This project builds a production-ready pipeline that not only predicts defaults but also provides **human-interpretable explanations** for every decision using SHAP.

## 💼 Business Impact
- **Reduced Credit Loss:** Early detection of high-risk applicants.
- **Regulatory Compliance:** Automated generation of "reason codes" for loan denials (ECOA/Reg B).
- **Operational Efficiency:** Faster loan processing with automated risk scoring.
- **Fairness & Transparency:** Bias detection and feature importance analysis ensure ethical AI.

## 📊 Model Comparison Table
Based on internal testing (5,000 synthetic samples):

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.9555 | 0.7753 | 0.8560 | 0.8745 |
| Random Forest | 0.9216 | 0.8358 | 0.8766 | 0.8382 |
| XGBoost | 0.9060 | 0.8754 | 0.8904 | 0.8214 |

*Note: Logistic Regression performed exceptionally well due to the linear nature of synthetic data generation.*

## 🛠️ Project Structure
```
credit-risk-model/
│
├── data/               # Raw and processed datasets
├── notebooks/          # EDA and Evaluation plots
│   ├── plots/
│   │   ├── evaluation/
│   │   ├── explanations/
│   │   └── bonus/
├── src/                # Modular source code
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   ├── bonus.py        # Fairness & Model comparison
├── models/             # Saved joblib models
├── app/                # Flask deployment
│   ├── app.py
├── requirements.txt
└── README.md
```

## 🚀 Deployment Instructions
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the Models:**
   ```bash
   python src/train.py
   ```
3. **Run the API:**
   ```bash
   python app/app.py
   ```

## 📡 Sample API Request/Response
**POST** `/predict`

**Request Body:**
```json
{
    "Age": 45,
    "Income": 80000,
    "LoanAmount": 20000,
    "CreditScore": 720,
    "EmploymentLength": 10,
    "LoanPurpose": "Debt Consolidation",
    "PreviousDefaults": 0,
    "DTI": 3.0
}
```

**Response:**
```json
{
    "probability": 0.996,
    "risk_level": "High",
    "top_contributing_features": [
        {"feature": "EmploymentLength", "impact": 2.417},
        {"feature": "DTI", "impact": 0.9844},
        {"feature": "Income", "impact": 0.8162}
    ]
}
```

## 🔍 Explainability & Fairness
- **SHAP Summary Plot:** Shows global drivers of credit risk across the entire portfolio.
- **SHAP Force Plot:** Explains individual loan applications.
- **Fairness Check:** Demographic parity analysis across age groups to ensure non-discriminatory lending.

## 🔮 Future Improvements
- **Alternative Data:** Integrate social media or utility payment history.
- **Dynamic Thresholding:** Optimize the default probability threshold based on business cost-benefit analysis.
- **Model Monitoring:** Implement drift detection for production models.
- **Cloud Deployment:** Containerize using Docker and deploy to AWS/GCP.
"# Credit-Risk-Modeling-with-Explainable-AI" 
