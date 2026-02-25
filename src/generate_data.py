import pandas as pd
import numpy as np
import os

def generate_credit_data(n_samples=5000):
    np.random.seed(42)
    
    # Features
    age = np.random.randint(18, 70, n_samples)
    income = np.random.randint(20000, 150000, n_samples)
    loan_amount = np.random.randint(1000, 50000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    employment_length = np.random.randint(0, 40, n_samples)
    loan_purpose = np.random.choice(['Education', 'Home Improvement', 'Medical', 'Personal', 'Debt Consolidation'], n_samples)
    previous_defaults = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # Debt-to-Income ratio
    dti = (loan_amount / (income / 12)).round(2)
    
    # Target: Default probability (synthetic logic)
    # Default is more likely if: low credit score, high loan amount relative to income, previous defaults
    logit = (
        0.005 * loan_amount / 1000 - 
        0.01 * credit_score / 100 - 
        0.0001 * income / 1000 + 
        2.0 * previous_defaults +
        0.5 * dti -
        0.02 * employment_length +
        np.random.normal(0, 1, n_samples)
    )
    
    # Sigmoid to get probability
    prob = 1 / (1 + np.exp(-logit))
    default = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'EmploymentLength': employment_length,
        'LoanPurpose': loan_purpose,
        'PreviousDefaults': previous_defaults,
        'DTI': dti,
        'Default': default
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/loan_data.csv', index=False)
    print(f"Dataset generated with {n_samples} samples and saved to data/loan_data.csv")
    print(df.head())

if __name__ == "__main__":
    generate_credit_data()
