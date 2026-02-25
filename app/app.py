from flask import Flask, request, jsonify, render_template
import sys
import os

# Add src to path to import CreditPredictor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import CreditPredictor

app = Flask(__name__, template_folder='templates')
# Define paths relative to this file
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'xgboost_model.joblib')
prep_path = os.path.join(base_dir, '..', 'models', 'preprocessor.joblib')

predictor = CreditPredictor(
    model_path=model_path, 
    preprocessor_path=prep_path
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Required fields check
        required_fields = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'EmploymentLength', 'LoanPurpose', 'PreviousDefaults', 'DTI']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        result = predictor.predict(data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
