# Import libraries
import pickle
import pandas as pd
import numpy as np

# Prediction
class LoanInference:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            assets = pickle.load(f)
        
        self.model = assets['model']
        self.preprocessor = assets['preprocessor']
        self.numerical = assets['feature_names']['numerical']
        self.categorical = assets['feature_names']['categorical']
    
    def predict(self, input_data):
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Preprocessing
        X_cat = self.preprocessor['encoder'].transform(input_df[self.categorical])
        X_num = self.preprocessor['scaler'].transform(input_df[self.numerical])
        
        X_processed = np.concatenate([X_num, X_cat], axis = 1)
        
        pred = self.model.predict(X_processed)[0]
        proba = self.model.predict_proba(X_processed)[0][1]
        
        return {
            'Prediction': int(pred),
            'Probability': float(proba),
            'Status': 'Approved' if pred == 1 else 'Rejected',
            'Confidence': f"{max(proba, 1 - proba) * 100:.1f}%"
        }

if __name__ == "__main__":
    # Model Prediction
    predictor = LoanInference("loan_model.pkl")
    
    # Input from dataset
    sample_input = {
    'person_age': 22.0,
    'person_income': 71948.0,
    'person_emp_exp': 0,
    'loan_amnt': 35000.0,
    'loan_int_rate': 16.02,
    'loan_percent_income': 0.49,
    'cb_person_cred_hist_length': 3.0,
    'credit_score': 561,
    'person_gender': 'female',
    'person_education': 'Master',
    'person_home_ownership': 'RENT',
    'loan_intent': 'PERSONAL',
    'previous_loan_defaults_on_file': 'No'
}
    
    # Make the prediction
    result = predictor.predict(sample_input)
    print("Prediction Result:", result)

