# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pickle

# OOP
class LoanModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.numerical = []
        self.categorical = []
        self.model = None
        self.preprocessor = {
            'encoder': None,
            'scaler': None
        }
    
    def eda(self):
        self.data = pd.read_csv(self.data_path)
        
        # Fix gender column
        self.data["person_gender"] = self.data["person_gender"].apply(
            lambda x: x.strip().lower().replace(" ", "") if isinstance(x, str) else x
        )
        
        # Drop missing value
        self.data = self.data.dropna(subset=['person_income'])
        
        # Separate features
        for col in self.data.columns:
            if self.data[col].dtype in ['int64', 'float64']:
                self.numerical.append(col)
            elif self.data[col].dtype == 'object':
                self.categorical.append(col)
        
        # Remove target from feature
        self.numerical.remove('loan_status')
    
    def preprocess(self):
        X = self.data[self.numerical + self.categorical]
        y = self.data['loan_status']
        
        # OneHot Encoding
        self.preprocessor['encoder'] = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
        X_cat = self.preprocessor['encoder'].fit_transform(X[self.categorical])
        
        # Scaling
        self.preprocessor['scaler'] = StandardScaler()
        X_num = self.preprocessor['scaler'].fit_transform(X[self.numerical])
        
        # Combine features
        X_processed = pd.DataFrame(
            np.concatenate([X_num, X_cat], axis = 1),
            columns = self.numerical + list(self.preprocessor['encoder'].get_feature_names_out(self.categorical))
        )
        
        # Splitting
        return train_test_split(X_processed, y, test_size = 0.2, random_state = 42)
    
    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess()
        
        self.model = XGBClassifier(random_state = 42, eval_metric = 'logloss')
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluation
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print("Model Performance:")
        print(classification_report(self.y_test, y_pred))
        print(f"ROC AUC: {roc_auc_score(self.y_test, y_proba):.4f}")
        
    def save(self, filepath):
        assets = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': {
                'numerical': self.numerical,
                'categorical': self.categorical
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(assets, f)

if __name__ == "__main__":
    trainer = LoanModel("Dataset_A_loan.csv")
    trainer.eda()
    trainer.train()
    trainer.save("loan_model.pkl")



