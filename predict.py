
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

required_features = [
                            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
                            ]

def load_model():
    try:
        model = joblib.load('random_forest_fraud_detector.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        print(json.dumps({"error": f"Error loading model: {str(e)}"}))
        sys.exit(1)

def predict_fraud(data, model, scaler):
    try:
        # Convert to DataFrame
       

        df = pd.DataFrame([[data[feature] for feature in required_features]], columns=required_features)

        
        # Scale the data
        scaled_data = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0]
        
        # Get feature importance for this prediction
        feature_importance = model.feature_importances_
        feature_names = df.columns.tolist()
        
        top_features = sorted(
            zip(feature_names, feature_importance), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        result = {
            "prediction": int(prediction),
            "probability": {
                "normal": float(probability[0]),
                "fraud": float(probability[1])
            },
            "confidence": float(max(probability)),
            "risk_level": "HIGH" if probability[1] > 0.7 else "MEDIUM" if probability[1] > 0.3 else "LOW",
            "top_features": [{"name": name, "importance": float(imp)} for name, imp in top_features]
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": f"Prediction error: {str(e)}"}))

if __name__ == "__main__":
    model, scaler = load_model()
    
    # Read input data from stdin
    input_data = json.loads(sys.stdin.read())
    predict_fraud(input_data, model, scaler)
