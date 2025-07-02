// server.js
const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Store for recent predictions (in production, use a database)
let recentPredictions = [];

// Python script for model prediction
const pythonScript = `
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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
        df = pd.DataFrame([data])
        
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
`;

// Save Python script to file
//fs.writeFileSync('predict.py', pythonScript);

// Routes

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'OK', message: 'Fraud Detection API is running' });
});

// Get model info
app.get('/api/model/info', (req, res) => {
    res.json({
        name: 'Random Forest Fraud Detector',
        version: '1.0.0',
        features: [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ],
        description: 'Machine learning model for detecting credit card fraud',
        lastTrained: new Date().toISOString()
    });
});

// Predict fraud
app.post('/api/predict', async (req, res) => {
    try {
        const inputData = req.body;
        
        // Validate input data
        const requiredFeatures = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ];
        
        const missingFeatures = requiredFeatures.filter(feature => 
            inputData[feature] === undefined || inputData[feature] === null
        );
        
        if (missingFeatures.length > 0) {
            return res.status(400).json({
                error: 'Missing required features',
                missingFeatures
            });
        }
        
        // Call Python prediction script
        const python = spawn('python', ['predict.py']);
        
        let result = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            result += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const prediction = JSON.parse(result);
                    
                    if (prediction.error) {
                        return res.status(500).json(prediction);
                    }
                    
                    // Add timestamp and store recent prediction
                    const predictionWithTimestamp = {
                        ...prediction,
                        timestamp: new Date().toISOString(),
                        id: Date.now().toString()
                    };
                    
                    recentPredictions.unshift(predictionWithTimestamp);
                    
                    // Keep only last 100 predictions
                    if (recentPredictions.length > 100) {
                        recentPredictions = recentPredictions.slice(0, 100);
                    }
                    
                    res.json(predictionWithTimestamp);
                } catch (parseError) {
                    res.status(500).json({
                        error: 'Failed to parse prediction result',
                        details: parseError.message
                    });
                }
            } else {
                res.status(500).json({
                    error: 'Python script execution failed',
                    details: error
                });
            }
        });
        
        // Send input data to Python script
        python.stdin.write(JSON.stringify(inputData));
        python.stdin.end();
        
    } catch (error) {
        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

// Get recent predictions
app.get('/api/predictions/recent', (req, res) => {
    const limit = parseInt(req.query.limit) || 10;
    res.json({
        predictions: recentPredictions.slice(0, limit),
        total: recentPredictions.length
    });
});

// Get prediction statistics
app.get('/api/stats', (req, res) => {
    const totalPredictions = recentPredictions.length;
    const fraudPredictions = recentPredictions.filter(p => p.prediction === 1).length;
    const normalPredictions = totalPredictions - fraudPredictions;
    
    const avgConfidence = totalPredictions > 0 
        ? recentPredictions.reduce((sum, p) => sum + p.confidence, 0) / totalPredictions 
        : 0;
    
    const riskDistribution = {
        HIGH: recentPredictions.filter(p => p.risk_level === 'HIGH').length,
        MEDIUM: recentPredictions.filter(p => p.risk_level === 'MEDIUM').length,
        LOW: recentPredictions.filter(p => p.risk_level === 'LOW').length
    };
    
    res.json({
        totalPredictions,
        fraudPredictions,
        normalPredictions,
        fraudRate: totalPredictions > 0 ? (fraudPredictions / totalPredictions * 100).toFixed(2) : 0,
        avgConfidence: avgConfidence.toFixed(4),
        riskDistribution
    });
});

// Generate sample data for testing
app.get('/api/sample-data', (req, res) => {
    // Generate random sample data for testing
    const sampleData = {
        Time: Math.random() * 172800, // Random time in seconds
        Amount: Math.random() * 1000 + 1, // Random amount between 1-1000
    };
    
    // Generate random values for V1-V28 (PCA components)
    for (let i = 1; i <= 28; i++) {
        sampleData[`V${i}`] = (Math.random() - 0.5) * 20; // Random values between -10 and 10
    }
    
    res.json(sampleData);
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Error:', error);
    res.status(500).json({
        error: 'Internal server error',
        message: error.message
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        error: 'Endpoint not found'
    });
});

app.listen(PORT, () => {
    console.log(`🚀 Fraud Detection API running on port ${PORT}`);
    console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
    console.log(`🔍 Model info: http://localhost:${PORT}/api/model/info`);
});

module.exports = app;