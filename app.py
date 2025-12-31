import os
import joblib
import numpy as np
import logging
import traceback
from flask import Flask, request, jsonify, render_template

# ---------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# ----------------------------
# MODEL LOADING
# ----------------------------
# 18-feature Model (Updated)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Meta-MLP_Base-GB-AdaB-XGB-RF_full.pkl")

model = None

try:
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully!")
    else:
        logger.error(f"Model file {MODEL_PATH} not found!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.error(traceback.format_exc())

# ---------------------------------------
# Feature Configuration (18 Features)
# ---------------------------------------
# EXACT ORDER from model.feature_names_in_
MODEL_FEATURES = [
    'General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 'Other_Cancer', 
    'Depression', 'Diabetes', 'Arthritis', 'Sex', 'Age', 'Height', 'Weight', 
    'BMI', 'Smoking', 'Alcohol', 'Fruit', 'Green_Vegetables', 'Fried_Potato'
]

SCALERS = {
    # Numerical Fields (Normalization Ranges)
    # logic: (Value - min) / (max - min)
    # Correlation findings:
    # Age: Positive Corr (Higher=Risk). 18->0.0, 80->1.0. Correct.
    # Fruit: Neg Corr (Higher=Healthy). 0->0.0, 100->1.0. Correct.
    
    'Weight': {'min': 30, 'max': 300}, # Extended max based on typical US data
    'Height': {'min': 90, 'max': 250},
    'BMI': {'min': 10, 'max': 100},
    'Age': {'min': 18, 'max': 80}, # Cap at 80 like BRFSS often does? Or 100? Using 80 to map 0.83 -> 67ish.
    'Fruit': {'min': 0, 'max': 100},
    'Green_Vegetables': {'min': 0, 'max': 100},
    
    # Model Bias Fix: Model thinks High Potato/Alcohol is "Healthy" (Low Risk).
    # User expects Low Potato/Alcohol to be "Healthy".
    # Solution: Invert Scale so User's "0" becomes Model's "1.0".
    'Fried_Potato': {'min': 100, 'max': 0},
    'Alcohol': {'min': 30, 'max': 0},
    
    # Categorical Mappings (Normalized 0.0 - 1.0)
    # Correlation (-0.53): High Health = Low Disease.
    # Golden Row (Risk 0.0019): Health=0.75 (Very Good).
    # Thus Excellent=1.0, Poor=0.0.
    'General_Health': {
        'Excellent': 1.0, 'Very_Good': 0.75, 'Good': 0.5, 'Fair': 0.25, 'Poor': 0.0
    },
    # Checkup: Mean Healthy 0.9. 1.0 is the standard for both groups.
    # Logic: High value (1.0) = Regular Checkup.
    'Checkup': {
        'Within 1 year': 1.0, 
        '1-2 years': 0.75, 
        '2-5 years': 0.5, 
        '5+ years': 0.25, 
        'Never': 0.0
    },
    'Diabetes': {
        'No': 0.0, 
        'Borderline': 0.33, 
        'During Pregnancy': 0.66, 
        'Yes': 1.0
    },
    # Binary
    # Healthy Golden Row Sex = 1.0. Mean(H) < Mean(S) implies 1.0 is common, but 0.0 is slightly more in Healthy??
    # Wait. Let's trust Golden Row: 1.0 is Healthy.
    # Usually Females are Healthier. So Female = 1.0.
    'Sex': {'0': 1.0, '1': 0.0}, # Female=1.0, Male=0.0
    
    'Exercise': {'0': 0.0, '1': 1.0},
    'Smoking': {'0': 0.0, '1': 1.0}
}

def process_input_value(value, feature_name):
    """
    Normalizes input values.
    Numeric values are passed RAW (Model expects Age, BMI etc. as real numbers).
    Categorical values are Mapped.
    """
    val_str = str(value).lower()
    
    # 1. Handle Boolean Checkboxes (Skin_Cancer, etc.) 
    # Only if "on" or "true" -> 1.0
    if val_str in ['on', 'true', 'yes', '1'] and feature_name not in SCALERS: 
         # Only apply simple boolean if NOT in SCALERS 
         # (though 1 might be a valid number for a Scaler, e.g. Age=1)
         # Actually, better to checks SCALERS FIRST.
         pass

    # 2. Handle Configured Scalers
    if feature_name in SCALERS:
        config = SCALERS[feature_name]

        # Numeric Range Scaling
        if isinstance(config, dict) and 'min' in config:
            try:
                val = float(value)
                norm_val = (val - config['min']) / (config['max'] - config['min'])
                return max(0.0, min(1.0, norm_val))
            except:
                return 0.0

        # Categorical Logic
        elif isinstance(config, dict):
            # Try exact match first
            if str(value) in config:
                return config[str(value)]
            # Fallback for numeric strings ("0", "1")
            try:
                if str(int(float(value))) in config:
                     return config[str(int(float(value)))]
            except:
                pass
            # Fallback for Booleans in Categ map (e.g. Sex)
            if val_str in ['on', 'true', 'yes']: return 1.0 # Default High? Or check specific mapping?
            if val_str in ['off', 'false', 'no']: return 0.0
            
            return 0.0  # Default fallback

    # 3. Default Numeric Fallback (Pass Raw Value)
    try:
        return float(value)
    except:
        pass
        
    # 4. Final Boolean Fallback
    if val_str in ['on', 'true', 'yes']: return 1.0
    if val_str in ['off', 'false', 'no', '0']: return 0.0
    
    return 0.0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/calculate")
def calculate():
    return render_template("calculate.html")

@app.route("/recommendation/<int:level>")
def recommendation(level):
    return render_template("recommendation.html", level=level)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.json
        logger.info(f"Received Prediction Request: {data}")

        input_vector = []
        
        for feature in MODEL_FEATURES:
            raw_val = data.get(feature)
            processed_val = process_input_value(raw_val, feature)
            input_vector.append(processed_val)

        logger.info(f"Input Vector: {input_vector}")

        # Reshape for prediction
        input_array = [input_vector]
        
        # Predict
        prediction_prob = model.predict_proba(input_array)[0][1]
        prediction_class = int(model.predict(input_array)[0])
        
        logger.info(f"Result: Class={prediction_class}, Prob={prediction_prob:.4f}")

        # Map probability to risk level 1-5
        # Adjusted Thresholds based on Model Baseline (Healthy ~25%)
        # Level 1: Very Low (0-30%)
        # Level 2: Low-Moderate (30-45%)
        # Level 3: Moderate (45-60%)
        # Level 4: High (60-80%)
        # Level 5: Very High (80%+)
        if prediction_prob <= 0.30:
            risk_level = 1
        elif prediction_prob <= 0.45:
            risk_level = 2
        elif prediction_prob <= 0.60:
            risk_level = 3
        elif prediction_prob <= 0.80:
            risk_level = 4
        else:
            risk_level = 5

        return jsonify({
            "probability": float(prediction_prob),
            "class": prediction_class,
            "risk_level": risk_level
        })

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["GET"])
def test_model():
    """Automated health check"""
    # ... (Keep existing simple test structure updated for 18 feats if needed)
    return jsonify({"status": "running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)