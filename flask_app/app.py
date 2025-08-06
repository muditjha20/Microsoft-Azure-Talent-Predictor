from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

# load model once at startup
model = joblib.load('../azure/xgb_model.pkl')  # Adjust path if needed

# 🚀 Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Azure Talent Predictor! Use the /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON
        input_data = request.get_json()

        # Convert to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            return jsonify({'error': 'Unsupported data format. Please use either a list or dict.'}), 400

        # Make prediction
        prediction = model.predict(df)

        # Return as JSON list
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
