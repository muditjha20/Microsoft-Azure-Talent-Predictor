import joblib
import pandas as pd
import numpy as np
import json
import os

# This will be called once at the beginning
def init():
    global model
    model_path = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")
    model = joblib.load(model_path)

# This will be called each time new data comes in
def run(raw_data):
    try:
        # raw_data is expected as JSON string so convert to DataFrame
        if isinstance(raw_data, str):
            data = pd.read_json(raw_data, orient="records")

            # if recieved as dict
        elif isinstance(raw_data, dict):
            data = pd.DataFrame.from_dict([raw_data])

             # if recieved as a list
        elif isinstance(raw_data, list):
            data = pd.DataFrame.from_dict(raw_data)

            # unsupported format
        else:
            return "Unsupported data format"
        
        # Predict using loaded model
        predictions = model.predict(data)

        # Return predictions as a list (so they can be serialized)
        return predictions.tolist()

    except Exception as e:
        return str(e)
