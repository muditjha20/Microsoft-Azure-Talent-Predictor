import requests
import json

url = "http://127.0.0.1:5000/predict"

# Load sample.json file
with open("f:/Project - Talent peredictor/microsoft-azure-talent-predictor/flask_app/sample.json", "r") as f:
    input_data = json.load(f)

response = requests.post(url, json=input_data)

print("Server response:")
print(response.json())

#output 
# Server response:
# {'prediction': [0]}