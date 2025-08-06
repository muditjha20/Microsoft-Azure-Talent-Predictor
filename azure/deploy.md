# ------------------------------------------------------------
# 📦 Azure ML CLI Deployment Simulation
# ------------------------------------------------------------

# 🔁 1. Create an Azure ML workspace
az ml workspace create --name azure-talent-workspace --resource-group my-rg --location eastus

# 📂 2. Register the model in the workspace
az ml model register \
  --name azure-hire-predictor \
  --path xgb_model.pkl \
  --resource-group my-rg \
  --workspace-name azure-talent-workspace \
  --description "XGBoost model to predict hiring eligibility for SWE candidates"

# 🧠 3. Define the scoring script and environment
az ml environment create --file env.yml --name azure-hire-env

# 🚀 4. Deploy the model as a web service (ACI for testing)
az ml online-endpoint create --name hire-predictor-endpoint --file score.py

# ✅ 5. Invoke the endpoint with sample data (JSON format)
az ml online-endpoint invoke --name hire-predictor-endpoint --request-file sample.json

# 📛 6. Delete the deployment if needed
az ml online-endpoint delete --name hire-predictor-endpoint
