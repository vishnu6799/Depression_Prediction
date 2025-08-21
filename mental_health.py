import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# -----------------------------
# Define the MLP architecture
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Load Preprocessor & Model
# -----------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")

    # Calculate input dimension
    num_features = len(preprocessor.transformers_[0][2])  # numeric cols
    cat_features = len(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out()
    )
    input_dim = num_features + cat_features

    # Init model & load weights
    model = MLP(input_dim)
    model.load_state_dict(torch.load("depression_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return preprocessor, model

preprocessor, model = load_artifacts()

# -----------------------------
# Expected feature set
# -----------------------------
expected_cols = [
    "Gender", "Age", "City", "Working Professional or Student", "Profession",
    "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction",
    "Job Satisfaction", "Sleep Duration", "Dietary Habits", "Degree",
    "Have you ever had suicidal thoughts ?", "Work/Study Hours",
    "Financial Stress", "Family History of Mental Illness"
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Depression Prediction App")
st.write("Fill in the details below to predict depression risk.")

# Collect inputs (simplified form)
age = st.number_input("Age", min_value=10, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
working_status = st.selectbox("Working Professional or Student", ["Working Professional", "Student"])
sleep = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
suicidal = st.selectbox("Ever had suicidal thoughts?", ["Yes", "No"])
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
work_hours = st.slider("Work/Study Hours", 0, 16, 8)
financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)

# Build partial input
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Working Professional or Student": working_status,
    "Sleep Duration": sleep,
    "Dietary Habits": diet,
    "Have you ever had suicidal thoughts ?": suicidal,
    "Family History of Mental Illness": family_history,
    "Work/Study Hours": work_hours,
    "Financial Stress": financial_stress,
}])

# ✅ Fill missing expected columns with np.nan instead of pd.NA
for col in expected_cols:
    if col not in input_data.columns:
        input_data[col] = np.nan

# Reorder columns to match training
input_data = input_data[expected_cols]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    try:
        # Preprocess input
        X_input = preprocessor.transform(input_data)

        # Convert to tensor
        if hasattr(X_input, "toarray"):  # handle sparse matrix
            X_input_tensor = torch.tensor(X_input.toarray(), dtype=torch.float32)
        else:
            X_input_tensor = torch.tensor(X_input, dtype=torch.float32)

        # Predict
        with torch.no_grad():
            prediction = model(X_input_tensor)
            pred_class = (prediction >= 0.5).float().item()

        # Show result
        st.subheader("Prediction:")
        if pred_class == 1:
            st.error("🔴 High risk of Depression")
        else:
            st.success("🟢 Not Depressed")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
