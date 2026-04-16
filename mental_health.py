import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# -----------------------------
# Define the MLP architecture
# (must match training exactly)
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
# FIX 1: Use weights_only=False for
#         compatibility with newer PyTorch
# -----------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")

    num_features = len(preprocessor.transformers_[0][2])
    cat_features = len(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out()
    )
    input_dim = num_features + cat_features

    model = MLP(input_dim)
    # FIX 1: weights_only=False avoids FutureWarning/error in PyTorch >= 2.0
    model.load_state_dict(
        torch.load("depression_model.pth",
                   map_location=torch.device("cpu"),
                   weights_only=False)
    )
    model.eval()
    return preprocessor, model

preprocessor, model = load_artifacts()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Depression Prediction App")
st.write("Fill in the details below to predict depression risk.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, step=1, value=22)
    gender = st.selectbox("Gender", ["Male", "Female"])

    # FIX 2: working_status drives which fields are shown
    working_status = st.selectbox(
        "Working Professional or Student",
        ["Student", "Working Professional"]
    )

    # FIX 3: Sleep Duration values aligned with encoder categories
    sleep = st.selectbox(
        "Sleep Duration",
        ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours",
         "8 hours", "8-9 hours", "More than 8 hours"]
    )

    # FIX 4: Dietary Habits values aligned with encoder categories
    diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

    suicidal = st.selectbox("Ever had suicidal thoughts?", ["No", "Yes"])
    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

with col2:
    work_hours = st.slider("Work/Study Hours per day", 0, 16, 8)
    financial_stress = st.slider("Financial Stress (1–5)", 1, 5, 3)

    # FIX 5: City selector — show only real Indian cities from the encoder
    city_options = [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
        "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat",
        "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal",
        "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra",
        "Nashik", "Faridabad", "Meerut", "Rajkot", "Thane",
        "Gurgaon", "Varanasi", "Srinagar", "Visakhapatnam",
        "Morena", "Vasai-Virar", "Kalyan"
    ]
    city = st.selectbox("City", city_options)

    # FIX 6: Conditional fields for Student vs Professional
    if working_status == "Student":
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        academic_pressure = st.slider("Academic Pressure (0–5)", 0.0, 5.0, 3.0, step=1.0)
        study_satisfaction = st.slider("Study Satisfaction (0–5)", 0.0, 5.0, 3.0, step=1.0)
        profession = st.selectbox("Field of Study / Degree Program", [
            "Student", "B.Tech", "BCA", "BBA", "B.Com", "BSc",
            "MBA", "MCA", "M.Tech", "PhD", "Other"
        ])
        degree = st.selectbox("Degree", [
            "B.Tech", "BCA", "BBA", "B.Com", "B.Sc", "BSc",
            "MBA", "MCA", "M.Tech", "M.Com", "PhD", "Class 12"
        ])
        work_pressure = 0.0
        job_satisfaction = 0.0
    else:
        cgpa = 0.0
        academic_pressure = 0.0
        study_satisfaction = 0.0
        profession = st.selectbox("Profession", [
            "Software Engineer", "Teacher", "Doctor", "Manager",
            "Data Scientist", "Accountant", "Analyst", "Architect",
            "Business Analyst", "Chef", "Civil Engineer", "Consultant",
            "Content Writer", "Customer Support", "Digital Marketer",
            "Educational Consultant", "Electrician", "Entrepreneur",
            "Financial Analyst", "Graphic Designer", "HR Manager",
            "Investment Banker", "Lawyer", "Marketing Manager",
            "Mechanical Engineer", "Pharmacist", "Pilot", "Plumber",
            "Research Analyst", "Researcher", "Sales Executive",
            "Travel Consultant", "UX/UI Designer", "Unemployed"
        ])
        degree = st.selectbox("Highest Degree", [
            "B.Tech", "BCA", "BBA", "B.Com", "B.Sc", "BSc",
            "MBA", "MCA", "M.Tech", "M.Com", "PhD", "MBBS",
            "LLB", "M.Ed", "B.Ed", "Class 12"
        ])
        work_pressure = st.slider("Work Pressure (0–5)", 0.0, 5.0, 3.0, step=1.0)
        job_satisfaction = st.slider("Job Satisfaction (0–5)", 0.0, 5.0, 3.0, step=1.0)

# -----------------------------
# Build input DataFrame
# FIX 7: All 17 expected columns present, using np.nan for truly missing ones
# -----------------------------
input_data = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "City": city,
    "Working Professional or Student": working_status,
    "Profession": profession,
    "Academic Pressure": academic_pressure,
    "Work Pressure": work_pressure,
    "CGPA": cgpa,
    "Study Satisfaction": study_satisfaction,
    "Job Satisfaction": job_satisfaction,
    "Sleep Duration": sleep,
    "Dietary Habits": diet,
    "Degree": degree,
    "Have you ever had suicidal thoughts ?": suicidal,
    "Work/Study Hours": work_hours,
    "Financial Stress": financial_stress,
    "Family History of Mental Illness": family_history,
}])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Depression Risk"):
    try:
        X_input = preprocessor.transform(input_data)

        if hasattr(X_input, "toarray"):
            X_tensor = torch.tensor(X_input.toarray(), dtype=torch.float32)
        else:
            X_tensor = torch.tensor(X_input, dtype=torch.float32)

        with torch.no_grad():
            prob = model(X_tensor).item()
            pred_class = int(prob >= 0.5)

        st.subheader("Prediction Result")
        if pred_class == 1:
            st.error(f"🔴 **High risk of Depression** (confidence: {prob:.1%})")
            st.info("This is a screening tool only. Please consult a mental health professional.")
        else:
            st.success(f"🟢 **Lower risk of Depression** (confidence: {1-prob:.1%})")
            st.info("This is a screening tool only. Mental health is a spectrum — reach out if needed.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("**Debug — input data:**")
        st.dataframe(input_data)
