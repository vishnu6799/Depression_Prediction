# Depression_Prediction
A survey-based ML project to predict depression likelihood. Built with a scikit-learn preprocessing pipeline and a PyTorch MLP model, deployed via Streamlit. Users can input academic, work, and personal details for predictions, showcasing an end-to-end ML workflow.

📌 Problem Statement

Depression is a growing concern, influenced by factors like academic pressure, work pressure, and life satisfaction. Early prediction can raise awareness and encourage timely consultation with professionals.

⚙️ Tech Stack

Python

Scikit-learn → preprocessing pipeline

PyTorch → MLP neural network model

Streamlit → web app deployment

Joblib → save/load preprocessing pipeline

🧑‍💻 Model Training

Features are preprocessed using a ColumnTransformer.

Trained on survey-based dataset with academic, work, and personal life factors.

Model: Multi-Layer Perceptron (MLP) in PyTorch.

Saved artifacts:

preprocessor.pkl → preprocessing pipeline

depression_model.pth → trained model

🎯 Features of the App

✅ Collects user input (age, gender, CGPA, academic/work pressure, satisfaction, etc.)
✅ Handles missing values automatically
✅ Runs predictions using trained PyTorch MLP model
✅ Provides an interactive Streamlit interface
