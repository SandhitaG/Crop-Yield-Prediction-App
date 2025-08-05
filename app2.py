import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# ---- Load model components for the selected crop ----
@st.cache_resource
def load_model_components(crop_name):
    base_dir = r"C:/Users/user/Desktop/Crop_Project/models"
    base_path = os.path.join(base_dir, crop_name.lower())
    
    model = pickle.load(open(os.path.join(base_path, 'model.sav'), 'rb'))
    scaler = pickle.load(open(os.path.join(base_path, 'scaler.sav'), 'rb'))
    encoders = pickle.load(open(os.path.join(base_path, 'encoders.sav'), 'rb'))
    cat_cols = pickle.load(open(os.path.join(base_path, 'cat_cols.sav'), 'rb'))
    num_cols = pickle.load(open(os.path.join(base_path, 'num_cols.sav'), 'rb'))
    columns = pickle.load(open(os.path.join(base_path, 'columns.sav'), 'rb'))
    threshold = pickle.load(open(os.path.join(base_path, 'threshold.sav'), 'rb'))
    
    return model, scaler, encoders, cat_cols, num_cols, columns, threshold


# ---- Sidebar Configuration ----
st.sidebar.title("🌱 Crop Selector")
model_dir = "C:/Users/user/Desktop/Crop_Project/models"
available_crops = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
selected_crop = st.sidebar.selectbox("Select a Crop", available_crops)

# ---- Load model and metadata ----
model, scaler, encoders, cat_cols, num_cols, columns, threshold = load_model_components(selected_crop)

# ---- App Main UI ----
st.title("🌾 Crop Yield Prediction App")
st.markdown(f"### Selected Crop: `{selected_crop.capitalize()}`")
st.write("Please enter the required input features:")

# ---- Dynamic Input Fields ----
user_input = {}
for col in columns:
    if col in cat_cols:
        user_input[col] = st.text_input(f"{col} (categorical)", "")
    else:
        user_input[col] = st.number_input(f"{col} (numeric)", value=0.0)

# ---- Prediction ----
if st.button("Predict Yield"):
    try:
        input_df = pd.DataFrame([user_input])

        # Encode categorical variables
        for col in cat_cols:
            le = encoders[col]
            # Handle unseen labels
            if input_df[col][0] not in le.classes_:
                input_df[col] = le.transform([le.classes_[0]])  # fallback to first known label
            else:
                input_df[col] = le.transform(input_df[col])

        # Scale numeric features
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Ensure column order
        input_df = input_df[columns]

        # Predict yield
        prediction = model.predict(input_df)[0]
        classification = "High Yield 🌟" if prediction >= threshold else "Low Yield 🌾"

       
        st.info(f"🧮 Classified as: **{classification}** ")

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")
