# 🌾 Crop Yield Prediction App
<img width="1893" height="920" alt="image" src="https://github.com/user-attachments/assets/cbd36df9-39b0-404b-bb92-b820ec008138" />


This project provides a Machine Learning-based solution for predicting crop yields across 10 major crops: **Rice, Maize, Cotton, Groundnut, Moong, Potato, Sesamum, Sugarcane, Urad, and Wheat**.
---

## Project Overview
- Predicts crop yields using agro-climatic and cultivation data.
- Supports **multi-crop yield estimation** through pre-trained `.sav` models.
- Interactive **Streamlit web app** for easy user interaction.
- Designed for farmers, agri-consultants, and research bodies.

---

## Project Structure
```bash
Crop_Project/
├── app.py  # Streamlit App (main file)
├── models/
│   ├── rice/
│   │   ├── model.sav
│   │   ├── scaler.sav
│   │   ├── encoders.sav
│   │   ├── cat_cols.sav
│   │   ├── num_cols.sav
│   │   ├── columns.sav
│   │   └── threshold.sav
│   ├── maize/
│   │   └── (same files)
│   ├── cotton/
│   ├── groundnut/
│   ├── moong/
│   ├── potato/
│   ├── sesamum/
│   ├── sugarcane/
│   ├── urad/
│   └── wheat/

```
## Installation Steps
- Run the Streamlit App : python -m streamlit run app2.py
## Tech Stack
- Python 3.13.3

- Streamlit

- Scikit-learn

- Pandas, Numpy

- Pickle (.sav models)

## Dataset
- Sourced from Kaggle public datasets.

- Contains crop-specific yield records, climatic data, and soil parameters.

- Preprocessed & feature-engineered during model training phase.

## Features
- Supports real-time yield predictions for 10 major crops.

- Quantile Regression models to handle yield variability.

- Lightweight and deployable on free-tier Streamlit Cloud.

- Dynamic input form for feature values.

- Categorical label encoding handling unseen values safely.
