# End-to-End Car Price Prediction System 🚗

A complete, end-to-end **Car Price Prediction** project built in **Python**, covering **EDA → feature engineering → modeling → evaluation → model persistence → Streamlit deployment**.
The trained model is a **RandomForestRegressor** wrapped inside a **scikit-learn Pipeline** (with a **ColumnTransformer**) and saved as a **pickle** file for production inference.

---

## 🌐 Live Demo
👉 https://car-price-predictionupdated-zbq9qxda5rkabnlhrnfwf8.streamlit.app/

---

## 1. Project Overview

This repository contains:
- A full ML workflow implemented in a notebook (EDA + training + evaluation)
- A production-ready preprocessing + model **pipeline** saved to disk
- A **Streamlit web app** for interactive price prediction

---

## 2. Objective

Build a robust regression system that predicts the target variable **`Price($)`** using car specifications and condition-related features, while ensuring:
- Clean separation of train/test (no leakage)
- Repeatable preprocessing via pipelines
- A deployable inference artifact (`.pkl`) compatible with a Streamlit UI

---

## 3. Machine Learning Workflow

1. Load dataset and validate schema
2. Perform Exploratory Data Analysis (EDA)
3. Clean data and select relevant features
4. Preprocess data using pipelines
5. Train a Random Forest regression model
6. Evaluate model performance
7. Save and reload the trained model
8. Deploy the predictor via a Streamlit web app

---

## 4. Dataset Description

- Size: **50,000 rows × 25 columns**
- Target: **`Price($)`**
- Task: **Supervised regression**
- Data types: Mixed **numerical** + **categorical** features

### Features Used

**Numerical**
- `CarAge`
- `Mileage(km)`
- `EngineSize(L)`
- `Horsepower`
- `Torque`
- `Doors`
- `Seats`
- `FuelEfficiency(L/100km)`

**Categorical**
- `Brand`
- `Condition`
- `FuelType`
- `Transmission`
- `DriveType`
- `BodyType`
- `AccidentHistory`
- `Insurance`
- `RegistrationStatus`

---

## 5. Feature Engineering & Preprocessing

### Key Decisions (Feature Engineering)

- Dropped high-cardinality columns to reduce sparse one-hot expansion (explicitly handling the **curse of dimensionality**):
	- `Model`, `Options`, `City`, `Color`, `Interior`
- Dropped `Year` due to redundancy with `CarAge`
- No hyperparameter tuning (Random Forest already achieved **~99% R²**)
- No heavy models due to local machine constraints
- No data leakage: all preprocessing is performed inside the pipeline

### Preprocessing (Inside the Pipeline)

- Numerical features: `StandardScaler`
- Categorical features: `OneHotEncoder(handle_unknown="ignore")`
- Combined using `ColumnTransformer` and executed within a single `Pipeline`

---

## 6. Model Details

- Model: **RandomForestRegressor**
- Training:
	- Train/test split
	- Pipeline-based preprocessing + training
- Model persistence:
	- Full pipeline (preprocessing + model) saved as `model/car_price_pipeline.pkl` using `pickle`

---

## 7. Why Random Forest?

Random Forest was selected because it:
- Captures non-linear relationships and feature interactions effectively
- Performs strongly on mixed feature types after encoding
- Is robust and accurate without heavy tuning
- Supports feature importance analysis for interpretability

---

## 8. Application (Streamlit App)

The Streamlit app provides:
- A form-based UI to enter car details
- Server-side loading of the saved pipeline (`.pkl`)
- Instant predictions using `model.predict(...)`

---

## 9. How to Run the Project

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the Streamlit application

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`).

---

## 10. Project Structure

```text
car-price-prediction/
├── data_notebook/
│   ├── car_price_dataset.csv
│   └── pred.ipynb
├── model/
│   └── car_price_pipeline.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## 11. Results

- The Random Forest model achieves approximately **~99% R² score** on a held-out test set.
- Evaluation includes:
	- **R² Score**
	- **MAE (Mean Absolute Error)**
- Diagnostics included:
	- Actual vs Predicted plot
	- Residual analysis
	- Feature importance ranking

---

## 12. Future Improvements

- Add selective target encoding (or frequency encoding) for high-cardinality categorical features
- Run hyperparameter optimization (RandomizedSearchCV / Optuna) on stronger compute resources
- Add K-Fold cross-validation for more stable generalization estimates
- Improve deployment with input validation and model versioning

---


