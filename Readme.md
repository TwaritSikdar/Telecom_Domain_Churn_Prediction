# Telecom Domain Churn Prediction

## 1. Overview
This project provides a robust solution for predicting customer churn for a telecom company. The application consists of a machine learning model deployed via a Python web API (supporting both FastAPI and Flask) and an interactive web frontend (HTML/CSS/JS).

The backend API handles:
1. Receiving customer features via the `/predict` endpoint.
2. Data preprocessing (scaling and one-hot encoding).
3. Model inference using the pre-trained Hyperparameter tuned Stacking Classifier.

## 2. Model Development and Analysis
The machine learning model was developed and trained in a Jupyter Notebook. This file contains the full end-to-end data science process, including:

* **Exploratory Data Analysis (EDA):** Initial data cleanup (Converting `TotalCharges` from object to float), Donut Charts (using Pie Charts) and Histplots to visualize distributions of values in Categorical and Numerical columns respectively.

* **Model Training:** Used several models in training in which Voting Classifier turned to be the best model and After hyperparameter tuning 3 best performing models (AdaBoost Classifier, Logistic Regression and LinearSVC) were stacked into a Stacking Classifier model.

* **Artifact Generation:** The process of fitting and saving all necessary preprocessing tools (`One_Hot_Encoder.pkl` , `Standard_Scaler.pkl`) and the final model ( `Customer_Churn_Telecom_Domain.pkl` ) for deployment.

**Source File:** `Telecom_Domain_Project_New_using_Bay_Optimisation.ipynb`

## 3. Prerequisites
You must have **Python 3.8+** installed. All necessary Python dependencies are listed in the respective requirements files.

## 4. Project Structure
```
├── Telecom_Domain_Project_New_using_Bay_Optimisation.ipynb
├── Telecom_Domain_Project_New_using_ADASYN.ipynb
├── Telecom_Domain_Project_New_using_SMOTE_ENN.ipynb
├── Telecom_Domain_Project.ipynb
├── Telcom Data.csv
├── Customer_churn_predictor_app/
│    ├── Resources/
│    │    ├── Columns_to_encode.json
│    │    ├── Customer_Churn_Telecom_Domain.pkl
│    │    ├── Feature_names.json
│    │    ├── One_Hot_Encoder.pkl
│    │    └── Standard_Scaler.pkl
│    ├── static/
│    │    ├── fonts/
│    │    │    ├── Dosis-VariableFont_wght.woff2
│    │    │    ├── Jost-VariableFont_wght.woff2
│    │    │    ├── OpenSans-VariableFont_wdth,wgth.woff2
│    │    │    └── ShareTech-Regular.woff2
│    │    └── style.css
│    ├── templates/
│    │    └── index.html
│    ├── App.py
│    └── requirements.txt
├── Customer_churn_predictor_app_FastAPI/
│    ├── Resources/
│    │    ├── Columns_to_encode.json
│    │    ├── Customer_Churn_Telecom_Domain.pkl
│    │    ├── Feature_names.json
│    │    ├── One_Hot_Encoder.pkl
│    │    └── Standard_Scaler.pkl
│    ├── static/
│    │    ├── fonts/
│    │    │    ├── Dosis-VariableFont_wght.woff2
│    │    │    ├── Jost-VariableFont_wght.woff2
│    │    │    ├── OpenSans-VariableFont_wdth,wgth.woff2
│    │    │    └── ShareTech-Regular.woff2
│    │    └── style.css
│    ├── templates/
│    │    └── index.html
│    ├── app.py
│    ├── requirements.txt  
│    └── schema.py
├── Readme.md
└── requirements.txt
```

## 5. Running the Application
Now, this project has two APIs one built using Flask and another one using FastAPI.

### A. FastAPI
FastAPI is used for high-performance, asynchronous API serving. This scenario uses the `app.py` file.

1. **Install Dependencies:** 
    ```
        pip install -r requirements.txt
    ```

2. **Run Development Server (Uvicorn):** 
Use this command during local development as it includes automatic reloading on code changes:
    ```
        uvicorn app:churn_predictor_app --reload
    ```

3. **Run Production server (Gunicorn + Uvicorn):** For real deployment, use Gunicorn as a process manager with Uvicorn workers:
    ```
        gunicorn -w 4 -k uviocrn.workers.UvicornWorker app:churn_predictor_app
    ```

### B. Flask
Flask is used for simpler, synchronous WSGI applications. This scenario assumes you have a file named `App.py`.

1. **Install Dependencies:** 
    ```
        pip install -r requirements.txt
    ```

2. **Run Development Server:** Use this command during local development as it includes automatic reloading on code changes:
    ```
        python App.py
    ```

3. **Run Production server (Gunicorn):** For real deployment, use Gunicorn with the Flask application object:
    ```
        gunicorn --bind 0.0.0.0:8000 App:app
    ```