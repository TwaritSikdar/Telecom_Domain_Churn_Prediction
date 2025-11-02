# Importing the required libraries
import uvicorn
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from schema import CustomerInfo
import json
import sys, os, traceback
from contextlib import asynccontextmanager
import pandas as pd

# Setting up varibles MODEL, PREPROCESSOR AND FEATURE NAMES
PREDICTOR = None
ONE_HOT_ENCODER = None,
STANDARD_SCALER = None,
FEATURES_IN_MODEL = [],
COLUMNS_TO_ENCODE = [],
NUMERICAL_FEATURES = [],
INPUT_FEATURES = []

# Directory names for Templates/static files
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"

# Initilaizing the templates object
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@asynccontextmanager
async def lifespan_handler(churn_predictor_app: FastAPI):
    """
    Handles application shutdown and startup events.
    Loads all necessary joblib and json files for the prediction pipeline on startup.
    """

    global PREDICTOR, ONE_HOT_ENCODER, STANDARD_SCALER, FEATURES_IN_MODEL, COLUMNS_TO_ENCODE, NUMERICAL_FEATURES, INPUT_FEATURES

    try:
        print('\n\n [INFO] Loading model, preprocessors and feature names.........')

        # 1. Loading the model
        PREDICTOR = joblib.load('Resources/Customer_Churn_Telecom_Domain.pkl')

        # 2. Loading the encoder
        ONE_HOT_ENCODER = joblib.load('Resources/One_Hot_Encoder.pkl')

        # 3. Loading the scaler
        STANDARD_SCALER = joblib.load('Resources/Standard_Scaler.pkl')

        # 4. Loading the column names
        with open('Resources/Feature_names.json', 'r') as f1:
            FEATURES_IN_MODEL = json.load(f1)
            print('\n[INFO] %s Feature names loaded:' %len(FEATURES_IN_MODEL), FEATURES_IN_MODEL)

        # 5. Loading the column names to be encoded
        with open('Resources/Columns_to_encode.json', 'r') as f2:
            COLUMNS_TO_ENCODE = json.load(f2)
            print('\n[INFO] %s Columns to encode:'%len(COLUMNS_TO_ENCODE), COLUMNS_TO_ENCODE)

        # 6. Setting up numerical features
        try:
            NUMERICAL_FEATURES = STANDARD_SCALER.feature_names_in_.tolist()
        except AttributeError as ae:
            print('\n [WARNING] Scaler does not expose feature_names_in_. Assuming the standard numerical features.')
            NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
            
        print('\n [INFO] %s numerical features loaded:' %len(NUMERICAL_FEATURES), NUMERICAL_FEATURES)

        # 7. Setting up input features
        INPUT_FEATURES = NUMERICAL_FEATURES + COLUMNS_TO_ENCODE
        print('\n [INFO] Expected input features (%s):' %len(INPUT_FEATURES), INPUT_FEATURES)

    except FileNotFoundError as fnfe:
        print(f"\n [FATAL ERROR] Missing deployment file: {fnfe}. Please ensure directory and file names are correct and also are present in the location.")
        sys.exit(1)
    except json.JSONDecodeError as jsde:
        print(f"\n [FATAL ERROR] Could not decode Feature_names.json or Columns_to_encode.json. Check the file format.")
        sys.exit(1)
    except Exception as e:
        print(f"\n [FATAL ERROR] Error during resource loading: {e}")
        sys.exit(1)

    # To pause execution here until the app shuts down
    yield

    print("[INFO] Application Shutdown complete.")

# Initialising the app
churn_predictor_app = FastAPI(
    title="Telecom Churn Predictor API",
    description="Deploying a hyperparmeter tuned Stacking Classifier Model",
    lifespan=lifespan_handler,
    version="1.0.0"
)

# Enabling CORS for all origins, methods and headers
churn_predictor_app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# Mounting the static directory to serve the frontend files
churn_predictor_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Defining core prediction logic
def preprocess_and_predict(raw_input_data: CustomerInfo):
    """
    Takes raw customer data as input, converts it into a pandas DataFrame and
    applies encoding and scaling for prediction.
    """

    # Converting pydantic object into a dictionary and then into a dataframe
    raw_dict = raw_input_data.model_dump()
    raw_data = pd.DataFrame([raw_dict])

    # DEBUG: Print raw input columns
    print(f"\n [DEBUG] Raw Input Columns: {raw_data.columns.tolist()}", file=sys.stderr)

    # Seperating the data
    try:
        cat_data = raw_data[COLUMNS_TO_ENCODE]
        num_data = raw_data[NUMERICAL_FEATURES]
    except KeyError as ke:
        raise ValueError(f"Input data is missing a feature required by the model: {ke}. Please check if the 'Rsources' files are correct.")

    # Encoding the categorical data
    print(f"[DEBUG] cat_data columns: {cat_data.columns.tolist()}", file=sys.stderr)
    encoded_array = ONE_HOT_ENCODER.transform(cat_data)
    encoded_data = pd.DataFrame(
        encoded_array,
        index=raw_data.index,
        columns=ONE_HOT_ENCODER.get_feature_names_out()
    )
    # DEBUG: Print encoded_data shape
    print(f"[DEBUG] encoded_data shape: {encoded_data.shape}", file=sys.stderr)

    # Scaling the numerical data
    print(f"[DEBUG] num_data columns: {num_data.columns.tolist()}", file=sys.stderr)
    scaled_array = STANDARD_SCALER.transform(num_data)
    scaled_data = pd.DataFrame(
       scaled_array,
        index=raw_data.index,
        columns=STANDARD_SCALER.get_feature_names_out()
    )
    # DEBUG: Print encoded_data shape
    print(f"[DEBUG] scaled_data shape: {scaled_data.shape}", file=sys.stderr)

    # Dropping scaled SeniorCitizen column as it is categorical column
    scaled_data = scaled_data.drop('SeniorCitizen', axis=1)

    # Dealing with overlapping columns
    overlap = list(set(scaled_data.columns) & set(encoded_data.columns))
    if overlap:
        print(f"[WARNING] Detected overlapping features in scaled and encoded data: {overlap}. Dropping from encoded data.", file=sys.stderr)
        encoded_data = encoded_data.drop(columns=overlap, errors='ignore')

    # Combining all the data
    preprocessed_data = pd.concat(
        [scaled_data,
        raw_data['SeniorCitizen'],
        encoded_data],
        axis=1
    )

    # Rearranging the data
    preprocessed_data = preprocessed_data.reindex(columns=FEATURES_IN_MODEL, fill_value=0)

    # Validate feature count before prediction
    if preprocessed_data.shape[1] != len(FEATURES_IN_MODEL):
        raise ValueError(f"Processed feature count ({preprocessed_data.shape[1]}) does not match expected model feature count ({len(FEATURES_IN_MODEL)}). Please check data consistency.")
    
    # DEBUG: Print final preprocessed shape
    print(f"[DEBUG] Final preprocessed data shape: {preprocessed_data.shape}", file=sys.stderr)
    
    # Prediction
    churn_prediction = PREDICTOR.predict(preprocessed_data)[0]

    return churn_prediction

# Defining the endpoints
@churn_predictor_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    """Serves the main frontend interface (index.html) using the Jinja2 templates."""
    return templates.TemplateResponse("index.html", {"request": request})

@churn_predictor_app.post("/predict", response_model=None)
async def predict_churn(request_data: CustomerInfo):
    """
    Handles POST requests for churn prediction.
    Validation is handled in CutomerInfo.
    """

    try:
        # Class label prediction
        prediction_label = preprocess_and_predict(request_data)

        # Since predict_proba is unavailable, we use the class label
        final_probability = 1.0 if prediction_label == 1 else 0.0

        return {
            "churn_probability": round(final_probability, 4),
            "prediction": int(prediction_label),
            "model": "Hyperparmeter tuned Stacking Classifier (Probability not supported)"
        }
    
    except ValueError as ve:
        # Catch errors related to dimension mismatch or invalid prediction
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        # Catch all other unexpected errors
        print(f"Prediction processing error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail="An internal server error occurred during prediction.")
    
if __name__=='__main__':
    uvicorn.run(churn_predictor_app)