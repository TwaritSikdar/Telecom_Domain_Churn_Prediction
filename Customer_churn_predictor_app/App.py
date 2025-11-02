# Importing the important libraries
import joblib, json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys

# Importing the prerequisites
try:
    print('\n\n[INFO] Loading model, preprocessors and feature names......')

    # 1. Loading the model
    PREDICTOR = joblib.load('Resources/Customer_Churn_Telecom_Domain.pkl')

    # 2. Loading the encoder
    ONE_HOT_ENCODER = joblib.load('Resources/One_Hot_Encoder.pkl')

    # 3. Loading the scaler
    STANDARD_SCALER = joblib.load('Resources/Standard_Scaler.pkl')

    # 4. Loading the column names
    with open('Resources/Feature_names.json', 'r') as f1:
        FEATURE_NAMES = json.load(f1)
        print('\nFeatures names loaded:', FEATURE_NAMES)

    # 5. Loading the column names to be encoded
    with open('Resources/Columns_to_encode.json', 'r') as f2:
        COLUMNS_TO_ENCODE = json.load(f2)
        print('\nColumns to encode:', COLUMNS_TO_ENCODE)

    # 6. Setting up numerical features and Features during input
    NUM_FEATURES = STANDARD_SCALER.feature_names_in_.tolist()
    print('\n', NUM_FEATURES)
    INPUT_FEATURES = COLUMNS_TO_ENCODE + NUM_FEATURES
    print('\n', INPUT_FEATURES)

    print('\n[INFO] All files loaded successfully.')

except FileNotFoundError as fnfe:
    print(f'\n[ERROR] Missing the files: {fnfe}')
    print('Ensure all the files are in the directory.')
    sys.exit(1)

except json.JSONDecodeError as jde:
    print(f'\n[ERROR] Could not load the files Feature_names.json and Columns_to_encode.json. Please check the file format.')
    sys.exit(1)

# Initialising the app
app = Flask(__name__)
CORS(app)

# Preprocessing and Prediction Logic
def preprocess_and_predict(raw_input_df: pd.DataFrame):

    # Separating the data
    cat_data = raw_input_df[COLUMNS_TO_ENCODE]
    num_data = raw_input_df[NUM_FEATURES]

    # Encoding the data
    encoded_data = pd.DataFrame(
        ONE_HOT_ENCODER.transform(cat_data),
        index=raw_input_df.index,
        columns=ONE_HOT_ENCODER.get_feature_names_out()
    )

    # Scaling the data
    scaled_data = pd.DataFrame(
        STANDARD_SCALER.transform(num_data),
        index=raw_input_df.index,
        columns=STANDARD_SCALER.get_feature_names_out()
    )

    # Dropping scaled SeniorCitizen column as it is categorical data
    scaled_data = scaled_data.drop('SeniorCitizen', axis=1)

    # Dealing with overlapping features if any
    overlap = list(set(scaled_data.columns) & set(encoded_data.columns))
    if overlap:
        app.logger.warning(f"[Warning] Overlapping features detected in scaled and encoded data: {overlap}. Dropping from encoded data")
        encoded_data = encoded_data.drop(columns=overlap, errors='ignore')

    # Combining the data
    preprocessed_data = pd.concat(
        [scaled_data,
         raw_input_df['SeniorCitizen'],
         encoded_data],
         axis=1
    )
    
    # Rearranging the data
    preprocessed_data = preprocessed_data.reindex(columns=FEATURE_NAMES, fill_value=0)

    # Prediction
    churn_prediction = PREDICTOR.predict(preprocessed_data)[0]

    return churn_prediction

# Rendering it to frontend
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def response():
    try:
        data = request.get_json()

        # Validating the input contains expected features
        if not data or not all(col in data for col in INPUT_FEATURES):
            return jsonify({
                "error": "Missing or malformed input data.",
                "required_features": INPUT_FEATURES
            }), 400
        
        # Converting json data into a pandas DataFrame
        input_df = pd.DataFrame([data])

        # Prediction: Passing it to the function preprocess_and_predict
        prediction_label = preprocess_and_predict(input_df)
        final_probability = 1.0 if prediction_label == 1 else 0.0

        # Returning the final output
        return jsonify({
            "churn_probability": round(final_probability, 4),
            "prediction": int(prediction_label),
            "model": "Hyperparameter tuned Stacking Classifier (Probability not supported)"
        })

    except Exception as e:
        app.logger.error(f'Prediction error: {e}', exc_info=True)
        return jsonify({
            "error": "An internal error occured during prediction.",
            "details": str(e)
        }), 500

# Running the app
if __name__ == '__main__':
    print('\n--- Starting the Flask App ---')
    app.run(debug=True)