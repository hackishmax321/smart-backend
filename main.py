# main.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
import shutil
from typing import List, Optional, Dict
from firestore_db import get_firestore_client
import joblib
import pandas as pd
import numpy as np
import traceback
import re
from autocorrect import Speller
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from google.cloud import vision
from google.oauth2 import service_account
import os

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:3001"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load MOdels Health
MODEL_DIABETIS = joblib.load('diabetes_model_v2.joblib')
MODEL_ANEMIA = joblib.load('anemia_model_v2.joblib')
MODEL_CARDIO = joblib.load('cardiovascular_model_v2.joblib')

# MODEL_EXE = joblib.load('decision_tree_regressor_model.joblib')

gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
smoking_history_mapping = {'No Info': 0, 'current': 1, 'ever': 2, 'former': 3, 'never': 4, 'not current': 5}

credentials = service_account.Credentials.from_service_account_file('google-ocr-key.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

# Db connection
# db = get_firestore_client()


class User(BaseModel):
    username: str
    full_name: str
    email:str
    contact: str
    password: str
    nic: str

class LoginUser(BaseModel):
    username: str
    password: str


users_db = {}

# @app.post("/register")
# async def register_user(user: User):
#     user_ref = db.collection("users").document(user.username)
#     if user_ref.get().exists:
#         raise HTTPException(status_code=400, detail="Username already registered")

#     # Hash the password before storing it
#     hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
#     user_data = user.dict()
#     user_data["password"] = hashed_password.decode('utf-8')

#     user_ref.set(user_data)
#     return {"message": "User registered successfully", "user": user_data}

# @app.post("/login")
# async def login_user(user: LoginUser):
#     user_ref = db.collection("users").document(user.username)
#     user_doc = user_ref.get()

#     if not user_doc.exists:
#         raise HTTPException(status_code=400, detail="Invalid username or password")

#     user_data = user_doc.to_dict()
    
#     # Check the hashed password
#     if not bcrypt.checkpw(user.password.encode('utf-8'), user_data["password"].encode('utf-8')):
#         raise HTTPException(status_code=400, detail="Invalid username or password")

#     user_data.pop("password")  # Remove the password field from the response

#     return {"message": "Login successful", "user": user_data}





@app.post("/extract-blood-report-image")
async def upload_images(files: list[UploadFile] = File(...)):  # Accepting multiple files
    try:
        extracted_data = []
        full_text = ""

        # Process each uploaded file
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Load the image into memory
            with open(file_path, "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            # Perform text detection using Google Vision OCR
            response = client.text_detection(image=image)
            texts = response.text_annotations

            if response.error.message:
                raise Exception(f'{response.error.message}')

            # Get the recognized text
            recognized_text = texts[0].description.replace("\n", " ") if texts else ""

            # print(recognized_text)

            # Auto-correct the extracted text
            # corrected_text = auto_correct_text(recognized_text)

            # Parse the corrected text
            full_text = full_text + " " +recognized_text

        properties = parse_blood_report(full_text)
        properties2 = parse_personalData(full_text)
        # print(full_text)
        print(properties2)   
        print(full_text)
        combined_properties = {**properties, **properties2}
        return {"properties": combined_properties}

    except Exception as e:
        return {"error": f"Error: {str(e)}"}


def auto_correct_text(text: str) -> str:
    """
    Auto-correct the extracted text using a spell checker.
    """
    spell = Speller(lang="en")
    corrected_lines = [spell(line) for line in text.splitlines()]
    return "\n".join(corrected_lines)



def parse_blood_report(text: str):
    """
    Parse the extracted text to find specific health-related properties such as hemoglobin, glucose, cholesterol, etc.
    """
    properties = {}

    # Define keywords and patterns for health metrics
    metric_keywords = {
        "Age": ["Age", "age"],
        "Hemoglobin": ["haemoglobin", "hemoglobin", "HAEKOGLOBM", "hemo", "Hemo", "HGB", "hgb"],
        "MCHC": ["MCHC", "mchc"],
        "MCH": ["MCH", "mch"],
        "MCV": ["MCV", "mcv"],
        "HCT": ["HCT", "hct"],
        "WhiteCellCount": ["total white cell count", "wbc", "WBC"],
        "RedCellCount": ["total white cell count", "rbc", "RBC"],
        "PlateletCount": ["platelet count", "platelets"],
        "Glucose": ["glucose", "sugar", "GLUCOSE"],
        "Cholesterol": ["cholesterol", "CHOLESTEROL"],
        "Triglycerides": ["triglycerides"],
        "HDL": ["hdl cholesterol", "hdl"],
        "LDL": ["ldl cholesterol", "ldl"],
        "PLT": ["PLT", "plt"],
        "MPV": ["mpv", "MPV"],
        "WBC": ["WBC", "wbc"],
        "TRIGLYCERIDES": ["TRIGLYCERIDES"]

    }

    lines = text.splitlines()

    for line in lines:
        line_lower = line.lower()
        for metric, keywords in metric_keywords.items():
            for keyword in keywords:
                match = re.search(rf"\b{re.escape(keyword)}\b\s*(\d+(\.\d+)?)", line_lower)
                if match:
                    value = float(match.group(1))  # Extract numeric value
                    properties[metric] = value
                    break  # Move to the next line after finding the metric

    return properties

import re

def parse_personalData(text: str):
    properties = {}

    # Define patterns for health metrics
    metric_patterns = {
        "Age": r"(\d+)\s*Y", 
        # "Age": r"(\d+)\s*Y", 
    }

    lines = text.splitlines()

    for line in lines:
        for metric, pattern in metric_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                # print(match)
                value = int(match.group(1))  # Extract the numeric value
                properties[metric] = value
                break  # Move to the next line after finding the metric

    return properties

# Example Usage
text = "John is 43Y old and his height is 175cm."
result = parse_personalData(text)
print(result)  # Output: {'Age': 43}


def extract_numeric_value(text: str):
    """
    Extract the first numeric value from a line of text.
    Handles both integers and decimals.
    """
    match = re.search(r"\d+\.?\d*", text)
    return float(match.group()) if match else None   


class HealthPredictionRequest(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    hemoglobin: float
    mch: float
    mchc: float
    mcv: float
    cholesterol: float
    triglyceride: float
    hdl: float
    ldl: float

def predict_cardiovascular_with_probability(age, gender, bmi, cholesterol, triglyceride, hdl, ldl):
    """
    Predicts whether a patient has cardiovascular disease or not and provides the probability of each outcome.

    Parameters:
    - age (float): Age of the patient.
    - gender (str): Gender ("Male", "Female", or "Other").
    - bmi (float): Body Mass Index (BMI).
    - cholesterol (float): Cholesterol level.
    - triglyceride (float): Triglyceride level.
    - hdl (float): HDL level.
    - ldl (float): LDL level.

    Returns:
    - Prediction (str): "No Cardiovascular Disease" or "Cardiovascular Disease".
    - Probability (float): Probability percentage for the predicted class.
    """
    # Gender encoding
    gender_encoded = gender_mapping.get(gender, -1)
    if gender_encoded == -1:
        raise ValueError("Invalid gender or smoking history value")

    # Prepare input as a 2D array for the model
    input_data = np.array([[age, gender_encoded, bmi, cholesterol, triglyceride, hdl, ldl]])

    # Make prediction and get probabilities
    prediction = MODEL_CARDIO.predict(input_data)[0]
    probabilities = MODEL_CARDIO.predict_proba(input_data)[0]

    # Map prediction to label
    prediction_label = "No Cardiovascular Disease" if prediction == 0 else "Cardiovascular Disease"

    # Extract the probability of the predicted class
    predicted_probability = round(max(probabilities)*100,2)

    return prediction_label, predicted_probability


def predict_diabetes_with_probability(gender, age, smoking_history, bmi, hbA1c_level, blood_glucose_level):
    """
    Predicts whether a patient has diabetes or not and provides the probability of each outcome.

    Parameters:
    - gender (str): Gender ("Male", "Female", or "Other").
    - age (float): Age of the patient.
    - smoking_history (str): Smoking history ("never", "No Info", "current", "former", "ever", "not current").
    - bmi (float): Body Mass Index (BMI).
    - hbA1c_level (float): HbA1c level.
    - blood_glucose_level (float): Blood glucose level.

    Returns:
    - Prediction (str): "No Diabetes" or "Diabetes".
    - Probability (float): Probability percentage for the predicted class.
    """
    gender_encoded = gender_mapping.get(gender, -1)
    smoking_encoded = smoking_history_mapping.get(smoking_history, -1)

    print(gender)
    
    if gender_encoded == -1 or smoking_encoded == -1:
        raise ValueError("Invalid gender or smoking history value")

    # Prepare input as a 2D array for the model
    input_data = np.array([[gender_encoded, age, smoking_encoded, bmi, hbA1c_level, blood_glucose_level]])

    # Make prediction and get probabilities
    prediction = MODEL_DIABETIS.predict(input_data)[0]
    probabilities = MODEL_DIABETIS.predict_proba(input_data)[0]

    # Map prediction to label
    prediction_label = "No Diabetes" if prediction == 0 else "Diabetes"

    # Extract the probability of the predicted class
    predicted_probability = round(max(probabilities)*100,2)

    return prediction_label, predicted_probability

# Function to predict anemia
def predict_anemia_with_probability(hemoglobin, mch, mchc, mcv, gender):
    """
    Predicts whether a patient has anemia or not and provides the probability of each outcome.

    Parameters:
    - hemoglobin (float): Hemoglobin level.
    - mch (float): Mean Corpuscular Hemoglobin.
    - mchc (float): Mean Corpuscular Hemoglobin Concentration.
    - mcv (float): Mean Corpuscular Volume.
    - gender (str): Gender ("Male" or "Female").

    Returns:
    - prediction (str): "Not Anemia" or "Anemia".
    - probability (float): Probability percentage for the predicted class.
    """
    # Ensure gender is encoded correctly
    gender_encoded = gender_mapping.get(gender, -1)
    
    if gender_encoded == -1:
        raise ValueError("Invalid gender value")

    # Prepare input as a 2D array for the model
    input_data = np.array([[hemoglobin, mch, mchc, mcv, gender_encoded]])
    print(input_data)

    # Make prediction and get probabilities using the pre-initialized model
    prediction = MODEL_ANEMIA.predict(input_data)[0]
    probabilities = MODEL_ANEMIA.predict_proba(input_data)[0]  # Get the probabilities for the first (and only) row
    print(probabilities)

    # Extract the probability of the predicted class
    predicted_probability = round(max(probabilities)*100,2)

    # Map prediction to a human-readable label
    # prediction_label = "Not Anemia" if prediction == 0 else "Anemia"

    return prediction, predicted_probability




@app.post("/predict_health_issues_all")
async def predict_health_issues(request: HealthPredictionRequest):
    try:
        # Diabetes prediction
        diabetes_result, diabetes_prob = predict_diabetes_with_probability(
            gender=request.gender,
            age=request.age,
            # hypertension=request.hypertension,
            # heart_disease=request.heart_disease,
            smoking_history=request.smoking_history,
            bmi=request.bmi,
            hbA1c_level=request.HbA1c_level,
            blood_glucose_level=request.blood_glucose_level
        )

        # Anemia prediction
        anemia_result, anemia_prob = predict_anemia_with_probability(
            gender=request.gender,
            hemoglobin=request.hemoglobin,
            mch=request.mch,
            mchc=request.mchc,
            mcv=request.mcv
        )

        # Cardio
        cardio_result, cardio_prob = predict_cardiovascular_with_probability(
            age=request.age,
            gender=request.gender,
            bmi=request.bmi,
            cholesterol=request.cholesterol,
            triglyceride=request.triglyceride,
            hdl=request.hdl,
            ldl=request.ldl
        )

        # Return combined result
        return {
            "diabetes_result": diabetes_result,
            "diabetes_probability": diabetes_prob,
            "anemia_result": anemia_result,
            "anemia_probability": anemia_prob,
            "cardio_result": cardio_result,
            "cardio_probability": cardio_prob
        }

    except ValueError as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

# Diet Planning
def scaling(dataframe):
    scaler = StandardScaler()
    prep_data = scaler.fit_transform(dataframe.iloc[:, 6:15].to_numpy())
    return prep_data, scaler

def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine', algorithm='brute')
    neigh.fit(prep_data)
    return neigh

def build_pipeline(neigh, scaler, params):
    transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
    pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])
    return pipeline

def extract_data(dataframe, ingredient_filter, max_nutritional_values):
    extracted_data = dataframe.copy()
    for column, maximum in zip(extracted_data.columns[6:15], max_nutritional_values.values()):
        extracted_data = extracted_data[extracted_data[column] < maximum]
    if ingredient_filter is not None:
        for ingredient in ingredient_filter:
            extracted_data = extracted_data[extracted_data['RecipeIngredientParts'].str.contains(ingredient, regex=False)]
    return extracted_data

def apply_pipeline(pipeline, _input, extracted_data):
    return extracted_data.iloc[pipeline.transform(_input)[0]]

def recommand(dataframe, _input, max_nutritional_values, ingredient_filter=None, params={'return_distance': False}):
    extracted_data = extract_data(dataframe, ingredient_filter, max_nutritional_values)
    prep_data, scaler = scaling(extracted_data)
    neigh = nn_predictor(prep_data)
    pipeline = build_pipeline(neigh, scaler, params)
    return apply_pipeline(pipeline, _input, extracted_data)


# Input model for the API
class RecommendationRequest(BaseModel):
    max_daily_fat: float
    max_nutritional_values: Dict[str, float]
    ingredient_filter: Optional[List[str]] = None

# Load the dataset
try:
    dataset = pd.read_csv('diet_dataset/dataset.csv')  # Assuming the dataset is in CSV format
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {str(e)}")

# API endpoint
@app.post("/recommend_recipe")
async def recommend_recipe(request: RecommendationRequest):
    try:
        # Extract parameters from request
        max_daily_fat = request.max_daily_fat
        max_nutritional_values = request.max_nutritional_values
        ingredient_filter = request.ingredient_filter

        # Prepare input vector
        test_input = np.array([[0] * 9])  # Assuming the input shape is (1, 9) for nutritional features
        test_input[0, 1] = max_daily_fat  # Set the daily fat in the input

        # Generate a recommendation
        recommended_recipe = recommand(
            dataframe=dataset,
            _input=test_input,
            max_nutritional_values=max_nutritional_values,
            ingredient_filter=ingredient_filter
        )

        # Drop unnecessary columns
        recommended_recipe = recommended_recipe.drop(
            columns=["RecipeId", "CookTime", "PrepTime", "TotalTime"], errors="ignore"
        )

        # Return the recommendation as a JSON response
        return recommended_recipe.to_dict(orient="records")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recipe recommendation: {str(e)}")


# Excercise Suggetion

EXE_MODEL =  joblib.load('decision_tree_regressor_model.joblib')

# Mapping Gender and Workout_Type
gender_mapping = {'Female': 0, 'Male': 1}
workout_type_mapping = {'Cardio': 0, 'HIIT': 1, 'Strength': 2, 'Yoga': 3}

# Define input data schema using Pydantic
class WorkoutData(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    max_bpm: int
    avg_bpm: int
    resting_bpm: int
    session_duration: float
    calories_burned: float
    fat_percentage: float
    water_intake: float
    workout_frequency: int
    experience_level: int
    bmi: float

# Prediction function
def predict_workout_type(data: WorkoutData) -> str:
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'Age': [data.age],
        'Gender': [gender_mapping.get(data.gender)],  # Convert gender to numeric
        'Weight (kg)': [data.weight],
        'Height (m)': [data.height],
        'Max_BPM': [data.max_bpm],
        'Avg_BPM': [data.avg_bpm],
        'Resting_BPM': [data.resting_bpm],
        'Session_Duration (hours)': [data.session_duration],
        'Calories_Burned': [data.calories_burned],
        'Fat_Percentage': [data.fat_percentage],
        'Water_Intake (liters)': [data.water_intake],
        'Workout_Frequency (days/week)': [data.workout_frequency],
        'Experience_Level': [data.experience_level],
        'BMI': [data.bmi]
    })
    
    # Predict the workout type (returns a numeric value)
    workout_type_numeric = EXE_MODEL.predict(input_data)
    
    # Convert numeric result back to workout type label
    workout_type_label = [key for key, value in workout_type_mapping.items() if value == round(workout_type_numeric[0])][0]
    
    return workout_type_label

# Define an API endpoint to predict workout type
@app.post("/predict-workout-type")
def predict_workout(data: WorkoutData):
    predicted_workout_type = predict_workout_type(data)
    return {"predicted_workout_type": predicted_workout_type}



# Stress Management 
MODEL_STRESS = joblib.load('logistic_regression_stress_model.joblib')
# Pydantic model for input validation
class StressPredictionRequest(BaseModel):
    snoring_range: float
    respiration_rate: float
    body_temperature: float
    limb_movement_rate: float
    blood_oxygen_level: float
    eye_movement: float
    hours_of_sleep: float
    heart_rate: float

stress_level_labels = {
    0: 'low/normal',        # 0 - low/normal
    1: 'medium low',        # 1 - medium low
    2: 'medium',            # 2 - medium
    3: 'medium high',       # 3 - medium high
    4: 'high'               # 4 - high
}

def predict_stress_level(model, snoring_range, respiration_rate, body_temperature,
                         limb_movement_rate, blood_oxygen_level, eye_movement,
                         hours_of_sleep, heart_rate):
    """
    Predict the stress level (0-4) based on user parameters.

    Returns:
    - Predicted stress level label (str)
    """
    # Create an array of input features
    input_features = np.array([[snoring_range, respiration_rate, body_temperature,
                                limb_movement_rate, blood_oxygen_level, eye_movement,
                                hours_of_sleep, heart_rate]])

    # Predict the class (stress level) using the model
    predicted_stress_level = model.predict(input_features)[0]

    # Map predicted stress level to the corresponding label
    predicted_stress_level_label = stress_level_labels.get(predicted_stress_level, 'Unknown')

    # Return the predicted stress level label
    return predicted_stress_level_label

# FastAPI endpoint to predict stress level
@app.post("/predict_stress_level")
def predict_stress_level_endpoint(request: StressPredictionRequest):
    # Use the provided data to make the prediction
    predicted_stress_level = predict_stress_level(
        model=MODEL_STRESS,  
        snoring_range=request.snoring_range,
        respiration_rate=request.respiration_rate,
        body_temperature=request.body_temperature,
        limb_movement_rate=request.limb_movement_rate,
        blood_oxygen_level=request.blood_oxygen_level,
        eye_movement=request.eye_movement,
        hours_of_sleep=request.hours_of_sleep,
        heart_rate=request.heart_rate
    )
    
    return {"predicted_stress_level": predicted_stress_level}



# Excercise 
model = joblib.load('decision_tree_regressor_exercise_model.joblib')  
# Gender and workout type mappings
gender_mapping = {'Female': 0, 'Male': 1}
workout_type_mapping = {'Cardio': 0, 'HIIT': 1, 'Strength': 2, 'Yoga': 3}

# Function to predict workout type based on input data
def predict_workout_type(age, gender, weight, height, max_bpm, avg_bpm, resting_bpm, session_duration,
                          calories_burned, fat_percentage, water_intake, workout_frequency, experience_level, bmi):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_mapping.get(gender)],  # Convert gender to numeric
        'Weight (kg)': [weight],
        'Height (m)': [height],
        'Max_BPM': [max_bpm],
        'Avg_BPM': [avg_bpm],
        'Resting_BPM': [resting_bpm],
        'Session_Duration (hours)': [session_duration],
        'Calories_Burned': [calories_burned],
        'Fat_Percentage': [fat_percentage],
        'Water_Intake (liters)': [water_intake],
        'Workout_Frequency (days/week)': [workout_frequency],
        'Experience_Level': [experience_level],
        'BMI': [bmi]
    })

    # Predict the workout type (returns a numeric value)
    workout_type_numeric = model.predict(input_data)

    # Convert numeric result back to workout type label
    workout_type_label = [key for key, value in workout_type_mapping.items() if value == round(workout_type_numeric[0])][0]

    return workout_type_label


# Define request body using Pydantic models
class WorkoutData(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    max_bpm: int
    avg_bpm: int
    resting_bpm: int
    session_duration: float
    calories_burned: float
    fat_percentage: float
    water_intake: float
    workout_frequency: int
    experience_level: int
    bmi: float

# Define the FastAPI endpoint
@app.post("/predict_workout")
async def predict_workout(data: WorkoutData):
    # Call the prediction function
    workout_type = predict_workout_type(
        age=data.age,
        gender=data.gender,
        weight=data.weight,
        height=data.height,
        max_bpm=data.max_bpm,
        avg_bpm=data.avg_bpm,
        resting_bpm=data.resting_bpm,
        session_duration=data.session_duration,
        calories_burned=data.calories_burned,
        fat_percentage=data.fat_percentage,
        water_intake=data.water_intake,
        workout_frequency=data.workout_frequency,
        experience_level=data.experience_level,
        bmi=data.bmi
    )
    return {"predicted_workout_type": workout_type}