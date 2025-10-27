from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = FastAPI(title="API Predicción Diabetes")

# Conexión a Neon (usa variable de entorno en Render)
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Cargar datos desde Neon
def cargar_datos():
    query = "SELECT * FROM diabetes"
    return pd.read_sql(query, engine)

# Preprocesamiento y entrenamiento (puedes hacerlo offline y guardar el modelo)
def entrenar_modelo():
    datos = cargar_datos()
    num_cols = ['age', 'bmi', 'hba1c_level', 'blood_glucose_level']
    scaler = StandardScaler()
    datos[num_cols] = scaler.fit_transform(datos[num_cols])
    X = pd.get_dummies(datos.drop(columns='diabetes'), columns=['gender', 'smoking_history'], drop_first=True)
    y = datos['diabetes']
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)
    joblib.dump(model, "modelo_diabetes.pkl")
    return {"message": "Modelo entrenado y guardado"}

# Esquema para predicciones
class DiabetesInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    hba1c_level: float
    blood_glucose_level: float

@app.get("/")
def home():
    return {"message": "API para predicción de diabetes activa"}

@app.get("/data")
def get_data(limit: int = 10):
    df = cargar_datos().head(limit)
    return df.to_dict(orient="records")

@app.post("/predict")
def predict(data: DiabetesInput):
    model = joblib.load("modelo_diabetes.pkl")
    df = pd.DataFrame([data.dict()])
    df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)
    # Ajustar columnas faltantes según entrenamiento
    prediction = model.predict(df)[0]
    return {"diabetes_prediction": int(prediction)}