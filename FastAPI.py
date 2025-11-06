from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import create_engine, text
import pandas as pd
import os

app = FastAPI(title="API de Datos de Diabetes")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_BDG2IiT0aqAy@ep-super-heart-agp17yzq-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require"
)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no definido correctamente")

engine = create_engine(DATABASE_URL)

@app.get("/")
def root():
    return {"message": "API activa"}

@app.get("/data")
def get_data():
    try:
        with engine.connect() as connection:
            df = pd.read_sql("SELECT * FROM diabetes", connection)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error carga de datos: {e}")

@app.post("/insert")
async def insert_data(request: Request):
    try:
        data = await request.json()
        query = text("""
            INSERT INTO diabetes (gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level, diabetes)
            VALUES (:gender, :age, :hypertension, :heart_disease, :smoking_history, :bmi, :hba1c_level, :blood_glucose_level, :diabetes)
        """)
        with engine.begin() as conn:
            conn.execute(query, data)  # ✅ Correcto
        return {"message": "Datos insertados correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al insertar datos: {e}")
    
@app.post("/predict")
async def predict_diabetes(request: Request):
    data = await request.json()
    prediction = 1 if data["hba1c_level"] > 6.5 or data["blood_glucose_level"] > 140 else 0
    return {"diabetes_prediction": prediction}