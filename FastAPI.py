from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import create_engine, text
import pandas as pd
import os
import logging
import snowflake.connector

app = FastAPI(title="API de Datos de Diabetes")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_BDG2IiT0aqAy@ep-super-heart-agp17yzq-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require"
)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no definido correctamente")

engine = create_engine(DATABASE_URL)

# Snowflake: usa variables de entorno
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")

logging.basicConfig(level=logging.INFO)

def insert_into_snowflake(data: dict):
    """Inserta un registro en Snowflake; levantar excepción si algo falla."""
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
    )
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO diabetes (
                    gender, age, hypertension, heart_disease, smoking_history,
                    bmi, hba1c_level, blood_glucose_level, diabetes
                )
                VALUES (%(gender)s, %(age)s, %(hypertension)s, %(heart_disease)s, %(smoking_history)s,
                        %(bmi)s, %(hba1c_level)s, %(blood_glucose_level)s, %(diabetes)s)
            """, data)
    finally:
        conn.close()

@app.post("/insert")
async def insert_data(request: Request):
    try:
        data = await request.json()
        # 1) Inserta en PostgreSQL (transacción segura)
        query = text("""
            INSERT INTO diabetes (
                gender, age, hypertension, heart_disease, smoking_history,
                bmi, hba1c_level, blood_glucose_level, diabetes
            )
            VALUES (:gender, :age, :hypertension, :heart_disease, :smoking_history,
                    :bmi, :hba1c_level, :blood_glucose_level, :diabetes)
        """)
        with engine.begin() as conn:
            conn.execute(query, data)

        # 2) Intenta replicar a Snowflake
        try:
            insert_into_snowflake(data)
            return {"message": "✅ Insertado en PostgreSQL y Snowflake"}
        except Exception as e_sf:
            # No romper el flujo: PostgreSQL ya guardó el dato.
            logging.error(f"Replica en Snowflake falló: {e_sf}")
            # Puedes decidir:
            # - devolver 207 Multi-Status (no estándar en FastAPI),
            # - o incluir un aviso en el mensaje:
            return {
                "message": "⚠️ Insertado en PostgreSQL, pero falló la replicación en Snowflake",
                "snowflake_error": str(e_sf)
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al insertar datos: {e}")


@app.get("/")
def root():
    return {"message": "API activate"}

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
        with engine.begin() as conn:  # ✅ Manejo seguro de transacción
            conn.execute(query, data)
        return {"message": "Datos insertados correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al insertar datos: {e}")
    
@app.post("/predict")
async def predict_diabetes(request: Request):
    data = await request.json()
    prediction = 1 if data["hba1c_level"] > 6.5 or data["blood_glucose_level"] > 140 else 0
    return {"diabetes_prediction": prediction}


@app.get("/get_roles")
def get_roles(usuario: str):
    query = text("""
        SELECT r.nombre FROM roles r
        JOIN usuario_rol ur ON r.id = ur.rol_id
        JOIN usuarios u ON u.id = ur.usuario_id
        WHERE u.nombre = :usuario
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"usuario": usuario}).fetchall()
    return {"roles": [row[0] for row in result]}
