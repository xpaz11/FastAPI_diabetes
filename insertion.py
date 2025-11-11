import pandas as pd
from sqlalchemy import create_engine
import streamlit as st # Necesario para leer el secreto

# --- CONFIGURACIÓN ---
# 1. Tu ruta de archivo CSV (usamos el nombre que proporcionaste)
CSV_FILE_PATH = "C:\\Users\\xpaz\\Documents\\diabetes\\diabetes_prediction_dataset.csv"

# 2. El nombre de la tabla de destino en Neon
TABLE_NAME = "diabetes"

try:
    # 3. CONEXIÓN (Usando el secreto que ya configuraste)
    print("Conectando a Neon a través de st.secrets...")
    connection_string = st.secrets["connections"]["neon"]["url"]
    engine = create_engine(connection_string)

    # 4. CARGAR EL CSV LOCAL A UN DATAFRAME
    # Nota: El archivo se lee en tu memoria local.
    print(f"Leyendo CSV local desde: {CSV_FILE_PATH}")
    datos= pd.read_csv(CSV_FILE_PATH)

    # 5. AJUSTAR COLUMNAS
    # Opcional: Asegúrate de que los nombres de las columnas coincidan exactamente
    # con los nombres de la tabla que creaste (id, age, gender, etc.).
    datos.columns = datos.columns.str.lower().str.replace(' ', '_').str.replace(':', '').str.replace('-', '_')

    # 6. IMPORTAR A LA BASE DE DATOS
    # La clave es df.to_sql().
    # if_exists='replace' borra la tabla si existe y la recrea con los nuevos datos.
    # index=False evita crear una columna 'index' adicional.
    print(f"Importando {len(datos)} filas a la tabla '{TABLE_NAME}' en Neon...")
    datos.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)

    print("\n✅ ¡IMPORTACIÓN FINALIZADA CON ÉXITO!")
    print(f"La tabla '{TABLE_NAME}' ahora contiene {len(datos)} registros.")

except Exception as e:
    print(f"\n❌ ERROR durante el proceso de importación: {e}")