import streamlit as st
import pandas as pd
import requests

class DataClient:
    """Cliente para manejar la carga de datos desde la API FastAPI"""

    def __init__(self, api_url: str):
        self.api_url = api_url

    def get_data(self):
        """Obtiene los datos de diabetes desde la API"""
        try:
            response = requests.get(f"{self.api_url}/data", timeout=10)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = response.json().get("detail", "Error desconocido")
                return {"success": False, "error": error_detail}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}


@st.cache_data
def load_data():
    """
    Carga los datos desde la API de FastAPI.
    Se cachea para evitar recargas innecesarias.
    """
    client = DataClient(api_url="https://fastapi-diabetes.onrender.com")  # Cambia por la URL real
    result = client.get_data()

    if not result["success"]:
        st.error(f"❌ Error al cargar datos: {result['error']}")
        st.info("Verifica que la API FastAPI esté en ejecución.")
        return pd.DataFrame()

    df = pd.DataFrame(result["data"])
    return df
