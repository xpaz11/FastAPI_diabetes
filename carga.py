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
            response = requests.get(f"{self.api_url}/data", timeout=100)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = response.json().get("detail", "Error desconocido")
                return {"success": False, "error": error_detail}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}


@st.cache_data
def load_data():
    client = DataClient(api_url="https://fastapi-diabetes.onrender.com/data")
    result = client.get_data()
    if not result["success"]:
        st.error(f"❌ Error al cargar datos: {result['error']}")
        return pd.DataFrame()
    return pd.DataFrame(result["data"])
