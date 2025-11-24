import streamlit as st
import pandas as pd
import requests

class DataClient:
    """Cliente para manejar la carga de datos desde la API FastAPI"""

    def __init__(self, api_url: str):
        # Asegurar que la URL SIEMPRE termine en '/'
        if not api_url.endswith("/"):
            api_url += "/"
        self.api_url = api_url
        
    @st.cache_data
    def get_data(self):
        """Obtiene los datos de diabetes desde la API"""
        try:
            response = requests.get(self.api_url + "data", timeout=50)
            response.raise_for_status()  # Detecta errores HTTP
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Error de conexión: {str(e)}"}


@st.cache_data
def load_data():
    client = DataClient(api_url="https://fastapi-diabetes-znau.onrender.com")
    result = client.get_data()
    if not result["success"]:
        st.error(f"❌ Error en la carga datos: {result['error']}")
        return pd.DataFrame()
    return pd.DataFrame(result["data"])
