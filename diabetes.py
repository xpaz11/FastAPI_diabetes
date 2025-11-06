import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sqlalchemy import create_engine
import os
from carga import load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Cargar datos
datos = load_data()
API_URL = "https://fastapi-diabetes.onrender.com"

st.title("🔐 Acceso seguro")
st.text("Para comenzar, inicia sesión con tu usuario. Una vez dentro, podrás navegar entre las secciones desde el menú lateral.")
# ✅ Login
usuarios = {"admin": "admin", "usuario": "usuario"}
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.title("Inicio de Sesión")
    usuario = st.text_input("Usuario")
    contraseña = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        if usuario in usuarios and usuarios[usuario] == contraseña:
            st.session_state.autenticado = True
            st.success("Inicio de sesión exitoso")
        else:
            st.error("Usuario o contraseña incorrectos")
    st.stop()

# ✅ Navegación lateral
opcion = st.sidebar.radio("Menú", ["Inicio","Formulario de Predicción", "Visualizaciones", "Predicción"])

# ✅ Formulario de predicción
if opcion == "Formulario de Predicción":
    st.title("Predicción de Diabetes")
    gender = st.selectbox("Género", ["Male", "Female"])
    age = st.slider("Edad", 0, 100, 30)
    hypertension = st.selectbox("Hipertensión", [0, 1])
    heart_disease = st.selectbox("Enfermedad cardíaca", [0, 1])
    smoking_history = st.selectbox("Historial de tabaquismo", ["never", "No info", "current", "former","ever", "not current"])
    bmi = st.number_input("BMI", value=25.0)
    hba1c_level = st.number_input("HbA1c", value=5.5)
    blood_glucose_level = st.number_input("Glucosa", value=120)
    diabetes = st.selectbox("¿Has tenido diabetes anteriormente?", [0, 1], help="0 = No, 1 = Sí")
   
    
    if st.button("Predecir"):
        payload = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "hba1c_level": hba1c_level,
            "blood_glucose_level": blood_glucose_level,
            "diabetes": diabetes
        }
        try:
            # ✅ Llamada para obtener predicción
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                st.success(f"Resultado: {response.json()['diabetes_prediction']}")

                # ✅ Guardar datos en la base de datos
                insert_response = requests.post(f"{API_URL}/insert", json=payload)
                if insert_response.status_code == 200:
                    st.info("✅ Datos guardados en la BD")
                else:
                    st.warning(f"No se pudieron guardar los datos: {insert_response.text}")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"No se pudo conectar a la API: {e}")

# ✅ Visualizaciones EDA
elif opcion == "Visualizaciones":
    st.title("Visualizaciones")
    bins = [0, 30, 45, 60, 75, 100]
    labels = ['0-30', '31-45', '46-60', '61-75', '76+']
    datos['age_group'] = pd.cut(datos['age'], bins=bins, labels=labels, right=False)
    datos['genero_enfermedad'] = datos['gender'] + ' - ' + datos['heart_disease'].astype(str)

    st.plotly_chart(px.histogram(datos, x='age_group', color='diabetes', barmode='group',
                                 title='Distribución de Diabetes por Grupo de Edad',
                                 category_orders={'age_group': labels}))
    st.plotly_chart(px.box(datos, x='diabetes', y='bmi', color='diabetes',
                           title='Boxplot de BMI por Clase de Diabetes'))
    st.plotly_chart(px.histogram(datos, x='smoking_history', color='gender', facet_col='diabetes',
                                 title='Distribución de Género y Tabaquismo por Clase de Diabetes'))
    st.plotly_chart(px.scatter(datos, x='hba1c_level', y='blood_glucose_level', color='diabetes',
                               title='Relación entre HbA1c y Glucosa por Clase de Diabetes'))
    st.plotly_chart(px.histogram(datos, x='gender', color='diabetes', barmode='group',
                                 title='Distribución de Diabetes por Género'))
    st.plotly_chart(px.histogram(datos, x='smoking_history', color='diabetes', barmode='group',
                                 title='Diabetes según Historial de Tabaquismo'))
    st.plotly_chart(px.histogram(datos, x='genero_enfermedad', color='diabetes', barmode='group',
                                 title='Diabetes según Combinación de Género y Enfermedad Cardiaca'))
elif opcion=="Inicio":
    st.title("🩺 Bienvenido a la Plataforma de Predicción de Diabetes")
    st.text("Esta aplicación te permite explorar datos clínicos relacionados con la diabetes, realizar predicciones personalizadas y entrenar modelos de inteligencia artificial para mejorar el diagnóstico.\n" \
    "🔍 ¿Qué puedes hacer aquí? Completar un formulario con tus datos para obtener una predicción sobre la probabilidad de tener diabetes.\n" \
    "Visualizar gráficas interactivas que muestran cómo se relacionan factores como edad, género, tabaquismo y niveles de glucosa con la diabetes.\n" \
    "Entrenar modelos de machine learning y comparar su rendimiento.Guardar tus datos en una base de datos segura para análisis posteriores." \
    "Esta herramienta está diseñada para fines educativos y de investigación. No sustituye el diagnóstico médico profesional.\n")
    st.image("diabetes-symptoms-information-infographic-free-vector.jpg", width=500)
    

elif opcion == "Predicción":
    st.title("Entrenamiento con Random Forest")
    if st.button("Entrenar Modelo"):
        # ✅ Mostrar spinner mientras se entrena
        with st.spinner("Entrenando el modelo, por favor espera..."):
            # ✅ Preprocesamiento
            num_cols = ['age', 'bmi', 'hba1c_level', 'blood_glucose_level']
            scaler = StandardScaler()
            datos[num_cols] = scaler.fit_transform(datos[num_cols])

            X = pd.get_dummies(datos.drop(columns='diabetes'), columns=['gender', 'smoking_history'], drop_first=True)
            y = datos['diabetes']
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

            # ✅ División y balanceo
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            # ✅ Entrenamiento Random Forest
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_train_res, y_train_res)
            rf_pred = rf.predict(X_test)

            # ✅ Matriz de confusión
            st.subheader("Matriz de confusión de Random Forest")
            fig_rf, ax_rf = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues')
            st.pyplot(fig_rf)

            # ✅ Métricas
            accuracy = accuracy_score(y_test, rf_pred)
            f1 = f1_score(y_test, rf_pred)
            st.write(f"**Accuracy:** {accuracy:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")

        # ✅ Mensaje cuando termina
        st.success("Entrenamiento completado ✅")
