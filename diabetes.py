import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
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

API_URL = "https://fastapi-diabetes.onrender.com/data"

st.title("Predicción de Diabetes")

# ✅ Formulario de predicción
with st.expander("Formulario de Predicción"):
    gender = st.selectbox("Género", ["Male", "Female"])
    age = st.slider("Edad", 0, 100, 30)
    hypertension = st.selectbox("Hipertensión", [0, 1])
    heart_disease = st.selectbox("Enfermedad cardíaca", [0, 1])
    smoking_history = st.selectbox("Historial de tabaquismo", ["never", "former", "current"])
    bmi = st.number_input("BMI", value=25.0)
    hba1c_level = st.number_input("HbA1c", value=5.5)
    blood_glucose_level = st.number_input("Glucosa", value=120)

    if st.button("Predecir"):
        payload = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "hba1c_level": hba1c_level,
            "blood_glucose_level": blood_glucose_level
        }
        try:
            response = requests.post(f"{API_URL}/data", json=payload)
            if response.status_code == 200:
                st.success(f"Resultado: {response.json()['diabetes_prediction']}")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"No se pudo conectar a la API: {e}")

# ✅ Visualizaciones EDA
with st.expander("Visualizaciones EDA"):
    if st.button("Mostrar Gráficas EDA"):
        bins = [0, 30, 45, 60, 75, 100]
        labels = ['0-30', '31-45', '46-60', '61-75', '76+']
        datos['age_group'] = pd.cut(datos['age'], bins=bins, labels=labels, right=False)

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
        
        st.plotly_chart(
            px.histogram(
                datos,
                x='genero_enfermedad',
                color='diabetes',
                barmode='group',
                title='Diabetes según Combinación de Género y Enfermedad Cardiaca'
            )
        )

# ✅ Entrenamiento de modelos
with st.expander("Entrenamiento de Modelos"):
    if st.button("Entrenar Modelos"):
        num_cols = ['age', 'bmi', 'hba1c_level', 'blood_glucose_level']
        scaler = StandardScaler()
        datos[num_cols] = scaler.fit_transform(datos[num_cols])
        X = pd.get_dummies(datos.drop(columns='diabetes'), columns=['gender', 'smoking_history'], drop_first=True)
        y = datos['diabetes']
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # ✅ Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train_res, y_train_res)
        rf_pred = rf.predict(X_test)
        st.subheader("Matriz de Confusión - Random Forest")
        fig_rf, ax_rf = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig_rf)

        # ✅ Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_res, y_train_res)
        lr_pred = lr.predict(X_test)
        st.subheader("Matriz de Confusión - Logistic Regression")
        fig_lr, ax_lr = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig_lr)

        # ✅ Keras NN
        model = Sequential([
            Dense(64, input_dim=X_train_res.shape[1], activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train_res, y_train_res, validation_split=0.2, epochs=30, batch_size=64, callbacks=[early_stop])
        keras_pred = (model.predict(X_test) > 0.5).astype(int)
        st.subheader("Matriz de Confusión - Keras NN")
        fig_keras, ax_keras = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, keras_pred), annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig_keras)

        # ✅ Comparativa de métricas
        resultados = pd.DataFrame({
            'Modelo': ['Random Forest', 'Logistic Regression', 'Keras NN'],
            'Accuracy': [accuracy_score(y_test, rf_pred), accuracy_score(y_test, lr_pred), accuracy_score(y_test, keras_pred)],
            'F1 Score': [f1_score(y_test, rf_pred), f1_score(y_test, lr_pred), f1_score(y_test, keras_pred)]
        })
        st.dataframe(resultados)
        st.subheader("Gráfico Comparativo de Métricas")
        for metrica in ['Accuracy', 'F1 Score']:
            fig = plt.figure()
            sns.barplot(x='Modelo', y=metrica, data=resultados)
            st.pyplot(fig)