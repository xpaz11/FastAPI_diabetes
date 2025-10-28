
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy import create_engine


connection_string = st.secrets["connections"]["neon"]["url"]
engine = create_engine(connection_string)


# Cargar datos
datos = pd.read_sql("SELECT * FROM diabetes", engine)


st.title("Predicción de Diabetes")
st.write("Este proyecto tiene como objetivo explorar patrones en datos clínicos y demográficos para desarrollar modelos de Machine Learning que permitan predecir la presencia de diabetes con la mayor precisión posible. A lo largo de esta plataforma" \
" se presentan diferentes secciones que incluyen la exploración visual del conjunto de datos, la comparación de modelos predictivos y" \
" la posibilidad de realizar predicciones personalizadas.En la primera parte se realiza un análisis exploratorio de datos (EDA) que permite identificar tendencias relevantes, como la relación entre edad, " \
"índice de masa corporal, niveles de glucosa y hemoglobina glicosilada, así como el impacto de factores como género y hábitos de tabaquismo. " \
"Este análisis ayuda a comprender mejor las variables que influyen en la aparición de la enfermedad.")
st.image('diabetes-symptoms-information-infographic-free-vector.jpg', width=500)

# 1. Distribución de Diabetes por Grupo de Edad
st.subheader("1️. Distribución de Diabetes por Grupo de Edad")
bins = [0, 30, 45, 60, 75, 100]
labels = ['0-30', '31-45', '46-60', '61-75', '76+']
datos['age_group'] = pd.cut(datos['age'], bins=bins, labels=labels, right=False)
fig_age = px.histogram(datos, x='age_group', color='diabetes', barmode='group',
                       title='Distribución de Diabetes por Grupo de Edad',
                       category_orders={'age_group': labels})
st.plotly_chart(fig_age)

# 2. Boxplot de BMI por Clase de Diabetes
st.subheader("2️. Comparativa de BMI por Clase de Diabetes")
fig_bmi = px.box(datos, x='diabetes', y='bmi', color='diabetes',
                 title='Boxplot de BMI por Clase de Diabetes',
                 labels={'diabetes': 'Clase de Diabetes', 'bmi': 'Índice de Masa Corporal'})
st.plotly_chart(fig_bmi)

# 3. Impacto Combinado de Género y Tabaquismo
st.subheader("3️. Impacto Combinado de Género y Tabaquismo")
fig_smoke_gender = px.histogram(datos, x='smoking_history', color='gender', facet_col='diabetes',
                                title='Distribución de Género y Tabaquismo por Clase de Diabetes',
                                barmode='group')
st.plotly_chart(fig_smoke_gender)

# 4. Relación entre HbA1c y Glucosa
st.subheader("4. Relación entre HbA1c y Glucosa por Clase de Diabetes")
fig_scatter = px.scatter(datos, x='hba1c_level', y='blood_glucose_level', color='diabetes',
                         title='Relación entre HbA1c y Glucosa por Clase de Diabetes',
                         labels={'HbA1c_level': 'Nivel de HbA1c', 'blood_glucose_level': 'Nivel de Glucosa'})
st.plotly_chart(fig_scatter)

# 5. Comparativa de Diabetes por Género
st.subheader("5️. Comparativa de Diabetes por Género")
fig_gender = px.histogram(datos, x='gender', color='diabetes', barmode='group',
                          title='Distribución de Diabetes por Género',
                          labels={'gender': 'Género', 'diabetes': 'Clase de Diabetes'})
st.plotly_chart(fig_gender)



# --- Análisis por género ---
st.header("6. Tendencias por Género")
fig_gender = px.histogram(
    datos, x="gender", color="diabetes", barmode="group",
    title="Distribución de Diabetes por Género",
    text_auto=True
)
st.plotly_chart(fig_gender)
st.markdown("**Insight:** ¿Hay un género con mayor proporción de diabetes? Observa la diferencia relativa, no solo el conteo.")

# --- Análisis por tabaquismo ---
st.header("7. Impacto del Historial de Tabaquismo")
fig_smoking = px.histogram(
    datos, x="smoking_history", color="diabetes", barmode="group",
    title="Diabetes según Historial de Tabaquismo",
    text_auto=True
)
st.plotly_chart(fig_smoking)
st.markdown("**Insight:** ¿Los fumadores actuales o exfumadores presentan más casos de diabetes que los que nunca fumaron?")


num_cols = ['age', 'bmi', 'hba1c_level', 'blood_glucose_level']
scaler = StandardScaler()
datos[num_cols] = scaler.fit_transform(datos[num_cols])

# Codificar categóricas con One-Hot Encoding
X = pd.get_dummies(datos.drop(columns='diabetes'), columns=['gender', 'smoking_history'], drop_first=True)
y = datos['diabetes']

# Eliminar columnas categóricas residuales (por ejemplo, si usaste pd.cut antes)
X = X.apply(pd.to_numeric, errors='coerce')

# Verificar que no haya NaN tras conversión
if X.isnull().sum().sum() > 0:
    print("Hay valores NaN, revisa las columnas categóricas originales.")
    X = X.fillna(0)  # O maneja según tu lógica

# Confirmar tipos
print("Tipos únicos en X:", X.dtypes.unique())  # Debe mostrar solo 'float64' o 'int64'

# División en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


st.title("Entrenamiento de modelos")
st.header("Random Forest")
st.write("Es un algoritmo basado en árboles de decisión que construye múltiples árboles y combina sus predicciones para mejorar " \
"la precisión y reducir el riesgo de sobreajuste. Cada árbol se entrena con una muestra aleatoria del conjunto de datos y " \
"selecciona características aleatorias en cada división, lo que aporta robustez y estabilidad." \
" Es especialmente eficaz en problemas con datos tabulares y relaciones no lineales.")
# --- MODELO 1: Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_res, y_train_res)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

cm_rf = confusion_matrix(y_test, rf_pred)
# Mostrar matriz de confusión para Random Forest
st.subheader("Matriz de Confusión - Random Forest")
fig_rf, ax_rf = plt.subplots()
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title("Random Forest")
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig_rf)

st.header("Regresión Logística")
st.write("Es un modelo estadístico clásico que se utiliza para problemas de clasificación binaria. Calcula la probabilidad de pertenencia" \
" a una clase mediante una función logística, lo que permite interpretar fácilmente los coeficientes y entender el impacto de cada variable. " \
"Aunque es simple, es un buen punto de partida y sirve como referencia para comparar modelos más complejos.")

# --- MODELO 2: Logistic Regression ---
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)


cm_lr = confusion_matrix(y_test, lr_pred)

# Mostrar matriz de confusión para Logistic Regression
st.subheader("Matriz de Confusión - Logistic Regression")
fig_lr, ax_lr = plt.subplots()
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title("Logistic Regression")
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig_lr)

st.header("Red Neuronal con el uso de la librería keras ")
st.write("Es un modelo inspirado en el funcionamiento del cerebro humano, compuesto por capas de neuronas artificiales que aprenden " \
"patrones complejos en los datos. En este caso, se utiliza una arquitectura con varias capas densas y funciones de activación no lineales, " \
"lo que permite capturar relaciones más sofisticadas entre las variables. Es útil cuando se busca mejorar el rendimiento frente a modelos lineales, " \
"aunque requiere más datos y potencia de cálculo.")



# --- MODELO 3: Red Neuronal con Keras ---
model = Sequential([
    Dense(64, input_dim=X_train_res.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train_res, y_train_res, validation_split=0.2, epochs=30, batch_size=64, callbacks=[early_stop])

keras_loss, keras_acc = model.evaluate(X_test, y_test, verbose=0)
keras_pred = (model.predict(X_test) > 0.5).astype(int)
keras_f1 = f1_score(y_test, keras_pred)

cm_keras = confusion_matrix(y_test, keras_pred)


# Mostrar matriz de confusión para Keras NN
st.subheader("Matriz de Confusión - Keras NN")
fig_keras, ax_keras = plt.subplots()
sns.heatmap(cm_keras, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title("Keras NN")
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig_keras)


# Función para calcular métricas desde matriz de confusión
def calcular_metricas(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return cm, accuracy, precision, recall, f1

# --- Comparativa ---
st.header("Comparativa de Modelos")
resultados = pd.DataFrame({
    'Modelo': ['Random Forest', 'Logistic Regression', 'Keras NN'],
    'Accuracy': [rf_acc, lr_acc, keras_acc],
    'F1 Score': [rf_f1, lr_f1, keras_f1]
})
st.dataframe(resultados)




# Calcular métricas para cada modelo
modelos = ['Random Forest', 'Logistic Regression', 'Keras NN']
predicciones = [rf_pred, lr_pred, keras_pred]

metricas = []
for nombre, pred in zip(modelos, predicciones):
    cm, acc, prec, rec, f1 = calcular_metricas(y_test, pred)
    metricas.append({
        'Modelo': nombre,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Confusion Matrix': cm
    })

# Mostrar tabla comparativa
st.title(" Comparativa de Modelos - Métricas desde Matriz de Confusión")
df_metricas = pd.DataFrame(metricas).drop(columns='Confusion Matrix')
st.dataframe(df_metricas.set_index('Modelo'))

# Gráficos comparativos de métricas
st.subheader("Gráfico Comparativo de Métricas")
for metrica in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    fig = plt.figure()
    sns.barplot(x='Modelo', y=metrica, data=df_metricas)
    plt.title(f'Comparativa de {metrica}')
    plt.ylim(0, 1)
    st.pyplot(fig)


