from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

app = FastAPI(title="API Predicción Diabetes")

# ✅ Conexión a Neon usando DATABASE_URL
DATABASE_URL = 'postgresql://neondb_owner:npg_BDG2IiT0aqAy@ep-super-heart-agp17yzq-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está definido. Configura la variable de entorno correctamente.")

engine = create_engine(DATABASE_URL)

# ✅ Cargar datos desde Neon
def cargar_datos():
    query = "SELECT * FROM diabetes"
    return pd.read_sql(query, engine)

# ✅ Entrenar modelos y generar gráficas
@app.get("/train")
def entrenar_modelos():
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

    # ✅ Modelo 1: Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_res, y_train_res)
    rf_pred = rf.predict(X_test)
    joblib.dump(rf, "modelo_diabetes_1.pkl")
    cm_rf = confusion_matrix(y_test, rf_pred)
    fig_rf = plt.figure()
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title("Random Forest")
    fig_rf.savefig("confusion_rf.png")
    plt.close(fig_rf)

    # ✅ Modelo 2: Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_res, y_train_res)
    lr_pred = lr.predict(X_test)
    joblib.dump(lr, "modelo_diabetes_2.pkl")
    cm_lr = confusion_matrix(y_test, lr_pred)
    fig_lr = plt.figure()
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title("Logistic Regression")
    fig_lr.savefig("confusion_lr.png")
    plt.close(fig_lr)

    # ✅ Modelo 3: Keras NN
    model_nn = Sequential([
        Dense(64, input_dim=X_train_res.shape[1], activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_nn.fit(X_train_res, y_train_res, validation_split=0.2, epochs=30, batch_size=64, callbacks=[early_stop])
    model_nn.save("modelo_diabetes_3.h5")
    keras_pred = (model_nn.predict(X_test) > 0.5).astype(int)
    cm_keras = confusion_matrix(y_test, keras_pred)
    fig_keras = plt.figure()
    sns.heatmap(cm_keras, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title("Keras NN")
    fig_keras.savefig("confusion_keras.png")
    plt.close(fig_keras)

    # ✅ Comparativa de métricas
    resultados = pd.DataFrame({
        'Modelo': ['Random Forest', 'Logistic Regression', 'Keras NN'],
        'Accuracy': [accuracy_score(y_test, rf_pred), accuracy_score(y_test, lr_pred), accuracy_score(y_test, keras_pred)],
        'F1 Score': [f1_score(y_test, rf_pred), f1_score(y_test, lr_pred), f1_score(y_test, keras_pred)],
        'Precision': [precision_score(y_test, rf_pred), precision_score(y_test, lr_pred), precision_score(y_test, keras_pred)],
        'Recall': [recall_score(y_test, rf_pred), recall_score(y_test, lr_pred), recall_score(y_test, keras_pred)]
    })
    resultados.to_csv("metricas_modelos.csv", index=False)

    for metrica in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
        fig = plt.figure()
        sns.barplot(x='Modelo', y=metrica, data=resultados)
        plt.title(f'Comparativa de {metrica}')
        fig.savefig(f"comparativa_{metrica.lower().replace(' ', '_')}.png")
        plt.close(fig)

    return {"message": "Modelos entrenados y gráficas generadas"}

# ✅ Esquema para predicciones
class DiabetesInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    hba1c_level: float
    blood_glucose_level: float
    modelo_id: int = 1

@app.post("/predict")
def predict(data: DiabetesInput):
    model_path = {1: "modelo_diabetes_1.pkl", 2: "modelo_diabetes_2.pkl", 3: "modelo_diabetes_3.h5"}.get(data.modelo_id)
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    df = pd.DataFrame([data.dict(exclude={"modelo_id"})])
    df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

    if data.modelo_id == 3:
        model = load_model(model_path)
        prediction = int((model.predict(df)[0][0] > 0.5))
    else:
        model = joblib.load(model_path)
        prediction = int(model.predict(df)[0])

    return {"diabetes_prediction": prediction}

@app.get("/grafica/{nombre}")
def get_grafica(nombre: str):
    filename = f"{nombre}.png"
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Gráfica no encontrada")
    return FileResponse(filename, media_type="image/png")

@app.get("/metrics")
def get_metrics():
    if not os.path.exists("metricas_modelos.csv"):
        raise HTTPException(status_code=404, detail="Archivo de métricas no encontrado")
    df = pd.read_csv("metricas_modelos.csv")
    return df.to_dict(orient="records")