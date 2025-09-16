# train_model.py
# Entrena el modelo de predicción de retrasos y lo guarda como .pkl

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from geopy.distance import geodesic

# ==========================
# 1. Cargar dataset
# ==========================
df = pd.read_csv("smart_logistics_dataset.csv")

# ==========================
# 2. Crear nuevas variables
# ==========================
# Distancia (usando primera fila como origen aproximado)
origen = (df['Latitude'].iloc[0], df['Longitude'].iloc[0])
df['Distancia_KM'] = df.apply(lambda row: geodesic(origen, (row['Latitude'], row['Longitude'])).km, axis=1)

# Score de tráfico
traffic_scores = {"Clear": 1, "Detour": 2, "Heavy": 3}
df['Trafico_Score'] = df['Traffic_Status'].map(traffic_scores)

# Riesgo climático
df['Riesgo_Clima'] = 0
df.loc[df['Temperature'] > 30, 'Riesgo_Clima'] = 1
df.loc[df['Humidity'] > 80, 'Riesgo_Clima'] = 1

# Ratio de inventario
df['Ratio_Inv'] = df['Inventory_Level'] / df['Demand_Forecast'].replace(0, 1)

# Codificación numérica del tráfico
df['Traffic_Num'] = df['Traffic_Status'].map({'Clear': 0, 'Detour': 1, 'Heavy': 2})

# ==========================
# 3. Definir variables
# ==========================
variables = [
    'Distancia_KM',
    'Trafico_Score',
    'Riesgo_Clima',
    'Ratio_Inv',
    'Waiting_Time',
    'Temperature',
    'Humidity',
    'Traffic_Num'
]

X = df[variables]
y = df['Logistics_Delay']

# ==========================
# 4. Entrenar modelo
# ==========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=50, random_state=42)
modelo.fit(X_train, y_train)

print("✅ Modelo entrenado correctamente")
print(f"Accuracy en train: {modelo.score(X_train, y_train):.2%}")
print(f"Accuracy en test : {modelo.score(X_test, y_test):.2%}")

# ==========================
# 5. Guardar modelo
# ==========================
joblib.dump(modelo, "modelo_retraso.pkl")
print("💾 Modelo guardado como 'modelo_retraso.pkl'")
