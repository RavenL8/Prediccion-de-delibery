# app.py
# Calculadora logística simplificada:
# Comparación de modalidades de envío + alertas operativas.

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

# =========================================
# Configuración de página
# =========================================
st.set_page_config(page_title="Calculadora Logística", page_icon="🚚", layout="wide")
st.title("🚚 Calculadora Inteligente de Envíos")
st.caption("Predicción de retraso + tiempo y costo estimados según modalidad y perfil.")

# =========================================
# Cargar modelo
# =========================================
@st.cache_resource
def load_model(path="modelo_retraso.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {e}")
        return None

modelo = load_model("modelo_retraso.pkl")

# =========================================
# Configuración de negocio
# =========================================
MODALIDADES = {
    "económico": {"tiempo": 1.3, "costo": 0.8},
    "estándar": {"tiempo": 1.0, "costo": 1.0},
    "express":  {"tiempo": 0.7, "costo": 1.5},
}

PERFILES = {
    "urgente":   {"tiempo": 0.9, "costo": 1.2},
    "económico": {"tiempo": 1.1, "costo": 0.8},
}

FEATURES = [
    "Distancia_KM",
    "Trafico_Score",
    "Riesgo_Clima",
    "Ratio_Inv",
    "Waiting_Time",
    "Temperature",
    "Humidity",
    "Traffic_Num",
]

# =========================================
# Funciones auxiliares
# =========================================
def calcular_tiempo_base(distancia_km: float) -> float:
    return (distancia_km / 50.0) + 2.0

def calcular_costo_base(distancia_km: float, peso_kg: float = 10.0) -> float:
    return 5.0 + (0.5 * distancia_km) + (1.0 * peso_kg)

def predecir_envio(modelo, datos_envio: dict, modalidad: str, perfil: str):
    df_input = pd.DataFrame([datos_envio], columns=FEATURES)
    prob_retraso = float(modelo.predict_proba(df_input)[0][1])

    distancia = float(datos_envio["Distancia_KM"])
    tiempo_base = calcular_tiempo_base(distancia)
    costo_base = calcular_costo_base(distancia)

    if prob_retraso > 0.6:
        tiempo_estimado = tiempo_base * 1.5
        costo_estimado = costo_base * 1.30
        riesgo = "alto"
    elif prob_retraso > 0.4:
        tiempo_estimado = tiempo_base * 1.2
        costo_estimado = costo_base * 1.15
        riesgo = "medio"
    else:
        tiempo_estimado = tiempo_base
        costo_estimado = costo_base
        riesgo = "bajo"

    tiempo_estimado *= MODALIDADES[modalidad]["tiempo"]
    costo_estimado *= MODALIDADES[modalidad]["costo"]

    tiempo_estimado *= PERFILES[perfil]["tiempo"]
    costo_estimado *= PERFILES[perfil]["costo"]

    return {
        "prob_retraso": prob_retraso,
        "riesgo": riesgo,
        "tiempo_horas": tiempo_estimado,
        "costo_usd": costo_estimado,
    }

# =========================================
# Tabs principales
# =========================================
tab1, tab2 = st.tabs(["🔮 Comparador de modalidades", "⚠️ Alertas operativas"])

# =========================================
# TAB 1: Comparador
# =========================================
with tab1:
    st.subheader("Comparador de modalidades de envío")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        distancia_km = st.number_input("Distancia aprox. (km)", 1, 2000, 50, key="distancia")
        waiting = st.slider("Tiempo de espera (min)", 0, 240, 10, 5, key="espera")
    with colB:
        trafico = st.selectbox("Tráfico", ["Clear", "Detour", "Heavy"], key="trafico")
        trafico_score = {"Clear": 1, "Detour": 2, "Heavy": 3}[trafico]
        trafico_num = {"Clear": 0, "Detour": 1, "Heavy": 2}[trafico]
    with colC:
        temp = st.number_input("Temperatura (°C)", -20, 55, 25, key="temp")
        hum = st.number_input("Humedad (%)", 0, 100, 60, key="hum")
        riesgo_clima = 1 if (temp > 30 or hum > 80) else 0
    with colD:
        inv_ratio = st.slider("Ratio Inventario/Demanda", 0.05, 3.0, 1.0, 0.05, key="inv")
        perfil = st.selectbox("Perfil usuario", list(PERFILES.keys()), key="perfil")

    if modelo and st.button("Calcular opciones", key="calc"):
        datos_envio = {
            "Distancia_KM": distancia_km,
            "Trafico_Score": trafico_score,
            "Riesgo_Clima": riesgo_clima,
            "Ratio_Inv": inv_ratio,
            "Waiting_Time": waiting,
            "Temperature": temp,
            "Humidity": hum,
            "Traffic_Num": trafico_num,
        }

        resultados = []
        for modalidad in MODALIDADES.keys():
            out = predecir_envio(modelo, datos_envio, modalidad, perfil)
            fecha_prob = datetime.now() + timedelta(hours=out["tiempo_horas"])
            resultados.append({
                "Modalidad": modalidad,
                "Tiempo estimado": f"{out['tiempo_horas']:.1f} h",
                "Fecha probable": fecha_prob.strftime("%Y-%m-%d %H:%M"),
                "Costo estimado": f"${out['costo_usd']:.2f}",
                "Prob. retraso": f"{out['prob_retraso']:.1%}",
                "Riesgo": out["riesgo"],
            })

        st.dataframe(pd.DataFrame(resultados), use_container_width=True)

# =========================================
# TAB 2: Alertas
# =========================================
with tab2:
    st.subheader("Alertas de optimización temporal")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        trafico_alert = st.selectbox("Tráfico actual", ["Clear", "Detour", "Heavy"], key="trafico_alert")
    with col2:
        temp_alert = st.number_input("Temperatura (°C)", -20, 55, 25, key="temp_alert")
    with col3:
        hum_alert = st.number_input("Humedad (%)", 0, 100, 60, key="hum_alert")
    with col4:
        inv_ratio_alert = st.slider("Ratio Inventario/Demanda", 0.05, 3.0, 1.0, 0.05, key="inv_alert")

    waiting_alert = st.slider("Tiempo de espera (min)", 0, 240, 10, 5, key="wait_alert")

    # Alertas simples
    if trafico_alert == "Heavy":
        st.warning("🚦 Tráfico pesado: riesgo alto de retraso.")
    else:
        st.success("✅ Tráfico en condiciones normales.")

    if (temp_alert > 30) or (hum_alert > 80):
        st.warning("🌧️ Clima adverso: añadir buffer de tiempo.")
    else:
        st.success("✅ Clima sin riesgos.")

    if inv_ratio_alert < 0.5:
        st.error("📦 Inventario insuficiente: riesgo de quiebre de stock.")
    else:
        st.success("✅ Inventario en rango seguro.")

    if waiting_alert > 60:
        st.error("⏱️ Tiempo de espera elevado: revisar operaciones.")
    else:
        st.success("✅ Tiempo de espera controlado.")

