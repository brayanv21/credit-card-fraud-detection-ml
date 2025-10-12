from utils import db_connect
engine = db_connect()

# your code here
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, RocCurveDisplay
)
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="💳 Detección de Fraudes", page_icon="💳", layout="wide")
st.title("💳 Detección de Fraudes con Tarjetas de Crédito")

st.markdown("""
Esta aplicación fue desarrollada por **Brayan Vera** para identificar transacciones fraudulentas usando modelos de Machine Learning previamente entrenados.  
⬆️ Sube un archivo `.csv` y selecciona un modelo para realizar predicciones.
""")

# --- CARGA DE MODELOS ---
st.sidebar.header("🔍 Selección de Modelo")
modelos = {
    "Regresión Logística": joblib.load("src/modelo_lr.pkl"),
    "Random Forest": joblib.load("src/modelo_rf.pkl"),
    "XGBoost": joblib.load("src/modelo_xgb.pkl")
}
modelo_seleccionado = st.sidebar.selectbox("Selecciona un modelo:", list(modelos.keys()))
modelo = modelos[modelo_seleccionado]

# --- CARGA DEL ARCHIVO CSV ---
archivo = st.file_uploader("📂 Sube tu archivo CSV con transacciones", type=["csv"])

if archivo:
    df = pd.read_csv(archivo)

    # --- VALIDACIÓN DE ESTRUCTURA ---
    columnas_esperadas = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    faltantes = set(columnas_esperadas) - set(df.columns)

    if faltantes:
        st.error(f"❌ El archivo no contiene las columnas requeridas: {', '.join(faltantes)}")
        st.stop()

    # --- ESCALADO DE 'Amount' y 'Time' ---
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    df['Amount'] = scaler_amount.fit_transform(df[['Amount']])
    df['Time'] = scaler_time.fit_transform(df[['Time']])

    # --- VISTA PREVIA DE LOS DATOS ---
    st.subheader("👀 Vista previa de los datos")
    st.dataframe(df.head(10), use_container_width=True)

    # --- DISTRIBUCIÓN DE CLASES ---
    st.subheader("📊 Distribución de Clases")
    fig1, ax1 = plt.subplots(figsize=(4, 2.5))
    sns.countplot(x='Class', data=df, palette='Set1', ax=ax1)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Legítima (0)', 'Fraude (1)'])
    ax1.set_title("Transacciones Legítimas vs Fraudulentas")
    st.pyplot(fig1)

    # --- PREDICCIÓN ---
    st.subheader("🧠 Resultados del Modelo")
    X = df.drop('Class', axis=1)
    y = df['Class']
    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:, 1]

    # --- MÉTRICAS ---
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    reporte = classification_report(y, y_pred, output_dict=True, zero_division=0)
    df_reporte = pd.DataFrame(reporte).transpose()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🎯 AUC - ROC", value=f"{auc:.4f}")
        st.write("📌 Matriz de Confusión")
        st.dataframe(pd.DataFrame(cm,
            columns=["Pred. No Fraude", "Pred. Fraude"],
            index=["Real No Fraude", "Real Fraude"]
        ))

    with col2:
        st.write("📌 Reporte de Clasificación")
        st.dataframe(df_reporte.style.background_gradient(cmap="Blues"))

    # --- CURVA ROC ---
    st.subheader("📈 Curva ROC")
    fig2, ax2 = plt.subplots()
    RocCurveDisplay.from_predictions(y, y_prob, ax=ax2)
    ax2.set_title("Curva ROC del modelo seleccionado")
    st.pyplot(fig2)

    # --- IMPORTANCIA DE VARIABLES (solo RF/XGB) ---
    if modelo_seleccionado in ["Random Forest", "XGBoost"]:
        st.subheader("🔎 Importancia de Variables")
        importancias = modelo.feature_importances_
        df_importancia = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importancias
        }).sort_values(by='Importance', ascending=False)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=df_importancia.head(10), ax=ax3, palette='viridis')
        ax3.set_title("Top 10 Variables más importantes")
        st.pyplot(fig3)

    # --- DESCARGA DE RESULTADOS ---
    df_resultado = df.copy()
    df_resultado['Predicción'] = y_pred
    df_resultado['Probabilidad_Fraude'] = y_prob

    csv = df_resultado.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar resultados como CSV",
        data=csv,
        file_name="resultados_prediccion.csv",
        mime="text/csv"
    )

    # --- INFORMACIÓN DEL MODELO ---
    st.sidebar.markdown("### ℹ️ Sobre el modelo seleccionado")
    if modelo_seleccionado == "Regresión Logística":
        st.sidebar.info("Modelo lineal interpretable que estima probabilidades de fraude. Ideal como línea base.")
    elif modelo_seleccionado == "Random Forest":
        st.sidebar.info("Modelo de ensamble robusto que utiliza múltiples árboles de decisión. Eficiente y preciso.")
    elif modelo_seleccionado == "XGBoost":
        st.sidebar.info("Algoritmo avanzado de boosting que detecta patrones complejos. Requiere más recursos, pero ofrece alto rendimiento.")

else:
    st.info("⬆️ Sube un archivo `.csv` con transacciones para comenzar.")
