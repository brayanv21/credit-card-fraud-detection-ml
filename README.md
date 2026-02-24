# 💳 Credit Card Fraud Detection (Machine Learning)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Descripción del Proyecto
Este proyecto aborda uno de los mayores retos en el sector tecnofinanciero (FinTech): la detección de transacciones fraudulentas. Utilizando un dataset de transacciones europeas, el objetivo es construir un modelo capaz de identificar el fraude con alta precisión en un entorno de **datos extremadamente desequilibrados** (donde solo el 0.17% de los datos son fraude).

## 🛠️ Tecnologías y Herramientas
* **Lenguaje:** Python
* **Librerías Principales:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.
* **Técnicas de ML:** Regresión Logística, Random Forest, SMOTE (Synthetic Minority Over-sampling Technique).

## 📈 Desafíos Técnicos
1. **Desequilibrio de Clases:** Se implementaron técnicas de *Oversampling* (SMOTE) y *Undersampling* para evitar que el modelo ignore la clase minoritaria.
2. **Feature Scaling:** Aplicación de `RobustScaler` en las columnas 'Amount' y 'Time' para manejar valores atípicos.
3. **Métricas de Evaluación:** Dado el desequilibrio, el éxito no se midió con *Accuracy*, sino mediante **Precision-Recall AUC** y **F1-Score**.

## 📊 Resultados obtenidos
| Modelo | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Regresión Logística | 0.88 | 0.62 | 0.73 |
| **Random Forest (Final)** | **0.94** | **0.81** | **0.87** |

> **Nota:** El modelo final logra detectar el 81% de los fraudes (Recall) manteniendo un margen muy bajo de falsas alarmas.

## 🚀 Cómo ejecutar el proyecto
1. Clona el repositorio:
   ```bash
   git clone [https://github.com/brayanv21/credit-card-fraud-detection-ml.git](https://github.com/brayanv21/credit-card-fraud-detection-ml.git)
