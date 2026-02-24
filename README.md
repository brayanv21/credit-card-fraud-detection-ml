# 💳 Credit Card Fraud Detection (Machine Learning)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/Library-XGBoost-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)

## 📌 Descripción del Proyecto
Este proyecto desarrolla un sistema de detección de fraudes en transacciones de tarjetas de crédito utilizando **XGBoost** y otras técnicas de Machine Learning. El desafío principal es el **desequilibrio extremo de las clases**, donde las transacciones fraudulentas representan apenas el 0.17% del total.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python
* **Algoritmos:** XGBoost, Random Forest, Regresión Logística.
* **Procesamiento:** Scikit-Learn, Pandas, NumPy.
* **Manejo de Desequilibrio:** SMOTE (Synthetic Minority Over-sampling Technique).

## 📈 Metodología y Estrategia
Para maximizar la detección de fraude, se aplicó la siguiente estrategia técnica:
1. **Escalado Robusto:** Uso de `RobustScaler` para las variables de tiempo y monto, minimizando el impacto de valores atípicos (outliers).
2. **Optimización de Balanceo:** Implementación de **SMOTE** para generar ejemplos sintéticos de la clase minoritaria (fraude).
3. **Métricas Críticas:** Se optimizó el modelo basándose en **AUPRC (Area Under the Precision-Recall Curve)** y **F1-Score**, garantizando que el modelo no solo sea preciso, sino que detecte la mayor cantidad de fraudes posible.

## 📊 Comparativa de Modelos
| Modelo | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Regresión Logística | 0.88 | 0.62 | 0.73 |
| Random Forest | 0.94 | 0.81 | 0.87 |
| **XGBoost (Final)** | **0.96** | **0.84** | **0.90** |

> **Conclusión:** **XGBoost** demostró ser el modelo más robusto, logrando un balance superior entre la precisión y la capacidad de detección (Recall), reduciendo significativamente los falsos negativos.

## 🚀 Instalación y Uso
1. Clona este repositorio:
   ```bash
   git clone [https://github.com/brayanv21/credit-card-fraud-detection-ml.git](https://github.com/brayanv21/credit-card-fraud-detection-ml.git)
