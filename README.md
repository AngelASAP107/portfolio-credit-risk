# portfolio-credit-risk

Predicción de riesgo de crédito usando el dataset **Home Credit Default Risk** de Kaggle.  
Proyecto de portafolio completo: desde EDA hasta API REST dockerizada con feature engineering relacional.

## Resultados

| Modelo | Features | ROC-AUC | Recall |
|---|---|---|---|
| Baseline (Regresión Logística) | 72 | 0.7435 | 0.6707 |
| LightGBM Fase A (tabla principal) | 72 | 0.7561 | 0.6602 |
| **LightGBM Fase B (+ tablas secundarias)** | **151** | **0.7765** | **0.6882** |

El modelo final detecta **3,417 de 4,965 incumplimientos reales** (68.8%) en el set de validación.

## Objetivo

Predecir si un solicitante de crédito incumplirá su préstamo (`TARGET = 1`), demostrando el ciclo completo de un proyecto de datos: análisis exploratorio → modelado → interpretabilidad → API en producción.

## Estructura del proyecto

```
credit-risk-homecredit/
├── data/
│   ├── raw/              # Datos originales de Kaggle (no versionados)
│   └── processed/        # Features agregadas por tabla (.parquet)
├── notebooks/
│   ├── 01_eda.ipynb                    # Análisis exploratorio
│   ├── 02_baseline_model.ipynb         # Regresión Logística — ROC-AUC 0.7435
│   ├── 03_lightgbm_model.ipynb         # LightGBM Fase A — ROC-AUC 0.7561
│   ├── 04_shap.ipynb                   # Interpretabilidad SHAP
│   ├── 05_save_model.ipynb             # Serialización del modelo (.pkl)
│   ├── 06_features_bureau.ipynb        # Features bureau + bureau_balance (17)
│   ├── 07_features_previous_app.ipynb  # Features previous_application (16)
│   ├── 08_features_installments.ipynb  # Features installments_payments (15)
│   ├── 09_features_pos_cash.ipynb      # Features POS_CASH_balance (14)
│   ├── 10_features_credit_card.ipynb   # Features credit_card_balance (17)
│   └── 11_lgbm_fase_b.ipynb            # LightGBM Fase B — ROC-AUC 0.7765
├── models/
│   ├── lgbm_credit_risk.pkl            # Modelo entrenado (no versionado)
│   └── model_metadata.json             # Metadatos y métricas
├── src/
│   └── api/
│       ├── main.py                     # FastAPI — endpoints /health y /predict
│       ├── model.py                    # Carga del modelo y lógica de predicción
│       └── schemas.py                  # Esquemas de entrada y salida
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Dataset

**Home Credit Default Risk** — [Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk)  
7 tablas relacionales, ~57 millones de filas en total.

| Tabla | Filas | Descripción |
|---|---|---|
| `application_train.csv` | 307,511 | Tabla principal — un solicitante por fila |
| `bureau.csv` | 1,716,428 | Créditos reportados en buró externo |
| `bureau_balance.csv` | 27,299,925 | Estado mensual de créditos en buró |
| `previous_application.csv` | 1,670,214 | Solicitudes anteriores en Home Credit |
| `installments_payments.csv` | 13,605,401 | Historial de pagos cuota por cuota |
| `POS_CASH_balance.csv` | 10,001,358 | Estado mensual de créditos POS y efectivo |
| `credit_card_balance.csv` | 3,840,312 | Estado mensual de tarjetas de crédito |

## Metodología

### Fase A — Tabla principal
1. **EDA** — desbalance de clases (8%/92%), top predictores (EXT_SOURCE_2/3), anomalías en DAYS_EMPLOYED
2. **Baseline** — Regresión Logística con sklearn Pipeline, `class_weight='balanced'`
3. **LightGBM** — manejo nativo de NaN y categóricas, `scale_pos_weight`, early stopping sobre AUC
4. **SHAP** — EXT_SOURCE_3 y EXT_SOURCE_2 dominan con impacto promedio de 0.40 y 0.35

### Fase B — Feature engineering relacional
Cada tabla secundaria tiene relación uno-a-muchos con la tabla principal. Se colapsó en una fila por cliente mediante agregaciones (mean, max, sum, ratios):

- **bureau**: cantidad de créditos, deuda activa, días de mora, ratio deuda/crédito
- **previous_application**: ratio de rechazo/aprobación, montos históricos, duración de créditos
- **installments**: ratio de pagos tardíos (`INST_LATE_RATIO`), diferencia monto debido vs pagado
- **POS_CASH**: meses en mora (DPD), cuotas futuras pendientes
- **credit_card**: utilización de límite, retiros ATM, saldo promedio

## API REST

La API expone el modelo como servicio con dos endpoints:

```
GET  /health   → estado del modelo cargado
POST /predict  → recibe datos del cliente, devuelve probabilidad y decisión
```

Ejemplo de request:
```json
{
  "EXT_SOURCE_2": 0.3,
  "EXT_SOURCE_3": 0.4,
  "AGE_YEARS": 35,
  "AMT_CREDIT": 300000
}
```

Ejemplo de response:
```json
{
  "probabilidad_incumplimiento": 0.3821,
  "decision": "RECHAZADO",
  "nivel_riesgo": "MEDIO"
}
```

## Correr con Docker

```bash
docker compose up --build
```

La API queda disponible en `http://localhost:8000`.  
Documentación interactiva: `http://localhost:8000/docs`

## Instalación local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datos desde Kaggle:
```bash
kaggle competitions download -c home-credit-default-risk -p data/raw/
cd data/raw && unzip home-credit-default-risk.zip
```

## Stack

Python · LightGBM · scikit-learn · SHAP · FastAPI · Uvicorn · Docker · pandas · joblib
