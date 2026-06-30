# portfolio-credit-risk

Predicción de riesgo de crédito usando el dataset **Home Credit Default Risk** de Kaggle.

## Objetivo

Predecir si un solicitante de crédito incumplirá su préstamo (`TARGET = 1`), combinando modelado serio de machine learning con despliegue como API REST en Docker.

## Estructura del proyecto

```
credit-risk-homecredit/
├── data/
│   ├── raw/              # Datos originales de Kaggle (no versionados)
│   └── processed/        # Datos transformados
├── notebooks/
│   ├── 01_eda.ipynb                          # Análisis exploratorio
│   ├── 02_baseline_model.ipynb               # Modelo baseline
│   ├── 03_feature_engineering_relacional.ipynb
│   └── 04_final_model_shap.ipynb
├── src/
│   ├── features/         # Feature engineering
│   ├── models/           # Entrenamiento y evaluación
│   └── api/              # FastAPI
├── docker/
├── tests/
├── requirements.txt
└── README.md
```

## Dataset

**Home Credit Default Risk** — Kaggle Competition  
9 tablas relacionales, ~57 millones de filas en total.

| Tabla | Filas | Descripción |
|---|---|---|
| `application_train.csv` | 307,511 | Tabla principal — un solicitante por fila |
| `bureau.csv` | 1,716,428 | Créditos en otras entidades |
| `bureau_balance.csv` | 27,299,925 | Historial mensual de créditos en buró |
| `previous_application.csv` | 1,670,214 | Solicitudes anteriores en Home Credit |
| `installments_payments.csv` | 13,605,401 | Pagos cuota por cuota |
| `POS_CASH_balance.csv` | 10,001,358 | Saldos mensuales créditos POS |
| `credit_card_balance.csv` | 3,840,312 | Saldos mensuales tarjetas |

## Fases del proyecto

### Fase A — Modelo funcional (tabla principal)
- EDA completo de `application_train`
- Modelo baseline (regresión logística)
- Modelo principal (LightGBM) con manejo de desbalance
- Interpretabilidad con SHAP
- API REST con FastAPI + Docker

### Fase B — Feature engineering relacional
- Agregaciones desde las 6 tablas secundarias
- Historial de comportamiento de pago por cliente
- Narrativa de mejora: métrica Fase A → métrica Fase B

## Métricas

Métrica principal: **ROC-AUC**. También: Precision, Recall, F1.  
No se usa accuracy como métrica principal dado el desbalance (~8% positivos).

## Stack

Python · LightGBM · scikit-learn · SHAP · FastAPI · Docker

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Los datos se descargan con la API de Kaggle:
```bash
kaggle competitions download -c home-credit-default-risk -p data/raw/
cd data/raw && unzip home-credit-default-risk.zip
```
