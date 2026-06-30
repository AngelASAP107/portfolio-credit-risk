from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .schemas import ClienteInput, PrediccionOutput
from .model import load_model, predict

# Estado global — modelo cargado una sola vez al arrancar
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state['model'], state['metadata'] = load_model()
    yield
    state.clear()


app = FastAPI(
    title='Credit Risk API',
    description='Predicción de riesgo de incumplimiento crediticio — Home Credit',
    version='1.0.0',
    lifespan=lifespan
)


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model': state['metadata']['model'],
        'roc_auc': state['metadata']['roc_auc'],
        'n_features': state['metadata']['n_features']
    }


@app.post('/predict', response_model=PrediccionOutput)
def predict_endpoint(cliente: ClienteInput):
    try:
        input_data = cliente.model_dump(exclude_none=False)
        proba = predict(state['model'], state['metadata'], input_data)

        if proba >= 0.5:
            decision = 'RECHAZADO'
        else:
            decision = 'APROBADO'

        if proba < 0.2:
            nivel_riesgo = 'BAJO'
        elif proba < 0.5:
            nivel_riesgo = 'MEDIO'
        else:
            nivel_riesgo = 'ALTO'

        return PrediccionOutput(
            probabilidad_incumplimiento=round(proba, 4),
            decision=decision,
            nivel_riesgo=nivel_riesgo
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
