from pydantic import BaseModel
from typing import Optional


class ClienteInput(BaseModel):
    """Datos de entrada para predecir riesgo de crédito.

    Todas las variables coinciden exactamente con las del modelo entrenado.
    Los campos opcionales pueden ser None — el modelo LightGBM maneja NaN nativo.
    """
    # Scores externos (las features más importantes según SHAP)
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None

    # Información del crédito
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    AMT_INCOME_TOTAL: Optional[float] = None

    # Información personal
    AGE_YEARS: Optional[float] = None
    CODE_GENDER: Optional[str] = None
    CNT_CHILDREN: Optional[float] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None

    # Empleo
    DAYS_EMPLOYED: Optional[float] = None
    FLAG_EMPLOYED_ANOMALY: Optional[int] = None
    ORGANIZATION_TYPE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None

    # Educación y familia
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    CNT_FAM_MEMBERS: Optional[float] = None

    # Tipo de contrato e ingresos
    NAME_CONTRACT_TYPE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None


class PrediccionOutput(BaseModel):
    probabilidad_incumplimiento: float
    decision: str
    nivel_riesgo: str
