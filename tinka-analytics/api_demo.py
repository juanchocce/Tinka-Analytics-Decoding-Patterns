from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import numpy as np

app = FastAPI(
    title="Tinka Analytics AI API",
    description="Demo deployment for Tinka Analytics Predictive Models",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    # Enforce exactly 6 ints between 1 and 50
    numeros: conlist(int, min_length=6, max_length=6)
    
class PredictionResponse(BaseModel):
    score_rareza: float
    mensaje: str
    recomendacion: str
    
@app.get("/")
def read_root():
    return {"status": "AI Model API is Running", "docs": "/docs"}

@app.post("/predict", response_model=PredictionResponse)
def predict_combination(request: PredictionRequest):
    numeros = request.numeros
    
    if any(n < 1 or n > 50 for n in numeros):
        raise HTTPException(status_code=400, detail="Los números deben estar entre 1 y 50.")
        
    if len(set(numeros)) != 6:
        raise HTTPException(status_code=400, detail="Los números no pueden repetirse.")
        
    suma = sum(numeros)
    media_teorica = 153 # 6 * 25.5
    
    desviacion = abs(suma - media_teorica) / media_teorica
    score_rareza = round(min(100.0, desviacion * 200.0), 2)
    
    if score_rareza < 20:
        mensaje = "Combinación común. Se alinea con el Centroide del Teorema del Límite Central."
        recomendacion = "Jugarla implicaría repartir un posible pozo con múltiples ganadores."
    elif score_rareza < 60:
        mensaje = "Combinación moderadamente inusual."
        recomendacion = "Balance aceptable entre probabilidad y exclusividad."
    else:
        mensaje = "Combinación altamente anómala (Rareza Extrema)."
        recomendacion = "Baja probabilidad de salir, pero de acertar, garantiza pozo único."
        
    return PredictionResponse(
        score_rareza=score_rareza,
        mensaje=mensaje,
        recomendacion=recomendacion
    )

# Para ejecutar:
# uvicorn api_demo:app --reload
