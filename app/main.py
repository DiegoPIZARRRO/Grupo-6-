import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from app.db import test_connection, get_clientes, get_connection_params
import psycopg

app = FastAPI(title="MVP pipline telco")

DIRECTORIO_APP = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_RAIZ = os.path.dirname(DIRECTORIO_APP)

ruta_modelo = os.path.join(DIRECTORIO_RAIZ, "artifacts", "predictor_churn_pipeline.joblib")

try:
    modelo_churn = joblib.load(ruta_modelo)
    print(f"¡ÉXITO! El Cerebro IA se ha cargado correctamente desde: {ruta_modelo}")
except Exception as e:
    modelo_churn = None
    print(f"Advertencia: No se pudo cargar el modelo IA. {e}")

@app.get("/")
def root():
    return {"message": "¡API MVP Telco activa!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/db-health")
def db_health_check():
    resultado = test_connection()
    if resultado.get("status") == "error":
        raise HTTPException(status_code=500, detail=resultado.get("detail"))
    return resultado

@app.get("/clientes")
def postulaciones_demo(limit: int = Query(default=20, ge=1)):
    try:
        data = get_clientes(limit=limit)
        return {
            "status": "ok",
            "count": len(data),
            "limit": limit,
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predecir/{customerid}")
def predecir_fuga(customerid: str):
    if modelo_churn is None:
        raise HTTPException(status_code=500, detail="El cerebro IA no está disponible.")
        
    try:
        params = get_connection_params()
        with psycopg.connect(**params) as conn:
            from psycopg.rows import dict_row
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM telco_mvp WHERE customerid = %s", (customerid,))
                cliente = cur.fetchone()
        
        if not cliente:
            raise HTTPException(status_code=404, detail="Cliente no encontrado en la base de datos")

        df_cliente = pd.DataFrame([cliente])
        columnas_a_borrar = ["customerid", "fecha_procesamiento", "churn", "cantidad_features_activas"]
        X_nuevo = df_cliente.drop(columns=columnas_a_borrar, errors="ignore")

        prediccion = modelo_churn.predict(X_nuevo)
        probabilidad = modelo_churn.predict_proba(X_nuevo)[0][1] 

        resultado = "Se fuga (Churn)" if prediccion[0] == 1 else "Se queda (Retención)"

        return {
            "status": "ok",
            "customerid": customerid,
            "prediccion_ia": resultado,
            "probabilidad_de_fuga_porcentaje": round(probabilidad * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")



print(f"DEBUG: Buscando modelo en: {ruta_modelo}")
print(f"DEBUG: ¿El archivo existe realmente?: {os.path.exists(ruta_modelo)}")

try:
    modelo_churn = joblib.load(ruta_modelo)
    print(f"¡ÉXITO! El Cerebro IA se ha cargado correctamente.")
except Exception as e:
    modelo_churn = None
    print(f"ERROR: No se pudo cargar el modelo IA. Razón: {e}")