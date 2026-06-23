from fastapi import FastAPI, HTTPException, Query
from scipy import stats
from app.db import test_connection, get_clientes

app = FastAPI(title="MVP pipline telco")

@app.get("/")
def root():
    return {"message": "¡API MVP Telco activa!"}       

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/db-health")
def db_health_check():
    return {"status": "ok"}

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

