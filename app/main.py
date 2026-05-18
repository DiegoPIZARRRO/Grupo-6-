from fastapi import FastAPI
from scipy import stats

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

