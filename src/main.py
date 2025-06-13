from fastapi import FastAPI
from src.db import init_db
init_db()
app = FastAPI()
@app.get('/health')
def health(): return {'status':'ok'}