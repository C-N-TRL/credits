from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.predict import router as predict_router
from src.services.scorer import Scorer

@asynccontextmanager
async def config(app: FastAPI):
    app.state.scorer = Scorer(path='model/credit_model.joblib')
    yield
    app.state.scorer = None

app = FastAPI(title="Credit Scoring API", lifespan=config)
app.include_router(predict_router)