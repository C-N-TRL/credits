from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.services.scorer import Scorer
ml_models = {}

@asynccontextmanager
async def config(app: FastAPI):
    ml_models['scorer'] = Scorer(path='model/credit_model.joblib')
    yield
    ml_models.clear