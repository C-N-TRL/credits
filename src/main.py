from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.predict import router as predict_router
from src.services.scorer import Scorer
from src.db.models import Base
from src.db.database import engine, get_async_session

@asynccontextmanager
async def config(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    app.state.scorer = Scorer(path='model/credit_model.joblib')
    yield
    app.state.scorer = None

app = FastAPI(title="Credit Scoring API", lifespan=config)
app.include_router(predict_router)