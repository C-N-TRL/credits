from datetime import datetime
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Scoring


async def create_scoring_log(
    db: AsyncSession,
    input_features: dict,
    score: float,
    prediction: bool,
    model_version: str,
    timestamp: datetime | None = None,
) -> Scoring:
    log_data = {
        "input_features": input_features,
        "score": score,
        "prediction": prediction,
        "model_version": model_version,
    }
    log = Scoring(**log_data)
    
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


async def get_scoring_logs(
    db: AsyncSession,
    limit: int = 100
) -> Sequence[Scoring]:
    query = select(Scoring).order_by(Scoring.timestamp.desc()).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()