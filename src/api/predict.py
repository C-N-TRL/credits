from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.crud import create_scoring_log, get_scoring_logs
from src.db.database import get_async_session
from src.schemas.applications import CreditData, ModelResponse

router = APIRouter()

@router.post('/predict', response_model=ModelResponse)
async def predict(
    data: CreditData, 
    request: Request, 
    db: AsyncSession = Depends(get_async_session)
) -> ModelResponse:
    payload = data.model_dump(by_alias=True)
    scorer = request.app.state.scorer
    result = scorer.predict(payload)
    
    await create_scoring_log(
        db=db,
        input_features=payload,
        score=result['default_probability'],
        prediction=result['approved'],
        model_version='v1'
    )
    return result


@router.get('/history')
async def get_history(
    limit: int = 100,
    db: AsyncSession = Depends(get_async_session)
):
    return await get_scoring_logs(db=db, limit=limit)