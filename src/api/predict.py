from fastapi import APIRouter, Request, HTTPException
from src.schemas.applications import ModelResponse, CreditData

router = APIRouter()

@router.post('/predict', response_model=ModelResponse)
def predict(data: CreditData, request: Request) -> ModelResponse:
    scorer = getattr(request.app.state, "scorer", None)
    if not scorer:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    input_data = data.model_dump(by_alias=True)
    return scorer.predict(input_data)