from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import JSON, DateTime, Boolean, Float, String
from typing import Any
from datetime import datetime, timezone

class Base(DeclarativeBase):
    pass

class Scoring(Base):
    __tablename__ = 'scoring'

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    input_features: Mapped[dict[str, Any]] = mapped_column(JSON)
    score: Mapped[float] = mapped_column(Float)
    prediction: Mapped[bool] = mapped_column(Boolean)
    model_version: Mapped[str] = mapped_column(String)
