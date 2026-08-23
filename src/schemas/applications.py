from pydantic import BaseModel, Field
from typing import Optional

class CreditData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: float = Field(0.0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: float
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: float = Field(0.0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    NumberOfDependents: float 

class ModelResponse(BaseModel):
   default_probability: float
   threshold: float
   class_pred: str
   approved: bool