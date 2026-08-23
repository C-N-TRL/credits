import joblib
import pandas as pd
from typing import Any, Dict

class Scorer:
    def __init__(self, path: str):
        artifacts = joblib.load(path)
        self.model = artifacts['model']
        self.threshold = artifacts.get('threshold', 0.2107)
        self.feature_names = artifacts.get('features', None)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([input_data])
        df.columns = df.columns.str.replace('_', '-', regex=False)

        df['NAN_found'] = (
            df[['MonthlyIncome', 'NumberOfDependents']]
            .isnull()
            .any(axis=1)
            .astype(int)
        )

        past_due_cols = [
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ]
        for col in past_due_cols:
            if col not in df.columns:
                df[col] = 0.0

        total_past_due = df[past_due_cols].sum(axis=1)
        df['HasAnyPastDue'] = (total_past_due > 0).astype(int)

        if self.feature_names:
            df = df.reindex(columns=self.feature_names)
        prob = float(self.model.predict_proba(df)[:, 1][0])
        is_default = prob >= self.threshold

        return {
            'default_probability': round(prob, 4),
            'threshold': self.threshold,
            'class_pred': 'Отказано' if is_default else 'Одобрено',
            'approved': not is_default
        }