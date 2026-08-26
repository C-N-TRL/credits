import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


def train_and_export():
  print('Обучение начато...')
  df = pd.read_csv('GiveMeSomeCreditTrain.csv', encoding='utf-8').drop(
      columns=['Unnamed: 0']
  )

  df['NAN_found'] = (
      df[['MonthlyIncome', 'NumberOfDependents']]
      .isnull()
      .any(axis=1)
      .astype(int)
  )
  df = df.fillna(df.median(numeric_only=True))
  df.loc[df['age'] == 0, 'age'] = df['age'].median()
  error_cols = [
      'NumberOfTime30-59DaysPastDueNotWorse',
      'NumberOfTime60-89DaysPastDueNotWorse',
      'NumberOfTimes90DaysLate',
  ]
  df[error_cols] = df[error_cols].replace([96, 98], np.nan)
  df[error_cols] = df[error_cols].fillna(df[error_cols].median())

  clip_cols = [
      'RevolvingUtilizationOfUnsecuredLines',
      'DebtRatio',
      'MonthlyIncome',
      'NumberOfOpenCreditLinesAndLoans',
  ]
  quantiles = df[clip_cols].quantile(0.99)
  df[clip_cols] = df[clip_cols].clip(upper=quantiles, axis=1)
  total_past_due = (
      df['NumberOfTime30-59DaysPastDueNotWorse']
      + df['NumberOfTime60-89DaysPastDueNotWorse']
      + df['NumberOfTimes90DaysLate']
  )
  df['HasAnyPastDue'] = (total_past_due > 0).astype(int)

  X = df.drop(columns=['SeriousDlqin2yrs'])
  y = df['SeriousDlqin2yrs']

  best_params = {
      'boosting_type': 'gbdt',
      'random_state': 1,
      'n_estimators': 300,
      'verbosity': -1,
      'n_jobs': -1,
      "learning_rate": 0.07677974079521428,
      "num_leaves": 42,
      "max_depth": 3,
      "min_child_samples": 41,
      "reg_alpha": 6.608993211083366
  }
  model = LGBMClassifier(**best_params)
  model.fit(X, y)
  artifact = {
      'model': model,
      'threshold': 0.2107,
      'features': list(X.columns),
  }
  joblib.dump(artifact, 'credit_model.joblib')
  print('Обучение завершено. Файл credit_model.joblib успешно создан!')


if __name__ == '__main__':
  train_and_export()