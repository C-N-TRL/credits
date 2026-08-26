from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('GiveMeSomeCredit-training.csv', encoding='utf-8')
train, prod = train_test_split(
    df,
    test_size=0.3,
    random_state=1,
    stratify=df['SeriousDlqin2yrs']
)
train.to_csv('GiveMeSomeCreditTrain.csv', index=False)
prod.to_csv('GiveMeSomeCreditProd.csv', index=False)