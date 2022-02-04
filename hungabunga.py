import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

csvfile = 'RealEstateData.csv'

df = pd.read_csv(csvfile)

df.dropna(axis=0, how='any', inplace=True)

df_train, df_test = tts(df, test_size=0.4, random_state=101)

df_x_train = df_train.drop('MEDV', axis=1)
df_y_train = df_train['MEDV']
df_x_test = df_test.drop('MEDV', axis=1)
df_y_test = df_test['MEDV']

model = AdaBoostRegressor()
model.fit(df_x_train, df_y_train)

scores = cross_val_score(
  estimator=model,
  X=df_x_train,
  y=df_y_train,
  scoring='neg_mean_squared_error'
)
rmse_scores = np.sqrt(-scores)
print(rmse_scores.mean())
