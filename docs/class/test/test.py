import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X_test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/dataset/main/used_car_X_test.csv')
X_train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/dataset/main/used_car_X_train.csv')
y_train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/dataset/main/used_car_y_train.csv')
# print(X_test.shape, X_train.shape)

id_test = X_test['id'].copy()

# print(X_train.head())
# print(y_train.head())
# print(X_test.head())

X_test = pd.get_dummies(X_test)
X_train = pd.get_dummies(X_train)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# print(X_test.shape, X_train.shape)
# print(X_test.head())

train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# print(train_X.shape, val_X.shape, train_y.shape, val_y.shape)

rf = RandomForestRegressor()
rf.fit(train_X, train_y)

prediction_val = rf.predict(val_X)
# print(prediction_val)

rmse = np.sqrt(mean_squared_error(val_y, prediction_val))
# print(rmse)

# print(X_train)
# print(X_test)
# print(X_test.shape, X_train.shape)

y_pred = rf.predict(X_test)

# 저장해둔 id_test 사용
df = pd.DataFrame({'id': id_test, 'price': y_pred})

print(df.head())