# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the car price dataset, select relevant numerical features (enginesize, horsepower, citympg, highwaympg) as input variables, and set price as the target variable. Split the data into training and testing sets.
2. Apply standardization to the training features using StandardScaler and transform the testing features using the same scaler to ensure consistent feature scaling.
3. Train a Linear Regression model using the scaled training data, predict prices for the test data, and evaluate model performance using MSE, RMSE, and R-squared metrics along with model coefficients.
4. Check linearity using actual vs predicted plots, test independence of errors using the Durbin–Watson statistic, assess homoscedasticity through residual plots, and verify normality of residuals using histogram and Q–Q plots. 

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#Load data
df = pd.read_csv('encoded_car_data (1).csv')
print(df.head())

#Select features & target
X = df.drop('price',axis=1)
y = df['price']
print(df.head())

#Select features & target
X = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']

#Split data
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=42)

#1. Linear Regression (with scaling)
lr = Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(X_train, y_train)
y_pred_linear = lr.predict(X_test)

#2.Polynomial Regression(degree =2)
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('model',LinearRegression())
])

poly_model.fit(X_train,y_train)
y_pred_poly = poly_model.predict(X_test)

# Evaluate models
print('Name: Saranya R')
print('Reg. No: 212225040384')
print('Linear Regression:')
mse = mean_squared_error(y_test, y_pred_linear)
print('MSE= ',mean_squared_error(y_test,y_pred_linear))
r2score = r2_score(y_test,y_pred_linear)
print('MAE= ',mean_absolute_error(y_test,y_pred_linear))
r2score = r2_score(y_test,y_pred_linear)

print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R2: {r2_score(y_test, y_pred_poly):.2f}")

#plot actual vs predicted
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_linear, label='Linear',alpha=0.6)
plt.scatter(y_test, y_pred_poly, label='Polynomial (degree=2)',alpha=0.6)
plt.plot([y.min(),y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
```

## Output:

<img width="352" height="209" alt="image" src="https://github.com/user-attachments/assets/0e41b56b-e006-4e1c-9eea-e92397aff788" />
<img width="1275" height="612" alt="image" src="https://github.com/user-attachments/assets/e6a9b54b-4897-48da-a99e-cfbdcaa97781" />



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
