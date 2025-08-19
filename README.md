# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date:19-08-2025
### NAME:GEDIPUDI DARSHANI
### REGISTER NO:212223230062
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
A - LINEAR TREND ESTIMATION
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data=pd.read_csv('/content/india.csv')
data.head()
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')
data['date'] = data['date'].apply(lambda x: x.toordinal())
X = data['date'].values.reshape(-1, 1)
y = data['open'].values
linear_model = LinearRegression()
linear_model.fit(X, y)
data['Linear_Trend'] = linear_model.predict(X)
plt.figure(figsize=(10,6))
plt.plot(data['date'], data['open'],label='Original Data')
plt.plot(data['date'], data['Linear_Trend'], color='orange', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('date')
plt.ylabel('open')
plt.legend()
plt.grid(True)
plt.show()
```
B- POLYNOMIAL TREND ESTIMATION
```
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
data['open'] = poly_model.predict(X_poly)
plt.figure(figsize=(10,6))
plt.bar(data['date'], data['open'], label='Original Data', alpha=0.6)
plt.plot(data['date'], data['open'],color='yellow', label='Poly Trend(Degree 2)')
plt.title('Polynomial Trend Estimation')
plt.xlabel('date')
plt.ylabel('open')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT
A - LINEAR TREND ESTIMATION
<img width="1363" height="784" alt="image" src="https://github.com/user-attachments/assets/155fd01b-ed9e-4ba1-b065-28bb157075c8" />


B- POLYNOMIAL TREND ESTIMATION
<img width="1447" height="750" alt="image" src="https://github.com/user-attachments/assets/b64392d2-da42-4d73-9f82-2e8753cd0d85" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
