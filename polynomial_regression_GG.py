# Polynomial Regression Model - Deciding if the a new employee is lying about salary.

#Importing the libraries
import numpy as np
#Needed for all maths things...use it for everything
import matplotlib.pyplot as plt
#Used for plotting graphs!
import pandas as pd
#Used for importing and managing data sets

#Importing the dataset - Need to specify that the X is a matrix using the :
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Don't use a training and test set for this one because only 10 data points and we want accuracy

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = "black")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Years Experience vs Salary using a Linear Regression")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = "black")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "green")
plt.title("Years Experience vs Salary using a Polynomial Regression")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()
