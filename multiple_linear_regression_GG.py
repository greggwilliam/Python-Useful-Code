# Data Preprocessing

#Importing the libraries
import numpy as np

#Needed for all maths things...use it for everything
import matplotlib.pyplot as plt
#Used for plotting graphs!
import pandas as pd
#Used for importing and managing data sets

#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Remove one dummy variable
X = X[:, 1:]

# Splitting the dataset into the Trainign set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting any multi-linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination (add column of 1s to X first)
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis =1)
X_opt = X[:, [0,1,2,3,4,5]]

# Significance level is 0.05

regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# Remove highest P-value variable and rerun (variable 2)
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# Remove highest P-value variable and rerun (variable 1)
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# Remove highest P-value variable and rerun (variable 4)
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# Remove highest P-value variable and rerun (variable 5)
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# Automatic backwards elimination using only p-values
"""import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)"""

# Automatic backwards elimination using p-values and adjusted R squared

"""import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)"""




