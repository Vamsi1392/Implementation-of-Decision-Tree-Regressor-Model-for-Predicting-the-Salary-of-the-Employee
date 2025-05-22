# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data -Collect and clean data on employee salaries and features

2.split data into training and testing sets

3.Define your model -Use a Decision Tree Regressor to recursively partition data based on input features

4.Determine maximum depth of tree and other hyperparameters

5.Train your model -Fit model to training data -Calculate mean salary value for each subset

6.Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance
Tune hyperparameters -Experiment with different hyperparameters to improve performance

7.Deploy your model Use model to make predictions on new data in real-world application.
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: M.V.Vamsidhar Reddy
RegisterNumber:  212224040205
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```
## Output:

# Initial dataset:
![image](https://github.com/user-attachments/assets/c161449e-5001-4438-a74b-d6e15ae014bd)
