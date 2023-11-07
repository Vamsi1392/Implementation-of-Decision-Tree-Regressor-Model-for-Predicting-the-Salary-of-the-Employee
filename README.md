# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.


## Program:
```

Developed by: Vishwa Vasu R
RegisterNumber: 212222040183 

import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
  
*/

```


## Output:

data.head()


![Screenshot (87)](https://github.com/MaheshMuthuL/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/135570619/1ffe0340-c9bb-42f4-94b9-9a7e5685476a)



data.info()





![Screenshot (88)](https://github.com/MaheshMuthuL/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/135570619/2c01c8da-ceda-4526-884c-90fe69fba2f6)







isnull() and sum()





![Screenshot (89)](https://github.com/MaheshMuthuL/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/135570619/c881dcb6-f1ee-44df-add5-7a31bf64b62e)








data.head() for salary





![Screenshot (90)](https://github.com/MaheshMuthuL/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/135570619/a8aec5f0-b3e7-4bc5-9314-e00e4354dd76)







MSE value




![Screenshot (91)](https://github.com/MaheshMuthuL/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/135570619/a41eb798-9949-4f4e-a378-ab125866932a)







r2 value





![Screenshot (92)](https://github.com/MaheshMuthuL/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/135570619/9f79b3e4-12d1-473a-b1d3-ebe00cd8c753)





data prediction






![Screenshot (93)](https://github.com/MaheshMuthuL/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/135570619/512a4cd9-8073-4167-9149-9527af26a22e)










## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
