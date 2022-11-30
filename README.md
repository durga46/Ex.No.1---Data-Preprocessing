## EX.NO.1
## DATE: 08.09.2022
# <p align="center">  Data-Preprocessing</p>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.

<br>
<br>
<br>

## ALGORITHM:
1. Importing the libraries
2. Importing the dataset
3. Taking care of missing data
4. Encoding categorical data
5. Normalizing the data
6. Splitting the data into test and train

## PROGRAM:
```python
import pandas as pd
import numpy as np
df = pd.read_csv("/content/Churn_Modelling.csv")
df.info()
df.isnull().sum()
df.duplicated()
df.describe()
df['Exited'].describe()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df.copy()
df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))
df1
df1.describe()
X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)
y = df1.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("Size of X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))
X_train.shape
```



<br>
<br>
<br>
<br>
<br>
<br>

## OUTPUT:
### Dataset
![op1](https://user-images.githubusercontent.com/75235704/191728139-8b634c41-6f69-4a91-9ea9-29bb9fb0d440.png)

### Checking for Null Values
![op2](https://user-images.githubusercontent.com/75235704/191728291-3d272a28-51fb-463b-87f8-fae13a5b1969.png)
<br>
<br><br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br>
### Checking for duplicate values
![o3](https://user-images.githubusercontent.com/75235704/191728419-94a14090-eec4-426e-b15e-5d639bdef583.png)

### Describing Data
![o5](https://user-images.githubusercontent.com/75235704/191728554-021bb47b-1533-4f85-b1b1-f7d3d2273b97.png)
### Checking for outliers in Exited Column
![o6](https://user-images.githubusercontent.com/75235704/191728638-8eb5f8ae-d41a-415f-91d2-7313c7208ea8.png)
<br><br>
<br>
<br>
<br>
<br>
<br>
<br>
### Normalized Dataset
![o7](https://user-images.githubusercontent.com/75235704/191728729-614e91ae-a171-427b-9bc7-26c991b91018.png)
### Describing Normalized Data
![o8](https://user-images.githubusercontent.com/75235704/191728804-2df579e0-b1ba-49dc-b997-77ac3e965c53.png)
### X - Values
![o9](https://user-images.githubusercontent.com/75235704/191728909-6e02c07a-85fe-486f-8692-b8d01a93453d.png)
### Y - Value
![o10](https://user-images.githubusercontent.com/75235704/191728999-035b8d96-4ed3-47d4-abaf-93bef67fbf0d.png)
### X_train values
![o11](https://user-images.githubusercontent.com/75235704/191729078-7d4680bc-923d-42a4-a478-5f83b41557b0.png)
### X_train Size
![012](https://user-images.githubusercontent.com/75235704/191729188-a66c3986-9108-4a87-bc0d-02a2d9f715c4.png)
### X_test values
![o13](https://user-images.githubusercontent.com/75235704/191729272-c6b4e916-e53a-4300-8a29-4b6ef9180a7d.png)
### X_test Size
![014](https://user-images.githubusercontent.com/75235704/191729366-ce53b180-b9bb-41b3-a484-9a8780cb0fba.png)
### X_train shape
![o15](https://user-images.githubusercontent.com/75235704/191729464-bb65cc05-12c1-4966-94b8-77025b284123.png)



## RESULT
Data preprocessing is performed in a data set downloaded from Kaggle.
