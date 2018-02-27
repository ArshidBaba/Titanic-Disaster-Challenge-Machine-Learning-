# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:38:54 2018

@author: Arshid
"""
#Importing all the needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
#cf.go_offline()

#Importing the Training set
train=pd.read_csv('C:/Users/Arshid/Desktop/Titanic/train.csv')


#Visualizing the null values using HeatMaps
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')

#Visualizing Survivors based on their Passenger Classes
sns.countplot(x='Survived',hue='Pclass',data=train)

#Plotting on the bases of the Age of the Passengers
sns.distplot(train['Age'].dropna(), kde=False, bins=30)

#Counting the number of Passengers who had boarded with their siblings and/or their spouses
sns.countplot(x='SibSp', data=train)

#Plotting Fare against the number of Passengers
train['Fare'].plot.hist(bins=40, figsize=(10,4))

#train['Fare'].iplot(kind='hist',bins=50)
#Getting the Age of the Passengers on the basis of their Class, also the average Age per class
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=train)

#Defing a  function that will impute the Age columns on the basis of the Pclass
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

        
#Calling the above defined function to impute the Age column
train['Age']=train[['Age', 'Pclass']].apply(impute_age, axis=1)  

#Visualizing after the Imputation of the Age column
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')  

#Dropping the Cabin column since it doesn't have any direct effect on the prediction
train.drop('Cabin', axis=1, inplace=True)   

#Checking the dataset
train.head()

#Using HeatMap again to visualize the resultant set
plt.figure(figsize=(10,7))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#Dropping the remaining rows with null values in them
train.dropna(inplace=True)

#Dealing with categorical column Sex, Embarked, making dummmies
sex=pd.get_dummies(train['Sex'], drop_first=True)
embark=pd.get_dummies(train['Embarked'], drop_first=True)
classes=pd.get_dummies(train['Pclass'])

#Concatinating the newly created dummy columns with the existing dataframe
train=pd.concat([train,sex,embark], axis=1)
train=pd.concat([train,classes], axis=1)

#Dropping the those columns which will not be used during training
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.drop('PassengerId', axis=1, inplace=True)
train.drop('Pclass', axis=1, inplace=True)

#Importing the Test set
test=pd.read_csv('C:/Users/Arshid/Desktop/Titanic/test.csv')


#Visulazing missing values using HeatMap
plt.figure(figsize=(18,12))
sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='viridis')

#Using BoxPlot to see which age group belongs to which Pclass
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=test)

#Defining a function which will be used to impute Age columns of the Test set
def impute_age2(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 42
        elif Pclass==2:
            return 26
        else:
            return 24
    else:
        return Age
#Imputing the Age column by calling the above function        
test['Age']=test[['Age', 'Pclass']].apply(impute_age2, axis=1)

#Imputing Fare column for missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(test.iloc[:, 8:9])
test.iloc[:, 8:9] = imputer.transform(test.iloc[:, 8:9])


#Using HeatMap again to visualize the remaining missing values from the set
plt.figure(figsize=(10,7))
sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#Dropping the Cabin column as done in the Training set
test.drop('Cabin', axis=1, inplace=True)

#Dropping the rows that had missing values in their columns
#test.dropna(inplace=True)

#Visualizing again to see the effect after Imputation
plt.figure(figsize=(10,5))
sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#Creating Dummy variables for categorical feaatures of the set and concatinating them to the Test set
sex2=pd.get_dummies(test['Sex'], drop_first=True)
embark2=pd.get_dummies(test['Embarked'], drop_first=True)
test=pd.concat([test,sex2,embark2], axis=1)
pclasses=pd.get_dummies(test['Pclass'])
test=pd.concat([test,pclasses], axis=1)

#Saving PassengerId before dropping it
result=test.iloc[:,0]

#Dropping the redundant features which were converted to dummies earlier
test.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)
test.drop('Pclass', axis=1, inplace=True)

#Building a Logistic Regression Model

#Train Test Split
x_train=train.iloc[:,1:]
y_train=train.iloc[:,0:1]
x_test=test.iloc[:,:]

#Training and Predicting 

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)

y_pred= logreg.predict(x_test)

#Writing the predictions to a csv file
df=pd.DataFrame(dict(PassengerId = result, Survived = y_pred)).reset_index()
df.drop('index', axis=1, inplace=True)

df.to_csv('result5.csv', index=False)

