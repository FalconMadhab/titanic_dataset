import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading in the titanic_train.csv file into a pandas dataframe.

train = pd.read_csv('titanic_train.csv')
print(train.head())

#Analyzing Data

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()   #Figure_1

# Roughly 20 percent of the Age data is missing. 

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
plt.show()   #Figure_2


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
plt.show()   #Figure_3

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
plt.show()   #Figure_4

sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
plt.show()   #Figure_5

train['Age'].hist(bins=30,color='darkred',alpha=0.7)
plt.show()   #Figure_6

# Data Cleaning
# Filling missing age data instead of just dropping the missing age data rows.
# I have checked using average age of Pclass

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
plt.show()   #Figure_7


#Wealthier passengers in the higher classes tend to be older. 
#These average age values is used to impute based on Pclass for Age.

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#Checking the Heatmap again for missing data(part of data cleaning)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Dropping the Cabin column and the row in Embarked that is NaN.

train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

#Converting Categorical Features 
#Converting categorical features to dummy variables using pandas! 
#Otherwise the machine learning algorithm won't be able to directly take in those features as inputs.

print(train.info())

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
print(train.head())
#Data ready for model

#Building a Logistic Regression model 
#Splitting our data into a training set and test set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


#Training and Predicting

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

#Evaluation

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

#Output of Classification Report
'''
             precision    recall  f1-score   support

          0       0.81      0.93      0.86       163
          1       0.85      0.65      0.74       104

avg / total       0.82      0.82      0.81       267
'''