#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import seaborn as sns
color=sns.color_palette()
sns.set_style('darkgrid')

import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn =ignore_warn #warnings from sklearn and seaborn


# In[2]:


df_train=pd.read_csv("AdvWorksCusts.csv")
ave_spend=pd.read_csv("AW_AveMonthSpend.csv")
bike_buyer=pd.read_csv('AW_BikeBuyer.csv')
df_test=pd.read_csv('AW_test.csv')


# In[3]:


df_train.head(5).T


# In[4]:


ave_spend.shape


# In[5]:


bike_buyer.head(5).T


# In[6]:


bike_buyer.shape


# In[7]:


df_test.head(5).T


# In[8]:


print('Shape before merging Data:', df_train.shape)
df_train=pd.merge(df_train,bike_buyer,how='inner', on='CustomerID')
print('Shape After merging Data:', df_train.shape)


# In[9]:


print('Shape before dropping duplicates', df_train.shape)
df_train.drop_duplicates(subset='CustomerID', keep='last')
print('Shape after dropping duplicates', df_train.shape)


# In[10]:


df_train.head(5).T


# In[11]:


df_train.describe()


# In[12]:


#Heat map
corrmat= df_train.corr()
f, ax =plt.subplots(figsize=(5,4))
sns.heatmap(corrmat, square=True)


# In[13]:


sns.distplot(bike_buyer['BikeBuyer'])


# In[14]:


df_train.describe(include=[np.object, pd.Categorical]).T


# In[15]:


df_train['Occupation'].value_counts()


# In[32]:


##Obtaining the average for the features in the Occupation column


# In[33]:


df_train["BirthDate"]=pd.to_datetime(df_train['BirthDate'], infer_datetime_format=True)
df_train['year']=df_train['BirthDate'].dt.year


# In[34]:


df_train['Age']=1998-df_train['year']
df_train.head(5).T


# In[35]:


df_test["BirthDate"]=pd.to_datetime(df_test['BirthDate'], infer_datetime_format=True)
df_test['year']=df_test['BirthDate'].dt.year
df_test['Age']=1998-df_test['year']
df_test.head(5).T


# In[36]:


cars={0:'No cars', 1:'>1 cars', 2:'>1 cars', 3:'>3 cars', 4:'>3 cars'}
dataset=[df_train, df_test]
for data in dataset:
    data['NumberCarsOwned']=data['NumberCarsOwned'].replace(cars)


# In[37]:


children={0:'No Children', 1:'More than 1', 2:'More than 1', 3:'More than 1', 4:'More than 1', 5:'More than 1'}
dataset=[df_train, df_test]
for data in dataset:
    data['NumberChildrenAtHome']=data['NumberChildrenAtHome'].replace(children)


# In[38]:


df_train.head(5)


# In[39]:


dataset=[df_train, df_test]
for data in dataset:
    del data['year']


# In[40]:


df_train.columns


# In[41]:


child={0:'No Children', 1:'More than 1', 2:'More than 1', 3:'More than 1', 4:'More than 1', 5:'More than 1'}
dataset=[df_train, df_test]
for data in dataset:
    data['TotalChildren']=data['TotalChildren'].replace(child)


# In[42]:


columns=['CustomerID','CountryRegionName', 'Education', 'Occupation', 'Gender', 'MaritalStatus', 
         'HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome','BikeBuyer', 'Age']
train=df_train[columns]

cols=['CustomerID','CountryRegionName', 'Education', 'Occupation', 'Gender', 'MaritalStatus', 
      'HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome', 'Age']
test=df_test[cols]


# In[43]:


test.head(5)


# In[44]:


train.head(5)


# In[45]:


train.shape


# In[46]:


##Observing Data to perform Data Transformation


# In[47]:


column=['YearlyIncome', 'Age']
def distplot(df, column, bins = 10, hist = False):
    for col in column:
        sns.distplot(df[col], bins=bins, rug=True, hist=hist)
        plt.title('Distribution for' + col)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

distplot(train, column, hist= True)


# In[48]:


train['Age']=np.log(train['Age'])
train['YearlyIncome']=(train['YearlyIncome'])**0.5


# In[49]:


test['Age']=np.log(test['Age'])
test['YearlyIncome']=(test['YearlyIncome'])**0.5


# In[50]:


train.head()


# In[51]:


train['CountryRegionName'].value_counts()


# In[52]:


test['CountryRegionName'].value_counts()


# In[53]:


col=['CountryRegionName', 'Education', 'Occupation', 'Gender', 'MaritalStatus','NumberCarsOwned', 
        'NumberChildrenAtHome','TotalChildren']
train=pd.get_dummies(train, prefix=col, columns=col)
test=pd.get_dummies(test, prefix=col, columns=col)


# In[54]:


train.columns


# In[55]:


print('Shape of train', train.shape)
print('Shape of test', test.shape)


# In[56]:


f,ax=plt.subplots(figsize=(18,15))
sns.heatmap(train.corr(),linewidth=2.0, ax=ax, annot=True)
ax.set_title('Correlation Matrix')


# In[57]:


##Splitting and Applying Algorithm


# In[58]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sklearn.metrics as sklm
from math import sqrt


# In[59]:


feature_col=['HomeOwnerFlag', 'YearlyIncome', 'Age',
       'CountryRegionName_Australia', 'CountryRegionName_Canada',
       'CountryRegionName_France', 'CountryRegionName_Germany',
       'CountryRegionName_United Kingdom', 'CountryRegionName_United States',
       'Education_Bachelors ', 'Education_Graduate Degree',
       'Education_High School', 'Education_Partial College',
       'Education_Partial High School', 'Occupation_Clerical',
       'Occupation_Management', 'Occupation_Manual', 'Occupation_Professional',
       'Occupation_Skilled Manual', 'Gender_F', 'Gender_M', 'MaritalStatus_M',
       'MaritalStatus_S', 'NumberCarsOwned_>1 cars', 'NumberCarsOwned_>3 cars',
       'NumberCarsOwned_No cars', 'NumberChildrenAtHome_More than 1',
       'NumberChildrenAtHome_No Children', 'TotalChildren_More than 1',
       'TotalChildren_No Children']
predicted_class_names=['BikeBuyer']
X=train[feature_col].values
y=train[predicted_class_names].values 
split_test_size=0.30
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=split_test_size, random_state=42)


# In[60]:


print("{0:0.2f}% in training set".format((len(X_train)/len(train.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(train.index)) * 100))


# In[62]:


column=['HomeOwnerFlag', 'YearlyIncome', 'Age',
       'CountryRegionName_Australia', 'CountryRegionName_Canada',
       'CountryRegionName_France', 'CountryRegionName_Germany',
       'CountryRegionName_United Kingdom', 'CountryRegionName_United States',
       'Education_Bachelors ', 'Education_Graduate Degree',
       'Education_High School', 'Education_Partial College',
       'Education_Partial High School', 'Occupation_Clerical',
       'Occupation_Management', 'Occupation_Manual', 'Occupation_Professional',
       'Occupation_Skilled Manual', 'Gender_F', 'Gender_M', 'MaritalStatus_M',
       'MaritalStatus_S', 'NumberCarsOwned_>1 cars', 'NumberCarsOwned_>3 cars',
       'NumberCarsOwned_No cars', 'NumberChildrenAtHome_More than 1',
       'NumberChildrenAtHome_No Children', 'TotalChildren_More than 1',
       'TotalChildren_No Children']
test1=test[column]


# In[63]:


print(X_train.shape)
print(test1.shape)


# In[64]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
test1=ss.transform(test1)


# In[65]:


reg=LogisticRegression()
reg.fit(X_train, y_train)


# In[66]:


def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])

def print_metrics(labels, probs, threshold):
    scores = score_model(probs, threshold)
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy        %0.2f' % sklm.accuracy_score(labels, scores))
    print('AUC             %0.2f' % sklm.roc_auc_score(labels, probs[:,1]))
    print('Macro precision %0.2f' % float((float(metrics[0][0]) + float(metrics[0][1]))/2.0))
    print('Macro recall    %0.2f' % float((float(metrics[1][0]) + float(metrics[1][1]))/2.0))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])
    
probabilities = reg.predict_proba(X_test)
print_metrics(y_test, probabilities, 0.5)   


# In[67]:


solution=reg.predict(test1)
np.savetxt('LogisticRegressionMicrosoft.csv', solution, delimiter=',')


# In[68]:


gbr=GradientBoostingClassifier()
gbr.fit(X_train, y_train.ravel())


# In[69]:


plt.bar(range(len(gbr.feature_importances_)), gbr.feature_importances_)
plt.title('Feature Importance')
plt.show()


# In[70]:


probabilities = gbr.predict_proba(X_test)
print_metrics(y_test, probabilities, 0.5)  


# In[71]:


solution=gbr.predict(test1)
my_submission=pd.DataFrame({'CustomerID':test.CustomerID,'BikeBuyer': solution})
my_submission.to_csv('GradientBoostingMicrosoft01.csv', index=False)


# In[72]:


import xgboost as xgb
xgb=xgb.XGBClassifier()
xgb.fit(X_train, y_train)


# In[73]:


plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
plt.show()


# In[74]:


probabilities = xgb.predict_proba(X_test)
print_metrics(y_test, probabilities, 0.5)  


# In[75]:


solution=xgb.predict(test1)
my_submission=pd.DataFrame({'CustomerID':test.CustomerID,'BikeBuyer': solution})
my_submission.to_csv('XgboostClassifierMicrosoft01.csv', index=False)


# In[76]:


from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier()
mlp.fit(X_train, y_train)


# In[77]:


probabilities = mlp.predict_proba(X_test)
print_metrics(y_test, probabilities, 0.5)  


# In[78]:


solution=mlp.predict(test1)
my_submission=pd.DataFrame({'CustomerID':test.CustomerID,'BikeBuyer': solution})
my_submission.to_csv('NeuralNetworkClassifierMicrosoft01.csv', index=False)


# In[79]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)


# In[80]:


plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
plt.show()


# In[81]:


probabilities = rf.predict_proba(X_test)
print_metrics(y_test, probabilities, 0.5)  


# In[82]:


solution=rf.predict(test1)
my_submission=pd.DataFrame({'CustomerID':test.CustomerID,'BikeBuyer': solution})
my_submission.to_csv('RandomForestMicrosoft01.csv', index=False)


# In[ ]:




