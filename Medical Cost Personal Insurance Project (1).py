#!/usr/bin/env python
# coding: utf-8

# In[6]:


#importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[7]:


#loading the data from csv file to a pandas dataframe
df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset4/main/medical_cost_insurance.csv')
df


# In[8]:


df.head()


# In[10]:


df.shape


# In[12]:


df.info()


# In[14]:


# categorical features
#Sex, smoke region


# In[15]:


#checking for missing values
df.isnull().sum()


# In[18]:


# Analysing the data
df.describe()


# In[22]:


# distribution of age value
sns.set()
plt.figure(figsize =(6,6))
sns.distplot(df['age'])
plt.title('Age Distribution')
plt.show


# In[24]:


# checking with gender
plt.figure(figsize =(6,6))
sns.countplot(x='sex',data=df)
plt.title('sex distribution')
plt.show()


# In[27]:


df['sex'].value_counts()


# In[29]:


# bmi distribution in Data sets
plt.figure(figsize =(6,6))
sns.countplot(x='bmi',data=df)
plt.title('bmi distribution')
plt.show()


# In[30]:


#Normal BMI Ranges to 18.5 to 24.9
# children column
plt.figure(figsize =(6,6))
sns.countplot(x='children',data=df)
plt.title('children distribution')
plt.show()


# In[33]:


df['children'].value_counts()


# In[34]:


# smoker distribution in Data sets
plt.figure(figsize =(6,6))
sns.countplot(x='smoker',data=df)
plt.title('bmi distribution')
plt.show()


# In[35]:


df['smoker'].value_counts()


# In[36]:


#Region distribution in Data sets
plt.figure(figsize =(6,6))
sns.countplot(x='region',data=df)
plt.title('region distribution')
plt.show()


# In[37]:


df['region'].value_counts()


# In[39]:


#Charge distribution in Data sets
plt.figure(figsize =(6,6))
sns.distplot(df['charges'])
plt.title('charge distribution')
plt.show()


# In[40]:


df['charges'].value_counts()


# In[41]:


# data preprocessing


# In[44]:


#encoding the data
#encoding sex
df.replace({'sex':{'male':0, 'female':1}}, inplace=True)


# In[45]:


#encoding smoker colunmn
df.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)


# In[46]:


#encoding region
df.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2,'northwest':3}}, inplace=True)



# In[47]:


#spliting the features


# In[48]:


x = df.drop(columns ='charges', axis =1  )


# In[49]:


y =df['charges']


# In[50]:


print(x)


# In[51]:


print(y)


# In[52]:


#spliting the data into training data & test data


# In[57]:


X_train,X_test, Y_train, Y_test = train_test_split(x,y,test_size= 0.2,random_state = 2)


# In[60]:


print(x.shape, X_train.shape, X_test.shape)


# In[61]:


# model training


# In[62]:


#linear regression
regressor = LinearRegression()


# In[64]:


regressor.fit(X_train, Y_train)


# In[66]:


# predicition of training data 
training_data_prediciton = regressor.predict(X_train)


# In[67]:


print(training_data_prediciton )


# In[70]:


# R square value
r2_train = metrics.r2_score(Y_train,training_data_prediciton)
print('R Square vale:', r2_train)


# In[72]:


# predicition of testdata 
test_data_prediciton = regressor.predict(X_test)


# In[73]:


print(test_data_prediciton )


# In[75]:


r2_test = metrics.r2_score(Y_test,training_data_prediciton)
print('R Square vale:', r2_test)


# In[98]:


input_data = (31,1,25.74,0,1,0)


# In[99]:


#changing input data as numpy array
input_data_as_numpy_array = np.array(input_data)


# In[101]:


# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[102]:


prediction = regressor.predict(input_data_reshaped)


# In[103]:


print(prediction)


# In[ ]:





# In[ ]:





# In[ ]:




