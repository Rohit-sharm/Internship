#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing data sets library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


import warnings


# In[3]:


warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv')
df


# In[5]:


df.head(15)


# In[6]:


# Checking missing value in data set
df.isnull().sum()


# In[7]:


df.shape


# In[8]:


df.columns


# In[11]:


df.columns.tolist()


# In[12]:


df.dtypes


# In[19]:


df.isnull().sum()


# In[21]:


df.info()


# In[22]:


# Visualisation by using heatmap
sns.heatmap(df.isnull())


# In[25]:


df['fixed acidity'].unique()


# In[26]:


df['fixed acidity'].nunique()


# In[27]:


# data analysis and visualization
df.describe()


# In[28]:


# Number of values of each quality
sns.catplot(x='quality', data = df, kind = "count")


# In[29]:


# volatile Acidity v/s Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x= 'quality', y = 'volatile acidity', data = df)
print('volatile acidity is inversely proptional to quality')


# In[30]:


# citric acid v/s Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x= 'quality', y = 'citric acid', data = df)
print('volatile acidity is directly proptional to quality')


# In[31]:


# residual sugar v/s Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x= 'quality', y = 'residual sugar', data = df)


# In[32]:


correlation = df.corr()
correlation


# In[35]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar = True, square = True, annot = True, annot_kws={'size':8},cmap = 'Blues')


# In[36]:


#data prepocessing 
x = df.drop('quality',axis=1)


# In[37]:


print(x)


# In[45]:


#label binarization
Y = df['quality'].apply(lambda y_value: 1 if y_value>= 7 else 0)


# In[47]:


print(Y)


# In[48]:


#train & Test Split


# In[66]:


X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=3)


# In[67]:


print(Y.shape, Y_train.shape, Y_test.shape)


# In[71]:


#Model Training: Random Forest Classifier
model = RandomForestClassifier()


# In[78]:


model.fit(X_train, Y_train)


# In[80]:


# model Evaluation 


# In[82]:


#Accuracy Score
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[83]:


print('Accuracy : ', test_data_accuracy)


# In[84]:


#Building a Predictive System


# In[86]:


input_data = ('7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0,')


# In[87]:


input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)

input_data_as_numpy_array = np.asarray(input_data)


# In[88]:


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')


# In[ ]:




