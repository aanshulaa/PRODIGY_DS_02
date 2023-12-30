#!/usr/bin/env python
# coding: utf-8

# In[ ]:




 EDA with Python and appying Logistic Regression
# IMPORT LIBRARIES
# Let's import some libraries to get started!

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# The Data
# Let's start bu reading in the titanic_train.csv files into a pandas dataframe

# In[7]:


import pandas as pd

train = pd.read_csv("C:\\Users\\kansh\\OneDrive\\Desktop\\PRODIGY\\titanic\\train.csv")

# Now you can proceed with the rest of your code
train.head()


# Exploratory Data Analysis
# Let's begin some exploratory data analysis! We will start by checing out missing data!

# Missing data
# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[9]:


train.isnull()


# In[11]:


import pandas as pd
import seaborn as sns

train = pd.read_csv("C:\\Users\\kansh\\OneDrive\\Desktop\\PRODIGY\\titanic\\train.csv")

# Now you can use seaborn functions
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[13]:


import pandas as pd
import seaborn as sns

train = pd.read_csv("C:\\Users\\kansh\\OneDrive\\Desktop\\PRODIGY\\titanic\\train.csv")

# Print column names
print(train.columns)

# Modify countplot based on column names and case sensitivity
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)  # Adjust column name as needed


# In[14]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')


# In[16]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue= 'Pclass', data=train, palette='rainbow')


# In[17]:


sns.distplot(train['Age'].dropna(), kde=False, color= 'darkred', bins=40)


# In[18]:


train['Age'].hist(bins=30, color='darkred', alpha=0.3)


# In[22]:


import pandas as pd
import seaborn as sns

train = pd.read_csv("C:\\Users\\kansh\\OneDrive\\Desktop\\PRODIGY\\titanic\\train.csv")

# Print column names
print(train.columns)

# Modify countplot based on column names and case sensitivity
sns.set_style('whitegrid')
sns.countplot(x='SibSp', data=train)  # Adjust column name as needed


# In[25]:


train['Fare'].hist(color='green', bins=40, figsize=(8.4, 4.8))


# 

# Data Cleaning

# In[29]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='Set1')


# In[41]:


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


# In[44]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[45]:


train.drop('Cabin', axis=1, inplace = True)


# In[46]:


train.head()


# In[47]:


train.dropna(inplace=True)


# Converting Categorical Features

# In[48]:


train.info()


# In[49]:


pd.get_dummies(train['Embarked'], drop_first=True).head()


# In[51]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)


# In[52]:


train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis =1, inplace = True)


# In[53]:


train.head()


# In[54]:


train = pd.concat([train, sex, embark], axis = 1)


# In[55]:


train.head()


# Building a logistic Regression model

# In[56]:


train.drop('Survived', axis =1).head()


# In[57]:


train['Survived'].head()


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


X_train, X_test, Y_train, Y_test = train_test_split(train.drop('Survived', axis=1), train['Survived'], test_size=0.30, random_state=101)

