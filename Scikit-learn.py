#!/usr/bin/env python
# coding: utf-8

# ## An End-to-end Scikit-Learn workflow

# # Introduction to Scikit-learn(sklearn)
# 
# #### This notebook demonstrates some of the most useful functions of the beautiful Scikit-learn library.
# 
# ### Workflow
# #### 1. Getting data ready
# #### 2. Choose the right estimator/model/algorithm for our problems
# #### 3. Fit the model and use it to make predictions on our data
# #### 4. Evaluating a model
# #### 5. Improve a model
# #### 6. Save and load a trained model
# #### 7. Putting it all together

# In[7]:


# 1. Get the data ready
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
heart_disease = pd.read_csv("../Additions/heart-disease.csv")
heart_disease


# In[8]:


# Create X (features matrix)
X = heart_disease.drop("target", axis=1)

# Create Y labels
Y = heart_disease["target"]


# In[9]:


## 2. Choose the right model and hyperparameters
## Hyperparameters are like dials in the model that we can tune to make model perform better or worse
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

## We'll keep the default hyperparameters
clf.get_params()


# In[10]:


## 3. Fit the model to the training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[11]:


clf.fit(X_train, y_train)


# In[13]:


## 3. Make a prediction
y_preds = clf.predict(X_test)
y_preds


# In[14]:


## 4. Evaluate the model
clf.score(X_train, y_train)


# In[15]:


clf.score(X_test, y_test)


# In[19]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_preds))


# In[20]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_preds))


# In[25]:


## 5. Improve a model
## Try different amount of n_estimators

np.random.seed(42)
for i in range(10, 110, 10):
    print(f"Trying model with {i} estimators...")
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    print(f"Model accuracy on test set: {clf.score(X_test, y_test) * 100}%")
    print("")
    


# In[26]:


## 6. Save a model and load it
import pickle

pickle.dump(clf, open("random_forest_model_1.pkl", "wb"))


# In[27]:


loaded_model = pickle.load(open("random_forest_model_1.pkl", "rb"))
loaded_model.score(X_test, y_test)


# In[ ]:




