#!/usr/bin/env python
# coding: utf-8

# In[27]:


pip install imblearn


# In[28]:


# check version number
import imblearn
print(imblearn.__version__)


# In[37]:


import pandas as pd
import seaborn as sns
from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


# In[30]:


X = pd.read_csv('X_vector.csv',index_col=0)
y = pd.read_csv('y.csv',index_col=0, dtype=int)


# In[31]:


X


# In[32]:


y


# In[33]:


# label encode the target variable
y_enc = LabelEncoder().fit_transform(y)


# In[42]:


# Sort the counter dictionary by values (class counts) in descending order
sorted_counter = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))

# Print the sorted class distribution summary
for k, v in sorted_counter.items():
    per = v / len(y_enc) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

# List of shades of blue for the plot
color_palette = ['lightblue', 'skyblue', 'dodgerblue', 'mediumblue', 'darkblue']

# Sort the color_palette based on the sorted_counter keys
sorted_colors = [color_palette[k] for k in sorted_counter.keys()]

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(sorted_counter.keys(), sorted_counter.values(), color=sorted_colors)

# Adding labels and title
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Class Distribution')

# Show the plot
plt.show()


# # **SMOTE with equal class weights**

# **Training set**

# In[45]:


# Create an instance of SMOTE
oversample = SMOTE()

# Perform oversampling
X_sm, y_sm = oversample.fit_resample(X, y_enc)

# Calculate the class distribution of the oversampled dataset
counter_oversampled = Counter(y_sm)

# Sort the oversampled counter dictionary by keys (class numbers) in ascending order
sorted_counter_oversampled = dict(sorted(counter_oversampled.items()))

# Print the sorted class distribution summary for the oversampled dataset
for k, v in sorted_counter_oversampled.items():
    per = v / len(y_sm) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

# List of shades of blue for the plot
color_palette = ['lightblue', 'skyblue', 'dodgerblue', 'mediumblue', 'darkblue']

# Sort the color_palette based on the sorted_counter_oversampled keys
sorted_colors = [color_palette[k] for k in sorted_counter_oversampled.keys()]

# Create the bar plot for the oversampled dataset
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(sorted_counter_oversampled.keys(), sorted_counter_oversampled.values(), color=sorted_colors)

# Adding labels and title
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Class Distribution (Oversampled)')

# Show the plot
plt.show()


# In[10]:


original_size = len(X)
original_size


# In[11]:


smote_size = len(X_sm)
smote_size


# In[12]:


X_sm = pd.DataFrame(X_sm)
y_sm = pd.DataFrame(y_sm)


# In[13]:


X_sm.to_csv('X_smote_equal_weights.csv')
y_sm.to_csv('y_smote_equal_weights.csv')


# In[ ]:





# # **SMOTE with unequal class weights**

# By default, SMOTE will oversample all classes to have the same number of examples as the class with the most examples.In this case, class 0 has the most examples, therefore, SMOTE will oversample all classes to have 249 examples as in y_train and 60 in y_test.

# **Training set**

# In[46]:


# Assigning stratified class weights

strategy = {0:316, 1:216, 2:216, 3:150, 4:75}

smote_unequal = SMOTE(sampling_strategy=strategy)
X_sm_un, y_sm_un = smote_unequal.fit_resample(X, y_enc)

# Calculate the class distribution of the oversampled dataset
counter_oversampled = Counter(y_sm_un)

# Sort the oversampled counter dictionary by keys (class numbers) in ascending order
sorted_counter_oversampled = dict(sorted(counter_oversampled.items()))

# Print the sorted class distribution summary for the oversampled dataset
for k, v in sorted_counter_oversampled.items():
    per = v / len(y_sm_un) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

# List of shades of blue for the plot
color_palette = ['lightblue', 'skyblue', 'dodgerblue', 'mediumblue', 'darkblue']

# Sort the color_palette based on the sorted_counter_oversampled keys
sorted_colors = [color_palette[k] for k in sorted_counter_oversampled.keys()]

# Create the bar plot for the oversampled dataset
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(sorted_counter_oversampled.keys(), sorted_counter_oversampled.values(), color=sorted_colors)

# Adding labels and title
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Class Distribution (Oversampled)')

# Show the plot
plt.show()


# In[15]:


original_size = len(X)
original_size


# In[16]:


smote_size_un = len(X_sm_un)
smote_size_un


# In[17]:


X_sm_un = pd.DataFrame(X_sm_un)
y_sm_un = pd.DataFrame(y_sm_un)


# In[18]:


X_sm_un.to_csv('X_smote_unequal_weights.csv')
y_sm_un.to_csv('y_smote_unequal_weights.csv')

