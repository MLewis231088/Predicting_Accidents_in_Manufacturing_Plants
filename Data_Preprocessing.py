#!/usr/bin/env python
# coding: utf-8

# # **Importing the necessary libraries**

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import keras
import nltk
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# # **Loading the Dataset**

# In[2]:


# read csv file using padas
df = pd.read_csv('../Data/Dataset-industrial_safety_and_health_database_with_accidents_description.csv', index_col=0)
df.head()


# **Checking for null values**

# In[3]:


df.isnull().sum()


# There are no null values in the dataset.

# # Finding Duplicates

# In[4]:


print(df.duplicated().sum())


# In[5]:


df = df.drop_duplicates()


# # Removing unnecessary columns

# In[6]:


# dropping the unnecessary columns

df.drop(columns=['Data'], axis=1, inplace=True)


# # Viewing the length of description

# In[7]:


desc_length = [len(comment.split()) for comment in df['Description']]

df["Desc_Length"] = desc_length


# # Encoding the Target Variable

# In[8]:


df["Accident Level"].replace({"I": "1", "II": "2","III": "3", "IV": "4","V": "5"}, inplace=True)


# # **Assigning Features and Target**

# In this use case, the feature will be the 'Description' column and the target will be the 'Accident Level' column.

# In[9]:


X = df['Description'].astype("str")
y = df['Accident Level']


# # **Data Preprocessing**
# 
# Data preprocessing is one of the critical steps in any machine learning project. It includes cleaning and formatting the data before feeding into a machine learning algorithm. For NLP, the preprocessing steps are comprised of the following tasks:
# 
# - Tokenizing the string
# - Lowercasing
# - Removing stop words and punctuation
# - Stemming
# 
# All the above can be achieved in TensorFlow using Tokenizer. The class expects a couple of parameters:
# 
# - num_words: the maximum number of words you want to be included in the word index
# - oov_token: the token to be used to represent words that won't be found in the word dictionary. This usually happens when processing the training data. The number 1 is usually used to represent the "out of vocabulary" token ("oov" token)
# 
# The fit_on_texts function is used to fit the Tokenizer on the training set once it has been instantiated with the preferred parameters.

# In[10]:


from keras.preprocessing.text import Tokenizer
vocab_size = 10000
oov_token = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X)


# In[11]:


len(tokenizer.word_index)


# The word_index can be used to show the mapping of the words to numbers.

# In[12]:


word_index = tokenizer.word_index


# In[13]:


word_index


# # **Converting text to sequences**

# The next step is to represent each sentiment as a sequence of numbers. This can be done using the texts_to_sequences function.

# In[14]:


X_sequences = tokenizer.texts_to_sequences(X)


# Here is how these sequences look.

# In[15]:


X_sequences[3:7]


# We can see that the length of the description vary in all the articles. For a deep learning model (especially LSTMs), the length of the inputs must be equal. So, in the next step, we'll pad the sequences to make all the sequences of equal length.

# # **Padding the sequences**

# While padding, longer sequences will be truncated while shorter ones will be padded with zeros. We'll therefore have to declare the truncation and padding type.
# 
# Let's start by defining the maximum length of each sequence, the padding type, and the truncation type. A padding and truncation type of "post" means that these operations will take place at the end of the sequence.

# In[16]:


max_length = df['Desc_Length'].max()
max_length


# In[17]:


max_length = 200
padding_type='post'
truncation_type='post'


# We'll use the pad_sequences function while passing the parameters defined above.

# In[18]:


from keras.preprocessing.sequence import pad_sequences

X_padded = pad_sequences(X_sequences,
                               maxlen=max_length,
                               padding=padding_type,
                               truncating=truncation_type)


# Now, let's see how the sequences are padded.

# In[19]:


X_padded


# In[20]:


# label encode the target variable
y_enc = LabelEncoder().fit_transform(y)


# **Saving the preprocessed and vectorized X and y dataframes**

# In[21]:


X_final = pd.DataFrame(X_padded)
y_final = pd.DataFrame(y_enc)

X_final.to_csv('X_vector.csv')
y_final.to_csv('y_enc.csv')


# In[22]:


X_final


# In[23]:


y.to_csv('y.csv')


# In[ ]:




