#!/usr/bin/env python
# coding: utf-8

# # **Data Augmentation using Synonym Replacement**

# In[1]:


pip install --user -U nltk


# **Importing Libraries**

# In[2]:


import pandas as pd 
import numpy as np 
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import wordnet, stopwords
import warnings
warnings.filterwarnings('ignore')


# **Loading the Dataset**

# In[3]:


# read csv file using padas
df = pd.read_csv('../Data/Dataset-industrial_safety_and_health_database_with_accidents_description.csv', index_col=0)
df.head()


# **Finding Duplicates**

# In[4]:


print(df.duplicated().sum())


# In[5]:


df = df.drop_duplicates()


# **Removing unnecessary columns**

# In[6]:


# dropping the unnecessary columns

df.drop(columns=['Data','Countries', 'Local', 'Industry Sector', 
       'Potential Accident Level', 'Genre', 'Employee or Third Party',
       'Critical Risk'], axis=1, inplace=True)


# In[7]:


# Converting classes of the target column to numerical
df["Accident Level"].replace({"I": "1", "II": "2","III": "3", "IV": "4","V": "5"}, inplace=True)


# In[8]:


df.head()


# In[9]:


df.nunique()  # Count the number of different modalities in each column


# In[10]:


df['Accident Level'].value_counts()  # Display the class distribution in the 'Accident Level' column


# **Data Augmentation using Synonym Replacement in Minority Classes**

# In[11]:


from nltk.corpus import wordnet

# a function to generate synonyms from wordnet
def get_synonyms(word):
    # creates a set of synonyms with unique words
    synonyms = set()

    for syn in wordnet.synsets(word):
        # lemmatized words
        for l in syn.lemmas():
            # converts dashes into spaces, and words into lowercase
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            # filters english words
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            # adds synonym to the set of unique words
            synonyms.add(synonym)
    if word in synonyms:
        # looks for words that are not present in the list of unique synonyms
        synonyms.remove(word)
    
    # returns a list of unique synonyms
    return list(synonyms)


# In[12]:


import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
nltk.download('stopwords')
nltk.download('wordnet')


# In[13]:


from nltk.corpus import stopwords
stop_words = []
for w in stopwords.words('english'):
    stop_words.append(w)
print(stop_words)


# In[14]:


import random

# a function to replace a word with its synonym
def synonym_replacement(words, n):
    """
      words - the sentence to replace its words with their synonyms
      n     - the number of examples to generate with the replaced synonyms
    """
    # splits words from a sentence
    words = words.split()
    # creates a copy of words
    new_words = words.copy()
    # generates a random word that are not stopwords
    random_word_list = list(set([word for word in words if word not in stop_words]))
    # shuffles the random words list
    random.shuffle(random_word_list)
    # initializing the count of number of replaced words to zero
    num_replaced = 0
    
    for random_word in random_word_list:
        # generates a synonym for a random word in the sentence
        synonyms = get_synonyms(random_word)
        
        """ if there is more than one synonym for the word, the word is replaced with the synonym
            else the word is retained """
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # if the word is replaced with the synonym, the count of replaced words is incremented
            num_replaced += 1
        
        #only replaces up to n words
        if num_replaced >= n: 
            break

    # joins the sentence with the replaced synonyms
    sentence = ' '.join(new_words)

    # returns the new sentence(s)
    return sentence


# In[15]:


# Displaying a sample text to display the synthetic data
trial_sent = df['Description'][23]
print(trial_sent)


# In[16]:


# Create 3 augmented sentences per data 

for n in range(3):
    print(f" Example of Synonym Replacement: {synonym_replacement(trial_sent,n)}")


# In[17]:


df_aug = df.copy()  # creating a copy of the dataframe


# In[18]:


augmented_sentences=[]
augmented_sentences_labels=[]

# Generate synthesized data for minority classes (II, III, IV and V) using the synonym_replacement(words, n) function.
for i in df_aug.index:
  if df_aug['Accident Level'][i]=='2':
    # generates 6 different sentences with synonyms for each sample belonging to Accident Level II
    for n in range(6):
      temps=synonym_replacement(df_aug['Description'][i],n)
      augmented_sentences.append(temps)
      augmented_sentences_labels.append(df_aug['Accident Level'][i])
  elif df_aug['Accident Level'][i]=='3':
    # generates 9 different sentences with synonyms for each sample belonging to Accident Level III
    for n in range(9):
      temps=synonym_replacement(df_aug['Description'][i],n)
      augmented_sentences.append(temps)
      augmented_sentences_labels.append(df_aug['Accident Level'][i])
  elif df_aug['Accident Level'][i]=='4':
    # generates 9 different sentences with synonyms for each sample belonging to Accident Level IV
    for n in range(9):
      temps=synonym_replacement(df_aug['Description'][i],n)
      augmented_sentences.append(temps)
      augmented_sentences_labels.append(df_aug['Accident Level'][i])
  elif df_aug['Accident Level'][i]=='5':
    # generates 37 different sentences with synonyms for each sample belonging to Accident Level V
    for n in range(37):
      temps=synonym_replacement(df_aug['Description'][i],n)
      augmented_sentences.append(temps)
      augmented_sentences_labels.append(df_aug['Accident Level'][i])


# In[19]:


# creating dataframes for augmented text and their labels
augmented_sentences = pd.DataFrame(augmented_sentences)
augmented_sentences_labels = pd.DataFrame(augmented_sentences_labels)


# In[20]:


# Concatenating the augmented sentences and their labels
aug_df = pd.concat([augmented_sentences,augmented_sentences_labels], axis=1, ignore_index=True)
aug_df.rename(columns={0:'Description',1:'Accident Level'}, inplace = True)
aug_df


# In[21]:


# Appending the augmented dataframe to the original dataframe
df_aug = pd.concat([df_aug, aug_df], ignore_index=True)
# df_aug = df_aug.append([aug_df], ignore_index=True)


# In[22]:


# Display the shapes of original, augmented and combined dataframes
df_aug.shape, df.shape


# In[23]:


# Define a function to visualize the count plot
def vis_countplot(self,x):
#   plt.style.use("dark_background")
  plt.figure(figsize=(10,5))
  subcategory_counts = x.value_counts()
  figure = sns.barplot(x=subcategory_counts.index, y=subcategory_counts.values, palette='winter')
  figure.set_xticklabels(figure.get_xticklabels(),rotation=90)
  for p in figure.patches:
      height = p.get_height()
      figure.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[24]:


# Visualizing the count plot of the original dataframe
vis_countplot(df,df['Accident Level'])


# In[25]:


# Visualizing the count plot of the new augmented dataframe
vis_countplot(df_aug,df_aug['Accident Level'])


# In[26]:


df_aug.columns


# **Splitting into Train-Test Sets**

# In[27]:


X = df_aug['Description']
y = df_aug['Accident Level']


# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state=786)


# In[29]:


import numpy as np

print(f"Numbers of train instances by class: {np.bincount(y_train)}")
print(f"Numbers of test instances by class: {np.bincount(y_test)}")


# In[30]:


# Converting the numpy arrays into dataframes
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[31]:


# Creating the train and test sets
df_train = pd.concat([X_train,y_train], axis=1)

df_test = pd.concat([X_test,y_test], axis=1)


# ## **Preprocessing Text for NLP**

# In[32]:


from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re

# Lemmatizes the words to its root form
lemmatizer = WordNetLemmatizer()

# Stems the word to its base form
stemmer = PorterStemmer() 


# In[33]:


def preprocess(sentence):
  # Converts the sentence to string format
  sentence=str(sentence)
  # converts the text to lowercase
  sentence = sentence.lower()
  # Removes the html tags
  sentence=sentence.replace('{html}',"") 
  # Removes the special characters
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', sentence)
  # Removes the http(s) tags
  rem_url=re.sub(r'http\S+', '',cleantext)
  # Removes numbers
  rem_num = re.sub('[0-9]+', '', rem_url)
  # Tokenizes text
  tokenizer = RegexpTokenizer(r'\w+')
  tokens = tokenizer.tokenize(rem_num)  
  # Removes stopwords
  filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
  # Stems the word to its base form
  stem_words=[stemmer.stem(w) for w in filtered_words]
  # Lemmatizes the word to its root form
  lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
  # Returns the preprocessed text
  return " ".join(filtered_words)


# In[34]:


# Applying the "preprocess(sentence)" function to train and test sets
df_train['cleaned_text']=df_train['Description'].map(lambda s:preprocess(s)) 
df_test['cleaned_text']=df_test['Description'].map(lambda s:preprocess(s))


# In[35]:


# Saving the train and test sets
df_train.to_csv('train_aug_fsl_1inp.csv', index=False)
df_test.to_csv('test_aug_fsl_1inp.csv', index=False)


# In[36]:


# Displaying the preprocessed text
df_train.head(10)


# In[37]:


labels = df_train['Accident Level'].unique()
labels


# In[ ]:




