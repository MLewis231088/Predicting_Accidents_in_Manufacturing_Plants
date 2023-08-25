#!/usr/bin/env python
# coding: utf-8

# # **Few Shot Learning (Siamese Networks) with Bidirectional LSTM, Glove Embeddings and Data Augmentation (Synonym Replacement for Minority Classes)**
# 
# ---
# 
# 

# Few-shot text classification is a fundamental NLP task in which a model aims to classify text into many categories, given only a few training examples per category. As this dataset contains only 411 samples, most of the machine learning models and neural network models tend to under fit due to low variance and high bias in the dataset. At the same time, when tried oversampling the lower classes, these models overfit. A better option could be to collect more samples but it time-consuming and expensive. 
# To overcome these problems, we can use few-shot learning to train a machine learning model with a minimal amount of data. Typically, machine learning models are trained on large volumes of data, the larger the better. However, few-shot learning is an important machine learning concept for a few different reasons.
# One reason for using few-shot learning is that it can dramatically cut the amount of data needed to train a machine learning model, which cuts the time needed to label large datasets down. Likewise, few-shot learning reduces the need to add specific features for various tasks when using a common dataset to create different samples. Few-shot learning can ideally make models more robust and able to recognize object-based on less data, creating more general models as opposed to the highly specialized models which are the norm.
# 
# 
# Steps to train a Few-Shot Learning Model:
# 1.	The data must be split into train and test sets.
# 2.	The text must be preprocessed by removing stopwords, punctuation marks, special characters, numbers. It should be stemmed to its general form and tokenized.
# 3.	Only the train set is used to create Siamese networks. To create a Siamese network, the samples belonging to the same class are paired together to form (a, p) pairs. The samples belonging to different classes are paired together to form (a, n) pairs.
# where,
#     - a – anchor
#     - p – positive
#     - n – negative
# 
#     The (a,p) pair is given a value as 1.0 and (a,n) pair as 0.0
# 

# **Importing Libraries**

# In[ ]:


import pandas as pd 
import numpy as np 
import tensorflow as tf 
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import wordnet, stopwords
import warnings
warnings.filterwarnings('ignore')


# **Loading the Dataset**

# In[ ]:


cd "/content/drive/MyDrive/Colab Notebooks/Capstone Project/Models"


# In[ ]:


# read csv file using padas
df = pd.read_csv('Data Set - industrial_safety_and_health_database_with_accidents_description.csv', index_col=0)
df.head()


# **Finding Duplicates**

# In[ ]:


print(df.duplicated().sum())


# In[ ]:


df = df.drop_duplicates()


# **Removing unnecessary columns**

# In[ ]:


# dropping the unnecessary columns

df.drop(columns=['Data','Countries', 'Local', 'Industry Sector', 
       'Potential Accident Level', 'Genre', 'Employee or Third Party',
       'Critical Risk'], axis=1, inplace=True)


# In[ ]:


# Converting classes of the target column to numerical
df["Accident Level"].replace({"I": "1", "II": "2","III": "3", "IV": "4","V": "5"}, inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.nunique()  # Count the number of different modalities in each column


# In[ ]:


df['Accident Level'].value_counts()  # Display the class distribution in the 'Accident Level' column


# **Data Augmentation using Synonym Replacement in Minority Classes**

# In[ ]:


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


# In[ ]:


import nltk
nltk.download('stopwords')
nltk.download('wordnet')


# In[ ]:


from nltk.corpus import stopwords
stop_words = []
for w in stopwords.words('english'):
    stop_words.append(w)
print(stop_words)


# In[ ]:


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


# In[ ]:


# Displaying a sample text to display the synthetic data
trial_sent = df['Description'][6]
print(trial_sent)


# In[ ]:


# Create 3 augmented sentences per data 

for n in range(3):
    print(f" Example of Synonym Replacement: {synonym_replacement(trial_sent,n)}")


# In[ ]:


df_aug = df.copy()  # creating a copy of the dataframe


# In[ ]:


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


# In[ ]:


# creating dataframes for augmented text and their labels
augmented_sentences = pd.DataFrame(augmented_sentences)
augmented_sentences_labels = pd.DataFrame(augmented_sentences_labels)


# In[ ]:


# Concatenating the augmented sentences and their labels
aug_df = pd.concat([augmented_sentences,augmented_sentences_labels], axis=1, ignore_index=True)
aug_df.rename(columns={0:'Description',1:'Accident Level'}, inplace = True)
aug_df


# In[ ]:


# Appending the augmented dataframe to the original dataframe
df_aug = df_aug.append([aug_df], ignore_index=True)


# In[ ]:


# Display the shapes of original, augmented and combined dataframes
df_aug.shape, df.shape


# In[ ]:


# Define a function to visualize the count plot
def vis_countplot(self,x):
  plt.style.use("dark_background")
  plt.figure(figsize=(10,5))
  figure = sns.countplot(x, palette='autumn')
  figure.set_xticklabels(figure.get_xticklabels(),rotation=90)
  for p in figure.patches:
      height = p.get_height()
      figure.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[ ]:


# Visualizing the count plot of the original dataframe
vis_countplot(df,df['Accident Level'])


# In[ ]:


# Visualizing the count plot of the new augmented dataframe
vis_countplot(df_aug,df_aug['Accident Level'])


# In[ ]:


df_aug.columns


# **Splitting into Train-Test Sets**

# In[ ]:


X = df_aug['Description']
y = df_aug['Accident Level']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state=786)


# In[ ]:


import numpy as np

print(f"Numbers of train instances by class: {np.bincount(y_train)}")
print(f"Numbers of test instances by class: {np.bincount(y_test)}")


# In[ ]:


# Converting the numpy arrays into dataframes
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


# Creating the train and test sets
df_train = pd.concat([X_train,y_train], axis=1)

df_test = pd.concat([X_test,y_test], axis=1)


# ## **Preprocessing Text for NLP**

# In[ ]:


from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re

# Lemmatizes the words to its root form
lemmatizer = WordNetLemmatizer()

# Stems the word to its base form
stemmer = PorterStemmer() 


# In[ ]:


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


# In[ ]:


# Applying the "preprocess(sentence)" function to train and test sets
df_train['cleaned_text']=df_train['Description'].map(lambda s:preprocess(s)) 
df_test['cleaned_text']=df_test['Description'].map(lambda s:preprocess(s))


# In[ ]:


# Saving the train and test sets
df_train.to_csv('/content/drive/MyDrive/Colab Notebooks/Capstone Project/Models/Imbalanced/train_aug_fsl_1inp.csv', index=False)
df_test.to_csv('/content/drive/MyDrive/Colab Notebooks/Capstone Project/Models/Imbalanced/test_aug_fsl_1inp.csv', index=False)


# In[ ]:


# Displaying the preprocessed text
df_train.head(10)


# In[ ]:


labels = df_train['Accident Level'].unique()
labels


# ## **Creating a Siamese DataFrame for Few-Shot Learning**
# 
# Only the train set is used to create Siamese dataframe. To create a Siamese dataframe, the samples belonging to the same class are paired together to form (a, p) pairs. The samples belonging to different classes are paired together to form (a, n) pairs. 
#   where,
# 
#     - a – anchor
#     - p – positive
#     - n – negative
#     
#     The (a,p) pair is given a value as 1.0 and (a,n) pair as 0.0

# In[ ]:


text_left = []
text_right = []
target = []


for label in labels:
    
    similar_texts = df_train[df_train['Accident Level']==label]['cleaned_text']
    group_similar_texts = list(itertools.combinations(similar_texts,2))
    
    text_left.extend([group[0] for group in group_similar_texts])
    text_right.extend([group[1] for group in group_similar_texts])
    target.extend([1.]*len(group_similar_texts))

    dissimilar_texts = df_train[df_train['Accident Level']!=label]['cleaned_text']
    for i in range(len(group_similar_texts)):
        text_left.append(np.random.choice(similar_texts))
        text_right.append(np.random.choice(dissimilar_texts))
        target.append(0.)
        
dataset = pd.DataFrame({'text_left':text_left,
                    'text_right':text_right,
                    'target': target})


# In[ ]:


# Saving the Siamese Dataframe
dataset.to_csv('/content/drive/MyDrive/Colab Notebooks/Capstone Project/Models/Imbalanced/siamese_dataframe.csv', index=False)


# In[ ]:


# Displaying the Siamese Dataframe # Add 2 extra columns - combination of accident level targets
dataset.sample(10)


# In[ ]:


# To view the number of siamese samples generated
dataset.info()


# There are 221294 samples generated using the (a,n) and (a,p) pairs.

# ## **Building a Bi-Directional LSTM Model to calculate the distance b/w (anchor, positive) & (anchor, negative) pairs**
# 
# Convert the text to sequence using the Tokenizer module in the tensorflow.keras.preprocessing.text library, and pad the sequence to a maximum length of 200 using the pad_sequences module from the tensorflow.keras.preprocessing.sequence library.

# In[ ]:


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Subtract, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Sequential, Model


# In[ ]:


MAX_SEQ_LENGTH = 200 # to pad the text with max sequence length
VOCAB_SIZE = 10000  # size of the vocabulary


# In[ ]:


tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(df_train.cleaned_text)

# tokenizes the left arm of the siamese dataframe
sequences_left = tokenizer.texts_to_sequences(dataset.text_left)
# tokenizes the right arm of the siamese dataframe
sequences_right = tokenizer.texts_to_sequences(dataset.text_right)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# the left and right arms of the siamese dataframe is padded to max seq length
x_left = pad_sequences(sequences_left, maxlen=MAX_SEQ_LENGTH)
x_right = pad_sequences(sequences_right, maxlen=MAX_SEQ_LENGTH)

print(x_left.shape)
print(x_right.shape)


# **Using Glove Word Embeddings**

# TensorFlow enables you to train word embeddings. However, this process not only requires a lot of data but can also be time and resource-intensive. To tackle these challenges you can use pre-trained word embeddings. Let's illustrate how to do this using GloVe (Global Vectors) word embeddings by Stanford.  These embeddings are obtained from representing words that are similar in the same vector space. This is to say that words that are negative would be clustered close to each other and so will positive ones.
# 
# The first step is to obtain the word embedding and append them to a dictionary. After that, you'll need to create an embedding matrix for each word in the training set. Let's start by downloading the GloVe word embeddings.

# Next, create that dictionary with those embeddings. Let's work with the glove.6B.200d.tx embeddings. The 200 in the name is the same as the maximum length chosen for the sequences.

# In[ ]:


embeddings_index = {}
f = open('/content/drive/MyDrive/Colab Notebooks/glove.6B.200d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# The next step is to create a word embedding matrix for each word in the word index that is obtained. If a word doesn't have an embedding in GloVe, it will be presented with a zero matrix.

# In[ ]:


EMBEDDING_DIM = 200


# In[ ]:


num_words = min(VOCAB_SIZE, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > VOCAB_SIZE:
        continue
    embedding_vector = embeddings_index.get(word) ## This references the loaded embeddings dictionary
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# Now, we define the exponent_neg_manhattan_distance(arms_difference) function that computes the similarity between the left and right pairs in the Siamese dataframe.

# In[1]:


# Function to calculate the Manhattan distance between (a,p) and (a,n) pairs
def exponent_neg_manhattan_distance(arms_difference):
    """ Compute the exponent of the opposite of the L1 norm of a vector, to get the left/right inputs
    similarity from the inputs differences. This function is used to turn the unbounded
    L1 distance to a similarity measure between 0 and 1"""

    return K.exp(-K.sum(K.abs(arms_difference), axis=1, keepdims=True))


# Create a sequential bidirectional LSTM model which takes the pairs of input text and computes the similarity score which is the output of the model.

# In[ ]:


def siamese_lstm_model(max_length):

    input_shape = (max_length,)
    input_left = Input(input_shape,name = 'input_left')
    input_right = Input(input_shape,name = 'input_right')

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_length,
                                trainable=False)

    seq = Sequential(name='sequential_network')
    seq.add(embedding_layer)
    seq.add(Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.)))

    output_left = seq(input_left)
    output_right = seq(input_right)

    # Here we subtract the neuron values of the last layer from the left arm 
    # with the corresponding values from the right arm

    subtracted = Subtract(name='pair_representations_difference')([output_left, output_right])
    malstm_distance = Lambda(exponent_neg_manhattan_distance, 
                             name='masltsm_distance')(subtracted)

    siamese_net = Model(inputs=[input_left, input_right], outputs=malstm_distance)
    siamese_net.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    return siamese_net


siamese_lstm = siamese_lstm_model(MAX_SEQ_LENGTH)
siamese_lstm.summary()


# We train the model with the Siamese dataframe to compute the similarity between all the pairs in the dataframe.

# In[ ]:


# Early Stopping to minimize the validation loss
from keras.callbacks import ModelCheckpoint, EarlyStopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

siamese_lstm.fit([x_left,x_right], dataset.target, epochs=50);


# In[ ]:


# fig, ax = plt.subplots(2,1 , figsize=(22,7))
# ax[0].plot(hist.history['loss'], color='b', label="Training loss")
# # ax[0].plot(hist.history['step'], color='r', label="validation loss",axes =ax[0])
# legend = ax[0].legend(loc='best', shadow=True)

# ax[1].plot(hist.history['accuracy'], color='r', label="Training accuracy")
# # ax[1].plot(hist.history['val_accuracy'], color='r',label="Validation accuracy")
# legend = ax[1].legend(loc='best', shadow=True)


# ## **Saving the Model**

# In[ ]:


# # serialize model to JSON
# model_json = siamese_lstm.to_json()
# with open("fsl_model_1input.json", "w") as json_file:
#     json_file.write(model_json)

# # serialize weights to HDF5
# siamese_lstm.save_weights("fsl_model_1input.h5")
# print("Saved model to disk")

# save model and architecture to single file in Keras model
siamese_lstm.save_weights("fsl_model_1input.h5")
print("Saved model to disk")


# ## **Loading the Saved Model**

# In[ ]:


# # load json and create model
# json_file = open('fsl_model_1input.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("fsl_model_1input.h5")
# print("Loaded model from disk")

from keras.models import load_model
# load model in Keras
siamese_lstm = load_model('fsl_model_1input.h5')


# ## **Predictions on Unseen Data**

# Create reference sequences of the text in the train set. The text is converted to a sequence using the Tokenizer module from the tensorflow.keras.preprocessing.text library, and then padded to the maximum sequence length which is set to 200.

# In[ ]:


reference_sequences = tokenizer.texts_to_sequences(df_train.cleaned_text)
x_reference_sequences = pad_sequences(reference_sequences, maxlen=MAX_SEQ_LENGTH)


# The unseen sample is tokenized using Tokenizer, flattened using the customized flatten_text_sequence(text) function, and then padded to the maximum sequence length.
# 
# The model then predicts the similarity of the test sample with all the reference sequences (all the text in train set). 
# 
# The class predicted will be the class of the index belonging to the most similar text in the train set.
# 

# In[ ]:


import itertools

def flatten_text_sequence(text):
    flatten = itertools.chain.from_iterable
    text = list(flatten(text))
    return text

def get_prediction(text):
    """ Get the predicted category, and the most similar text
    in the train set. """
    x = tokenizer.texts_to_sequences(text.split())
    x = flatten_text_sequence(x)
    x = pad_sequences([x], maxlen=MAX_SEQ_LENGTH)

    # Compute similarities of the text with all text's in the train set
    result = np.repeat(x, len(x_reference_sequences), axis=0)
    similarities = siamese_lstm.predict([result, x_reference_sequences])
    most_similar_index = np.argmax(similarities)
    
    # The predicted category is the one with the most similar example from the train set
    prediction = df_train['Accident Level'].iloc[most_similar_index]
    most_similar_example = df_train['cleaned_text'].iloc[most_similar_index]

    return prediction, most_similar_example


# In[ ]:


# Viewing the dimension of the result

x  = df_train['cleaned_text'].iloc[34]

x = tokenizer.texts_to_sequences(x.split())
x = flatten_text_sequence(x)
x = pad_sequences([x], maxlen=MAX_SEQ_LENGTH)  

result = np.repeat(x, len(x_reference_sequences), axis=0)
print(result.shape)


# In[ ]:


# Displaying a sample prediction

sample_idx = 24

pred, most_sim = get_prediction(df_test.Description[sample_idx])

print(f'Sampled Text: {df_test["cleaned_text"].iloc[sample_idx]}')
print(f'True Class: {df_test["Accident Level"].iloc[sample_idx]}')
print(f'Predicted Class : {pred}')
print(f'Most similar example in train set: {most_sim}')


# In[ ]:


# Label Encoding the Target Variable in Train and Test sets
from sklearn.preprocessing import LabelEncoder

classes_encoder = LabelEncoder()

y_train = classes_encoder.fit_transform(y_train)
y_test = classes_encoder.transform(y_test)


# In[ ]:


from sklearn.metrics import accuracy_score

# Predicting the classes on unseen data
y_pred = [get_prediction(Description)[0] for Description in df_test['Description']]
accuracy = accuracy_score(classes_encoder.transform(y_pred), y_test)

print(f'Test accuracy (siamese model): {100*accuracy:.2f} %')


# In[ ]:


# from sklearn.metrics import log_loss

# log_loss = log_loss(y_test, classes_encoder.transform(y_pred), y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix

# Displaying the confusion matrix

target_names = ['Acc Level 1', 'Acc Level 2', 'Acc Level 3', 'Acc Level 4', 'Acc Level 5']
cm = confusion_matrix(classes_encoder.transform(y_pred), y_test)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(cm=cm, classes=target_names, title='Confusion Matrix')


# In[ ]:


# Displaying the classification report
from sklearn.metrics import classification_report

print(classification_report(classes_encoder.transform(y_pred), y_test, target_names=target_names))


# In[ ]:


from sklearn import metrics

# Generating the precision score
precision = metrics.precision_score(y_test, classes_encoder.transform(y_pred),average='macro')

# Generating the recall score
recall = metrics.recall_score(y_test, classes_encoder.transform(y_pred), average='micro')

# Generating the f1 score
f1 = metrics.f1_score(y_test, classes_encoder.transform(y_pred), average='weighted')


# In[ ]:


# Creating a dataframe to store model's performance results
model_performance = pd.DataFrame(columns=['Model', 'Accuracy', 'Log Loss','Precision', 'Recall', 'F1'])


# In[ ]:


model_performance = model_performance.append({'Model':'FSL + BiLSTM + Data Aug (Syn replacement) for class II, III, IV and V',
                                              'Accuracy': accuracy,
                                              'Log Loss': 'NA',
                                              'Precision': precision,
                                              'Recall': recall,
                                              'F1': f1                                    
                                              }, ignore_index=True)

model_performance


# **Observation**
# 
# - This model has a good performance when few shot learning model was trained on augmented data.
# - The model predicts the unseen data with an accuracy of 95% approx.
# - The model also clocks 94% and above for other metrics such as precision, recall and f1 score.
# - Surprisingly, the Accident Level V had only 8 number of samples. With data augmentation, the model is able to predict this class with 100% for precision, recall and f1 score.
# - The precision of Accident Level I is 84%, which is the least of all the classes. This could be due to the reason that synthetic data was not generated for this class.
# 
