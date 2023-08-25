#!/usr/bin/env python
# coding: utf-8

# # 4. Data Modelling using Pycaret

# In this section, we will perform the following steps:
# - Set the target column
# - Create Model: create a model, perform stratified cross validation and evaluate classification metrics
# - Tune Model: tune the hyper-parameters of a classification model
# - Ensemble Model: uses bagging and boosting techniques to reduce variance and bias respectively
# - Blend Model: combines different machine learning algorithms and use a majority vote or the average predicted probabilities in case of classification to predict the final outcome
# - Stacking: builds a meta model that generates the final prediction using the prediction of multiple base estimators
# - Plot Model: analyze model performance using various plots
# - Finalize Model: finalize the best model at the end of the experiment
# - Predict Model: make predictions on new / unseen data
# - Save / Load Model: save / load a model for future use

# ## Loading the libraries

# In[1]:


get_ipython().system('pip install pycaret')


# In[2]:


import numpy as np 
import pandas as pd 
import pycaret
from pycaret import classification
from pycaret.classification import *

# to remove future warnings with seaborn
import warnings
warnings.filterwarnings('ignore')

# to disable INFO LOG messages
import logging, sys
logging.disable(sys.maxsize)


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


cd "/content/drive/MyDrive/Colab Notebooks/Capstone Project/Models/Balanced/SMOTE"


# ## Importing the dataset

# In[5]:


X = pd.read_csv('X_smote_equal_weights.csv', index_col=0)
X.head()


# In[6]:


y = pd.read_csv('y_smote_equal_weights.csv',index_col=0)

y.head()


# In[7]:


df = pd.concat([X,y], axis=1, ignore_index=True)
df.rename(columns={200:'Accident_Level'}, inplace = True)
df


# ## Overview

# PyCaret's classification module (pycaret.classification) is a supervised machine learning module which is used for classifying the elements into a binary group based on various techniques and algorithms. 
# The PyCaret classification module can be used for Binary or Multi-class classification problems. It has over 18 algorithms and 14 plots to analyze the performance of models.

# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#4863A0"> Preparing the Dataset

# In[8]:


data = df.sample(frac=0.95, random_state=786)
data_unseen = df.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#4863A0"> Setting the Target column

# The setup() function initializes the environment in pycaret, and prepares the data for modeling and deployment. setup() must be called before executing any other function in pycaret. It takes two mandatory parameters: a pandas dataframe and the name of the target column. All other parameters are optional.

# When setup() is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To account for this, PyCaret displays a table containing the features and their inferred data types after setup() is executed. If all of the data types are correctly identified enter can be pressed to continue or quit can be typed to end the expriment. 

# <span style="font-family: Arial; font-weight:bold;font-size:1.2em;color:#4E9258"> Train-Test Split (70:30)

# In Machine Learning, we spilt the dataset into train and test data. We perform optimization of the hyperparameters in PyCaret using k-fold cross validation on train dataset only. Test dataset is used only to evaluate metrics and determine if the model has over-fitted the data. By default, PyCaret uses 70% of the dataset for training and 30% for testing. The proportion of split can be changed using train_size parameter within setup().

# In[9]:


data_setup=setup(data=df,
                 target='Accident_Level')


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#4863A0"> Comparing All Models

# Comparing all models to evaluate performance is the first stage of modeling, once the setup is completed (unless you exactly know what kind of model you need, which is often not the case). This function trains all the models in the model library and scores them using stratified cross validation for metric evaluation. The output prints a score grid that shows average Accuracy, AUC, Recall, Precision, F1, Kappa, and MCC accross the folds (10 by default) along with training times.

# In[10]:


# selecting top 3 models
best_models = compare_models(n_select=3, sort='F1')


# The score grid printed above highlights the highest performing metric for comparison purposes only. The grid by default is sorted using 'Accuracy' (highest to lowest) which can be changed by passing the sort parameter. For example compare_models(sort = 'Recall') will sort the grid by Recall instead of Accuracy. If you want to change the fold parameter from the default value of 10 to a different value then you can use the fold parameter. For example compare_models(fold = 5) will compare all models on 5 fold cross validation. Reducing the number of folds will improve the training time. By default, compare_models return the best performing model based on default sort order but can be used to return a list of top N models by using n_select parameter.

# In[11]:


print(best_models)


# # **Creating Models**

# We will now use the top 5 models to train and evaluate models using cross validation.

# Before we create a model, we must know the ID of each model. We can find the ID using the models() function.

# In[12]:


models()


# **Creating an Extra Trees Classifier Model**

# In[13]:


et = create_model('et',fold = 8,round = 2)


# **Creating a Light Gradient Boosting Machine Model**

# In[14]:


lightgbm = create_model('lightgbm',fold = 8,round = 2)


# **Creating a Random Forest Classifier Model**

# In[15]:


rf = create_model('rf',fold = 8,round = 2)


# **Creating a Gradient Boosting Classifier Model**

# In[16]:


gbc = create_model('gbc',fold = 8,round = 2)


# **Creating a Logistic Regression Model**

# In[17]:


lr = create_model('lr',fold = 8,round = 2)


# **Creating a K Neighbors Classifier Model**

# In[18]:


knn = create_model('knn',fold = 8,round = 2)


# **Creating a Support Vector Machine Model**

# In[19]:


svm = create_model('svm',fold = 8,round = 2)


# **Creating a Ridge Classifier Model**

# In[20]:


ridge = create_model('ridge',fold = 8,round = 2)


# **Creating a Linear Discriminant Analysis Model**

# In[21]:


lda = create_model('lda',fold = 8,round = 2)


# **Creating a Decision Tree Classifier Model**

# In[22]:


dt = create_model('dt',fold = 8,round = 2)


# **Creating a Naive Bayes Model**

# In[23]:


nb = create_model('nb',fold = 8,round = 2)


# **Creating a Ada Boost Classifier	 Model**

# In[24]:


ada = create_model('ada',fold = 8,round = 2)


# **Creating a Quadratic Discriminant Analysis Model**

# In[25]:


qda = create_model('qda',fold = 8,round = 2)


# # **Tuning Models**

# When a model is created using the create_model() function, it uses the default hyperparameters to train the model. In order to tune hyperparameters, the tune_model() function is used. This function automatically tunes the hyperparameters of a model using Random Grid Search on a pre-defined search space. The output prints a score grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa, and MCC by fold for the best model.

# **Tuning the Light GBM Classifier Model**

# In[26]:


tuned_lightgbm = tune_model(lightgbm, optimize = 'Recall', fold = 8,round = 2)


# **Tuning the Extra Trees Classifier Model**

# In[27]:


tuned_et = tune_model(et, optimize = 'Recall', fold = 8,round = 2)


# **Tuning the Random Forest Classifier Model**

# In[28]:


tuned_rf = tune_model(rf, optimize = 'Recall', fold = 8,round = 2)


# **Tuning the Support Vector Machine Classifier Model**

# In[29]:


tuned_svm = tune_model(svm, optimize = 'Recall', fold = 8,round = 2)


# **Tuning the Decision Tree Classifier Model**

# In[30]:


tuned_dt = tune_model(dt, optimize = 'Recall', fold = 8,round = 2)


# **Tuning the Gradient Boosting Classifier Model**

# In[31]:


tuned_gbc = tune_model(gbc, optimize = 'Recall', fold = 8,round = 2)


# **Tuning the KNN Model**

# In[32]:


tuned_knn = tune_model(knn, optimize = 'Recall', fold = 8,round = 2)


# **Tuning the Logistic Regression Model**

# In[33]:


tuned_lr = tune_model(lr, optimize = 'Recall', fold = 8,round = 2)


# **Tuning the Support Vector Machine Model**

# In[34]:


tuned_svm = tune_model(svm, optimize = 'Recall', fold = 8,round = 2)


# **Tuning the Logistic Regression Model**

# In[35]:


tuned_lr = tune_model(lr, optimize = 'Recall', fold = 8,round = 2)


# **Tuning the Naive Bayes Classifier Model**

# In[36]:


tuned_nb = tune_model(nb, optimize = 'Recall', fold = 8,round = 2)


# # **Plotting Models**

# Before model finalization, the plot_model() function can be used to analyze the performance across different aspects such as AUC, confusion_matrix, decision boundary etc. This function takes a trained model object and returns a plot based on the test / hold-out set.

# **Plotting Extra Trees Classifier Model**

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> AUC Plot

# In[37]:


plot_model(tuned_et, plot = 'auc')


# - Receiver Operating Characteristic (ROC) measures the performance of models by evaluating the trade-offs between sensitivity (True Positive Rate) and 1-Specificity (False Positive Rate)
# - Sensitivity or Recall is the measure of actual observations which are predicted correctly
# - Specificity is the measure of how many observations of false category predicted correctly
# - Area Under Curve (AUC) is the measure of accuracy judged by the area under curve for ROC
# 

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Precision-Recall Curve

# In[38]:


plot_model(tuned_et, plot = 'pr')


# - Precision is the measure of correctness achieved in prediction. It is measured as TP/TP+FP
# - Recall is the measure of actual observations which are predicted correctly. It is measured as TP/TP+FN

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Feature Importance Plot

# In[39]:


plot_model(tuned_et, plot='feature')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Confusion Matrix

# In[40]:


plot_model(tuned_et, plot = 'confusion_matrix')


# - True Positives: 
#     - Number of observations in which the actual value is positive and the model predicted a positive value
#     - The number of True Positives are 11
# - True Negatives: 
#     - Number of observations in which the actual value is negative and the model predicted a negative value
#     - The number of True Negatives are 31
# - False Positives (Type I error): 
#     - Number of observations in which the actual value is negative and the model predicted a positive value
#     - The number of False Positives are 3
# - False Negatives (Type II error): 
#     - Number of observations in which the actual value is positive and the model predicted a negative value
#     - The number of False Negatives are 11
#   

# Decision Boundary

# In[41]:


plot_model(tuned_et, plot = 'boundary')


# **Observation:**
# 
# -	The ROC curves for ET Model is 1.0 for Accident Levels V, 0.99 for levels III and IV, and 0.97 for levels I and II. Its performance is also reasonably good.
# -	 It has predicted Accident Level II wrongly most number of times compared to other classes.
# - The precision-recall plot is good.
# 

# **Plotting LightGBM Classifier Model**

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> AUC Plot

# In[42]:


plot_model(tuned_lightgbm, plot = 'auc')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Precision-Recall Curve

# In[43]:


plot_model(tuned_lightgbm, plot = 'pr')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Feature Importance Plot

# In[44]:


plot_model(tuned_lightgbm, plot='feature')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Confusion Matrix

# In[45]:


plot_model(tuned_lightgbm, plot = 'confusion_matrix')


# Decision Boundary

# In[46]:


plot_model(tuned_lightgbm, plot = 'boundary')


# **Observation:**
# 
# -	The ROC curves for LightGBM Model is 1.0 for Accident Levels III, IV and V, 0.99 for level II, and 0.96 for I which is reasonably good.
# -	It predicts false positives and false negatives almost equally.
# -	The precision-recall plot is good.
# 

# **Plotting Support Vector Machine Classifier Model**

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> AUC Plot

# In[ ]:


# plot_model(tuned_svm, plot = 'auc')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Precision-Recall Curve

# In[48]:


plot_model(tuned_svm, plot = 'pr')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Feature Importance Plot

# In[49]:


plot_model(tuned_svm, plot='feature')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Confusion Matrix

# In[50]:


plot_model(tuned_svm, plot = 'confusion_matrix')


# Decision Boundary

# In[51]:


plot_model(tuned_svm, plot = 'boundary')


# **Observation:**
# 
# -	SVM Model has a good performance. It is better than DT and RF Models.
# -	The model should be improved in making lesser false negative predictions especially for class I.
# -	The precision-recall plot is good.
# 

# **Plotting Logistic Regression Classifier Model**

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> AUC Plot

# In[52]:


plot_model(tuned_lr, plot = 'auc')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Precision-Recall Curve

# In[53]:


plot_model(tuned_lr, plot = 'pr')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Feature Importance Plot

# In[54]:


plot_model(tuned_lr, plot='feature')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Confusion Matrix

# In[55]:


plot_model(tuned_lr, plot = 'confusion_matrix')


# Decision Boundary

# In[56]:


plot_model(tuned_lr, plot = 'boundary')


# **Observation:**
# 
# -	The ROC curves for RF Model is greater than 0.9 for Accident Levels II, III, IV and V, and 0.70 for level I. The model needs to improve its predictions on class I.
# -	Compared to the DT and RF models, this model has made fewer false predictions. 
# -	The precision-recall plot is average.
# 

# **Plotting Decision Tree Classifier Model**

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> AUC Plot

# In[57]:


plot_model(tuned_dt, plot = 'auc')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Precision-Recall Curve

# In[58]:


plot_model(tuned_dt, plot = 'pr')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Feature Importance Plot

# In[59]:


plot_model(tuned_dt, plot='feature')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Confusion Matrix

# In[60]:


plot_model(tuned_dt, plot = 'confusion_matrix')


# Decision Boundary

# In[61]:


plot_model(tuned_dt, plot = 'boundary')


# **Observation:**
# 
# -	The ROC curves for DT Model is similar to RF Model.
# -	Its prediction is not great as it has more false predictions.
# -	The precision-recall plot is average.
# 

# **Plotting Naive Bayes Classifier Model**

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> AUC Plot

# In[62]:


plot_model(tuned_nb, plot = 'auc')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Precision-Recall Curve

# In[63]:


plot_model(tuned_nb, plot = 'pr')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Feature Importance Plot

# In[65]:


# plot_model(tuned_nb, plot='feature')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Confusion Matrix

# In[66]:


plot_model(tuned_nb, plot = 'confusion_matrix')


# Decision Boundary

# In[67]:


plot_model(tuned_nb, plot = 'boundary')


# **Observation:**
# 
# -	Amongst all the models, NB Model has poor performance
# 

# **Plotting KNN Model**

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> AUC Plot

# In[68]:


plot_model(tuned_knn, plot = 'auc')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Precision-Recall Curve

# In[69]:


plot_model(tuned_knn, plot = 'pr')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Feature Importance Plot

# In[71]:


# plot_model(tuned_knn, plot='feature')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Confusion Matrix

# In[72]:


plot_model(tuned_knn, plot = 'confusion_matrix')


# Decision Boundary

# In[73]:


plot_model(tuned_knn, plot = 'boundary')


# **Observation:**
# 
# -	The ROC curves for GBC Model is 1.0 for Accident Levels II, III, IV and V, and 0.98 for level I, which is very high. Therefore, the model can be considered for prediction.
# -	The GBC model is able to predict most of the times correctly with only few false positives and false negatives. Surprisingly, 24 instances were wrongly predicted as class II instead of I.
# -	The precision-recall plot is also good.
# 

# **Plotting RF Classifier Model**

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> AUC Plot

# In[74]:


plot_model(tuned_rf, plot = 'auc')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Precision-Recall Curve

# In[75]:


plot_model(tuned_rf, plot = 'pr')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Feature Importance Plot

# In[76]:


plot_model(tuned_rf, plot='feature')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Confusion Matrix

# In[77]:


plot_model(tuned_rf, plot = 'confusion_matrix')


# Decision Boundary

# In[78]:


plot_model(tuned_rf, plot = 'boundary')


# **Observation:**
# 
# -	The ROC curves for RF Model is greater than 0.9 for Accident Levels II, III, IV and V, and 0.79 for level I. The model needs to improve its predictions on class I.
# -	Compared to the top models, this model has predicted wrongly more number of times. 
# -	The precision-recall plot is average.
# 

# **Plotting Gradient Boosting Classifier Model**

# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> AUC Plot

# In[79]:


plot_model(tuned_gbc, plot = 'auc')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Precision-Recall Curve

# In[80]:


plot_model(tuned_gbc, plot = 'pr')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Feature Importance Plot

# In[81]:


plot_model(tuned_gbc, plot='feature')


# <span style="font-family: Arial; font-weight:bold;font-size:1.0em;color:#920445"> Confusion Matrix

# In[82]:


plot_model(tuned_gbc, plot = 'confusion_matrix')


# Decision Boundary

# In[83]:


plot_model(tuned_gbc, plot = 'boundary')


# # **Predicting on Test / Hold-out Sample**

# In[84]:


predict_model(tuned_gbc);


# In[85]:


predict_model(tuned_knn);


# In[86]:


predict_model(tuned_lightgbm);


# In[87]:


predict_model(tuned_et);


# In[88]:


predict_model(tuned_rf);


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#4863A0"> Ensembling Models

# Ensembling a trained model is as simple as writing ensemble_model. It takes only one mandatory parameter i.e. the trained model object. This functions returns a table with k-fold cross validated scores of common evaluation metrics along with trained model object. The evaluation metrics used are:
# 
# Classification: Accuracy, AUC, Recall, Precision, F1, Kappa, MCC
# Regression: MAE, MSE, RMSE, R2, RMSLE, MAPE
# The number of folds can be defined using fold parameter within ensemble_model function. By default, the fold is set to 10. All the metrics are rounded to 4 decimals by default by can be changed using round parameter. There are two methods available for ensembling that can be set using method parameter within ensemble_model function. Both the methods require re-sampling of the data and fitting multiple estimators, hence the number of estimators can be controlled using n_estimators parameter. By default, n_estimators is set to 10.

# <span style="font-family: Arial; font-weight:bold;font-size:1.2em;color:#4E9258"> Applying the Bagging and Boosting Ensemble Techniques

# In[89]:


# # train a bagging classifier on Extra Trees Classifier
# bagged_gbc = ensemble_model(gbc, method = 'Bagging')


# In[90]:


# # train an boosting classifier on Extra Trees Classifier with 100 estimators
# boosted_gbc = ensemble_model(gbc, method = 'Boosting', n_estimators = 100)


# In[91]:


# bagged_knn = ensemble_model(knn, method = 'Bagging')


# In[92]:


# boosted_knn = ensemble_model(knn, method = 'Boosting', n_estimators = 100)


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#4863A0"> Blending Models

# Blending models is a method of ensembling which uses consensus among estimators to generate final predictions. The idea behind blending is to combine different machine learning algorithms and use a majority vote or the average predicted probabilities in case of classification to predict the final outcome. Blending models in PyCaret is as simple as writing blend_models. This function can be used to blend specific trained models that can be passed using estimator_list parameter within blend_models or if no list is passed, it will use all the models in model library. In case of Classification, method parameter can be used to define ‘soft‘ or ‘hard‘ where soft uses predicted probabilities for voting and hard uses predicted labels. This functions returns a table with k-fold cross validated scores of common evaluation metrics along with trained model object. The evaluation metrics used are:
# 
# - Classification: Accuracy, AUC, Recall, Precision, F1, Kappa, MCC
# - Regression: MAE, MSE, RMSE, R2, RMSLE, MAPE
# 
# The number of folds can be defined using fold parameter within blend_models function. By default, the fold is set to 10. All the metrics are rounded to 4 decimals by default by can be changed using round parameter within blend_models.

# In[93]:


# # train a votingclassifier on all models in library
# blender = blend_models()


# In[94]:


# # train a voting classifier on specific models
# blender_specific = blend_models(estimator_list = [gbc,et,lightgbm,rf], method = 'soft')


# In[95]:


# # train a voting classifier dynamically
# blender_specific = blend_models(estimator_list = compare_models(n_select = 5), method = 'hard')


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#4863A0"> Stacking Models (Training a Meta Classifier)

# Stacking models is method of ensembling that uses meta learning. The idea behind stacking is to build a meta model that generates the final prediction using the prediction of multiple base estimators. Stacking models in PyCaret is as simple as writing stack_models. This function takes a list of trained models using estimator_list parameter. All these models form the base layer of stacking and their predictions are used as an input for a meta model that can be passed using meta_model parameter. If no meta model is passed, a linear model is used by default. In case of Classification, method parameter can be used to define ‘soft‘ or ‘hard‘ where soft uses predicted probabilities for voting and hard uses predicted labels. This function returns a table with k-fold cross validated scores of common evaluation metrics along with trained model object. The evaluation metrics used are:
# 
# - Classification: Accuracy, AUC, Recall, Precision, F1, Kappa, MCC
# - Regression: MAE, MSE, RMSE, R2, RMSLE, MAPE
# 
# The number of folds can be defined using fold parameter within stack_models function. By default, the fold is set to 10. All the metrics are rounded to 4 decimals by default by can be changed using round parameter within stack_models. restack parameter controls the ability to expose the raw data to meta model. By default, it is set to True. When changed to False, meta-model will only use predictions of base models to generate final predictions.

# In[96]:


# # stacking models
# stacker = stack_models(estimator_list = [knn,et,lightgbm,rf], meta_model = gbc)


# In[97]:


# # stack models dynamically
# top5 = compare_models(n_select = 5)
# stacker = stack_models(estimator_list = top5[1:], meta_model = top5[0])


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#4863A0"> Finalizing Model for Deployment

# This is the last step in the experiment. The finalize_model() function fits the model onto the complete dataset including the test/hold-out sample. The purpose of this function is to train the model on the complete dataset before it is deployed in production.

# In[98]:


final_gbc = finalize_model(tuned_gbc)
print(final_gbc)


# CAUTION: Once the model is finalized using finalize_model(), the entire dataset including the test/hold-out set is used for training. As such, if the model is used for predictions on the hold-out set after finalize_model() is used, the information grid printed will be misleading as you are trying to predict on the same data that was used for modeling.

# # **Predicting Unseen Data**

# The predict_model() function is also used to predict on the unseen dataset. The only difference is that this time, we will pass the data_unseen parameter.data_unseen is the variable created at the beginning and contains 5% of the original dataset which was never exposed to PyCaret. 

# In[99]:


unseen_predictions = predict_model(final_gbc, data=data_unseen)


# The Label and Accident_Level columns get added onto the data_unseen set. Label is the prediction and score is the probability of the prediction. Notice that predicted results are concatenated to the original dataset while all the transformations are automatically performed in the background. You can also check the metrics on this since you have actual target column default available. To do that, we will use pycaret.utils module.

# In[100]:


from pycaret.utils import check_metric
check_metric(unseen_predictions.Accident_Level, unseen_predictions['Label'].astype(int), 'Accuracy')


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#4863A0"> Saving the Model

# In[101]:


save_model(final_gbc,'Final GBC Classifier Model')


# # Conclusion

# **Observation:**
# 
# - The ROC curves for GBC Model is 1.0 for Accident Levels II, III, IV and V, and 0.98 for level I, which is very high. Therefore, iti s considered as the final model.
# - The GBC model is able to predict most of the times correctly with only few false positives and false negatives.
# - The precision-recall plot is better.
# 
