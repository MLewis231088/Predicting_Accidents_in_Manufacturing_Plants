#!/usr/bin/env python
# coding: utf-8

# ## Domain: Industrial safety - NLP based Chatbot

# ## Importing necessary libraries

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
# %env HV_DOC_HTML=true
from wordcloud import WordCloud, STOPWORDS
from nltk import tokenize,stem


# **Reading data**

# In[2]:


df = pd.read_csv('../Data/Dataset-industrial_safety_and_health_database_with_accidents_description.csv', index_col=0)
df.head()


# **Viewing missing values**

# In[3]:


df.isnull().sum()    # to find the count of missing values in the entire dataset


# **Renaming the columns**

# In[4]:


# Renaming the columns with appropriate ones
df.rename(columns={'Data':'Date', 'Countries':'Country', 'Genre':'Gender', 'Employee or Third Party':'Employee type'}, 
          inplace=True)

# Converting argument to datetime
df['Date']= pd.to_datetime(df['Date']) 

df.head()


# In[5]:


# creating additional columns for date column analysis
df['Year']       = df['Date'].apply(lambda x : x.year)
df['Month']      = df['Date'].apply(lambda x : x.month)
df['Day']        = df['Date'].apply(lambda x : x.day)
df['Weekday']    = df['Date'].apply(lambda x : x.day_name())
df['WeekofYear'] = df['Date'].apply(lambda x : x.weekofyear)

df.head(3)


# # **Univariate Analysis**

# **Univariate Plots**

# *Month*

# In[6]:


month = pd.DataFrame(df['Month'].values)
# month.value_counts()


# In[24]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(10,5))
month = sns.countplot(x='Month', 
              palette='winter',
              data=df)
month.set_xticklabels(month.get_xticklabels(),rotation=90)
for p in month.patches:
    height = p.get_height()
    month.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# *Year*

# In[10]:


year = pd.DataFrame(df['Year'].values)
# year.value_counts()


# In[23]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(6,6))
year = sns.countplot(x='Year', 
              palette='winter',
              data=df
              # order = df['Year'].value_counts().index
              )
year.set_xticklabels(year.get_xticklabels(),rotation=90)
for p in year.patches:
    height = p.get_height()
    year.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# *Day*

# In[12]:


day = pd.DataFrame(df['Day'].values)
# day.value_counts()


# In[22]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(10,6))
day = sns.countplot(x='Day', 
              palette='winter',
              data=df)
day.set_xticklabels(day.get_xticklabels(),rotation=90)
for p in day.patches:
    height = p.get_height()
    day.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# **OBSERVATION**
# 
# - There are lots of records for the year 2016, contributing to approximately 67%, whereas only approximately 33% for the year 2017.
# - Most of the incidents are recorded during the first 6 months than the last six months of a year.
# - Eighth day of a month has recorded the highest incidents contributing to 6% approx. whereas the least is on 28th day contributing to only 1.4% approx.
# 

# *Country*

# In[25]:


con = pd.DataFrame(df['Country'].values)
# con.value_counts()


# In[26]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(10,5))
con = sns.countplot(x='Country', 
              palette='winter',
              data=df)
con.set_xticklabels(con.get_xticklabels(),rotation=90)
for p in con.patches:
    height = p.get_height()
    con.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# **OBSERVATION**
# 
# - More incidents are recorded for Country_01 which contributes to approximately 59% whereas Country_02 and Country_03 accounts for 31% and 10% respectively.

# *Locality*

# In[27]:


local = pd.DataFrame(df['Local'].values)
# local.value_counts()


# In[28]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(10,5))
local = sns.countplot(x='Local', 
              palette='winter',
              data=df)
local.set_xticklabels(local.get_xticklabels(),rotation=90)
for p in local.patches:
    height = p.get_height()
    local.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# **OBSERVATION**
# 
# - More incidents are recorded for the Local_03 plant which contributes to approximately 21.2% whereas the Local_09 and Local_11 plants account for 0.5% approximately.

# *Industry Sector*

# In[29]:


sec = pd.DataFrame(df['Industry Sector'].values)
# sec.value_counts()


# In[30]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(10,5))
sec = sns.countplot(x='Industry Sector', 
              palette='winter',
              data=df)
sec.set_xticklabels(sec.get_xticklabels(),rotation=90)
for p in sec.patches:
    height = p.get_height()
    sec.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# **OBSERVATION**
# 
# - Most of the incidents are recorded for the ‘Mining’ sector which is 57% approx.. ‘Metals’ sector contributes to almost only half of Mining sector, contributing to 32% approx.. Whereas, the least contribution comes from ‘Others’ industry sector which is 12%.

# *Accident Level*

# In[31]:


acc = pd.DataFrame(df['Accident Level'].values)
# acc.value_counts()


# In[32]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(10,5))
acc = sns.countplot(x='Accident Level', 
              palette='winter',
              data=df)
acc.set_xticklabels(acc.get_xticklabels(),rotation=90)
for p in acc.patches:
    height = p.get_height()
    acc.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# **OBSERVATION**
# 
# - The accident level I dominates with respect to all the rest of the accident levels. Level I contributes the most of all the incidents which is 74% approx.. whereas, the most severe accidents have taken place only 2.5% approx..  

# *Potential Accident Level*

# In[33]:


pot_acc = pd.DataFrame(df['Potential Accident Level'].values)
# pot_acc.value_counts()


# In[34]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(10,5))
pot_acc = sns.countplot(x='Potential Accident Level', 
              palette='winter',
              data=df)
pot_acc.set_xticklabels(pot_acc.get_xticklabels(),rotation=90)
for p in pot_acc.patches:
    height = p.get_height()
    pot_acc.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# **OBSERVATION**
# 
# - The Potential Accident Level IV has been predicted the most number of times leading to 34% approx.. And VI is recorded only once which is 0.02% approx.. 

# *Gender*

# In[35]:


gen = pd.DataFrame(df['Gender'].values)
# gen.value_counts()


# In[36]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(10,5))
gen = sns.countplot(x='Gender', 
              palette='winter',
              data=df)
gen.set_xticklabels(gen.get_xticklabels(),rotation=90)
for p in gen.patches:
    height = p.get_height()
    gen.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# **OBSERVATION**
# 
# - The ‘Male’ gender dominates ‘Female’ gender by 95% approx.. 

# *Employee Type*

# In[37]:


emp = pd.DataFrame(df['Employee type'].values)
# emp.value_counts()


# In[38]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(10,5))
emp = sns.countplot(x='Employee type', 
              palette='winter',
              data=df)
emp.set_xticklabels(emp.get_xticklabels(),rotation=90)
for p in emp.patches:
    height = p.get_height()
    emp.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# **OBSERVATION**
# 
# - Surprisingly, both ‘Third Party’ and ‘Employee’ have performed the activities equal number of times contributing to 44% and 42% respectively. The ‘Third Party (Remote)’ category have occurred only 13% in total.

# *Critical Risk*

# In[39]:


crit_risk = pd.DataFrame(df['Critical Risk'].values)
# crit_risk.value_counts()


# In[40]:


# count plot
# plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(10,5))
crit_risk = sns.countplot(x='Critical Risk', 
              palette='winter',
              data=df)
crit_risk.set_xticklabels(crit_risk.get_xticklabels(),rotation=90)
for p in crit_risk.patches:
    height = p.get_height()
    crit_risk.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# **OBSERVATION**
# 
# - The ‘Others’ category appears to have occurred the maximum number of times, contributing to 55% in total.

# # **Bivariate Analysis**

# **BiVariate Plots**

# *Year and Country*

# In[41]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(30,10))
        return sns.countplot(x, hue=hue, palette='winter');


# In[42]:


vis_countplot(df,df['Year'],df['Country'])


# *Month and Country*

# In[43]:


vis_countplot(df,df['Month'],df['Country'])


# *Day and Country*

# In[44]:


vis_countplot(df,df['Day'],df['Country'])


# *Year and Local*

# In[45]:


vis_countplot(df,df['Local'],df['Year'])


# *Month and Local*

# In[46]:


vis_countplot(df,df['Local'],df['Month'])


# *Day and Local*

# In[47]:


vis_countplot(df,df['Day'],df['Local'])


# *Year and Industry Sector*

# In[48]:


vis_countplot(df,df['Year'],df['Industry Sector'])


# *Month and Industry Sector*

# In[49]:


vis_countplot(df,df['Month'],df['Industry Sector'])


# *Day and Industry Sector*

# In[50]:


vis_countplot(df,df['Day'],df['Industry Sector'])


# *Year and Accident Level*

# In[51]:


vis_countplot(df,df['Year'],df['Accident Level'])


# *Month and Accident Level*

# In[52]:


vis_countplot(df,df['Month'],df['Accident Level'])


# In[146]:


df_2016 = df[df['Year']==2016]
df_2017 = df[df['Year']==2017]


# In[147]:


vis_countplot(df_2016,df_2016['Month'],df_2016['Accident Level'])


# In[148]:


vis_countplot(df_2017,df_2017['Month'],df_2017['Accident Level'])


# *Day and Accident Level*

# In[53]:


vis_countplot(df,df['Day'],df['Accident Level'])


# *Year and Potential Accident Level*

# In[54]:


vis_countplot(df,df['Year'],df['Potential Accident Level'])


# *Month and Potential Accident Level*

# In[55]:


vis_countplot(df,df['Month'],df['Potential Accident Level'])


# *Day and Potential Accident Level*

# In[56]:


vis_countplot(df,df['Day'],df['Potential Accident Level'])


# *Year and Gender*

# In[57]:


vis_countplot(df,df['Year'],df['Gender'])


# *Month and Gender*

# In[58]:


vis_countplot(df,df['Month'],df['Gender'])


# *Day and Gender*

# In[59]:


vis_countplot(df,df['Day'],df['Gender'])


# *Year and Employee type*

# In[60]:


vis_countplot(df,df['Year'],df['Employee type'])


# *Month and Employee type*

# In[61]:


vis_countplot(df,df['Month'],df['Employee type'])


# *Day and Employee type*

# In[62]:


vis_countplot(df,df['Day'],df['Employee type'])


# In[63]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(20,15))
        fig1 = sns.countplot(x, hue=hue, palette='winter');
        fig1.set_xticklabels(fig1.get_xticklabels(),rotation=90)
        # fig2 = countplot.get_figure()
        # fig2.savefig("output.png") 
        # fig1.savefig("output.png")
        # plt.legend(loc='upper right')
        # fig1.fig.legend(loc='upper center')
        # fig._legend.set_bbox_to_anchor((0.5, 0.5))


# *Year and Critical Risk*

# In[64]:


vis_countplot(df,df['Critical Risk'],df['Year'])


# *Month and Critical Risk*

# In[65]:


vis_countplot(df,df['Critical Risk'],df['Month'])


# *Day and Critical Risk*

# In[66]:


vis_countplot(df,df['Critical Risk'],df['Day'])


# In[67]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(15,18))
        fig = sns.countplot(x, hue=hue, palette='winter');
        fig.set_xticklabels(fig.get_xticklabels(),rotation=90)


# *Local and Country*

# In[68]:


vis_countplot(df,df['Local'],df['Country'])


# *Industry Sector and Country*

# In[69]:


vis_countplot(df,df['Country'],df['Industry Sector'])


# *Accident Level and Country*

# In[160]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(16,7))
        fig = sns.countplot(x, hue=hue, palette='winter');
        fig.set_xticklabels(fig.get_xticklabels(),rotation=90)


# In[161]:


vis_countplot(df,df['Country'],df['Accident Level'])


# *Potential Accident Level and Country*

# In[71]:


vis_countplot(df,df['Country'],df['Potential Accident Level'])


# Gender and Country

# In[72]:


vis_countplot(df,df['Country'],df['Gender'])


# *Employee Type and Country*

# In[73]:


vis_countplot(df,df['Country'],df['Employee type'])


# *Critical Risk and Country*

# In[74]:


vis_countplot(df,df['Country'],df['Critical Risk'])


# In[75]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(20,10))
        fig = sns.countplot(x, hue=hue, palette='winter');
        fig.set_xticklabels(fig.get_xticklabels(),rotation=90)


# *Local and Industry Sector*

# In[76]:


vis_countplot(df,df['Local'],df['Industry Sector'])


# *Local and Accident Level*

# In[77]:


vis_countplot(df,df['Local'],df['Accident Level'])


# *Local and Potential Accident Level*

# In[78]:


vis_countplot(df,df['Local'],df['Potential Accident Level'])


# *Local and Gender*

# In[79]:


vis_countplot(df,df['Local'],df['Gender'])


# *Local and Employee/Third Party*

# In[80]:


vis_countplot(df,df['Local'],df['Employee type'])


# *Local and Critical Risk*

# In[81]:


vis_countplot(df,df['Critical Risk'],df['Local'])


# In[82]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(20,8))
        fig = sns.countplot(x, hue=hue, palette='winter');
        fig.set_xticklabels(fig.get_xticklabels(),rotation=90)


# *Industry Sector and Accident Level*

# In[83]:


vis_countplot(df,df['Industry Sector'],df['Accident Level'])


# *Industry Sector and Potential Accident Level*

# In[84]:


vis_countplot(df,df['Industry Sector'],df['Potential Accident Level'])


# *Industry Sector and Gender*

# In[85]:


vis_countplot(df,df['Industry Sector'],df['Gender'])


# *Industry Sector and Employee/Third Party*

# In[86]:


vis_countplot(df,df['Industry Sector'],df['Employee type'])


# *Industry Sector and Critical Risk*

# In[87]:


vis_countplot(df,df['Critical Risk'],df['Industry Sector'])


# In[88]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(20,15))
        fig = sns.countplot(x, hue=hue, palette='winter');
        fig.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        fig.set_xticklabels(fig.get_xticklabels(),rotation=90)


# *Accident Level and Potential Accident Level*

# In[89]:


vis_countplot(df,df['Accident Level'],df['Potential Accident Level'])


# *Accident Level and Gender*

# In[165]:


vis_countplot(df,df['Accident Level'],df['Gender'])


# *Accident Level and Employee/Third Party*

# In[164]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(20,8))
        fig = sns.countplot(x, hue=hue, palette='winter');
        fig.set_xticklabels(fig.get_xticklabels(),rotation=90)


# In[163]:


vis_countplot(df,df['Accident Level'],df['Employee type'])


# *Accident Level and Critical Risk*

# In[92]:


vis_countplot(df,df['Accident Level'],df['Critical Risk'])


# In[93]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(25,10))
        fig = sns.countplot(x, hue=hue, palette='winter');
        fig.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        fig.set_xticklabels(fig.get_xticklabels(),rotation=90)


# *Potential Accident Level and Gender*

# In[94]:


vis_countplot(df,df['Potential Accident Level'],df['Gender'])


# *Potential Accident Level and Employee/Third Party*

# In[95]:


vis_countplot(df,df['Potential Accident Level'],df['Employee type'])


# *Potential Accident Level and Critical Risk*

# In[96]:


vis_countplot(df,df['Critical Risk'],df['Potential Accident Level'])


# In[97]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(20,10))
        fig = sns.countplot(x, hue=hue, palette='winter');
        # fig.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        fig.set_xticklabels(fig.get_xticklabels(),rotation=90)


# *Gender and Employee/Third Party*

# In[98]:


vis_countplot(df,df['Gender'],df['Employee type'])


# *Gender and Critical Risk*

# In[99]:


vis_countplot(df,df['Critical Risk'],df['Gender'])


# In[100]:


def vis_countplot(self,x,hue):
#         plt.style.use("dark_background")
        plt.figure(figsize=(20,10))
        fig = sns.countplot(x, hue=hue, palette='winter');
        # fig.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        fig.set_xticklabels(fig.get_xticklabels(),rotation=90)


# *Employee/Third Party and Critical Risk*

# In[101]:


vis_countplot(df,df['Critical Risk'],df['Employee type'])


# In[102]:


set(df['Critical Risk'].values)


# **Tabular Analysis**

# In[103]:


# Filtering rows with 'Year' equal to 2016

df_2016 = df[df['Year'] == 2016]
df_2016


# In[104]:


# Filtering rows with 'Year' equal to 2017

df_2017 = df[df['Year'] == 2017]
df_2017


# In[105]:


pd.DataFrame(df.groupby(['Accident Level', "Potential Accident Level"]).size())


# In[106]:


pd.DataFrame(df.groupby(['Country','Local']).size())


# In[107]:


pd.DataFrame(df_2016.groupby(['Country','Local']).size())


# In[108]:


pd.DataFrame(df_2017.groupby(['Country','Local']).size())


# In[109]:


pd.DataFrame(df.groupby(['Country','Accident Level', "Potential Accident Level"]).size())


# In[110]:


pd.DataFrame(df.groupby(['Local','Accident Level']).size())


# In[111]:


pd.DataFrame(df_2016.groupby(['Local','Accident Level']).size())


# In[112]:


pd.DataFrame(df_2017.groupby(['Local','Accident Level']).size())


# In[113]:


pd.DataFrame(df.groupby(['Critical Risk','Country']).size())


# In[114]:


pd.DataFrame(df_2016.groupby(['Critical Risk','Country']).size())


# In[115]:


pd.DataFrame(df_2017.groupby(['Critical Risk','Country']).size())


# In[116]:


pd.DataFrame(df.groupby(['Employee type','Country','Accident Level']).size())


# In[117]:


pd.DataFrame(df.groupby(['Employee type','Accident Level']).size())


# In[118]:


pd.DataFrame(df.groupby(['Employee type','Country']).size())


# In[119]:


pd.DataFrame(df.groupby(['Critical Risk','Accident Level']).size()).head(50)


# In[120]:


pd.DataFrame(df.groupby(['Critical Risk','Accident Level']).size()).tail(12)


# In[121]:


pd.DataFrame(df.groupby(['Accident Level']).size())


# In[122]:


pd.DataFrame(df_2016.groupby(['Accident Level']).size())


# In[123]:


pd.DataFrame(df_2017.groupby(['Accident Level']).size())


# In[124]:


pd.DataFrame(df.groupby(['Country','Accident Level']).size())


# In[125]:


pd.DataFrame(df.groupby(['Gender','Accident Level']).size())


# In[126]:


pd.DataFrame(df.groupby(['Gender','Employee type']).size())


# In[127]:


pd.DataFrame(df.groupby(['Gender','Employee type','Accident Level']).size())


# severe accidents IV and V vs Critical Risks

# In[128]:


levels = ['IV','V']
sev_acc= df[df['Accident Level'].isin(levels)]

sev_acc


# In[129]:


pd.DataFrame(sev_acc.groupby(['Industry Sector','Accident Level']).size())


# In[130]:


pd.DataFrame(sev_acc.groupby(['Accident Level', 'Critical Risk']).size())


# **Conclusion**
# 
# - Observations are collected only for the first 7 months in the year 2017.
# - 5 out of 8 most severe accidents took place within 7 months in 2017, whereas it was only 3 in the entire year of 2016. Intuitively, more accidents took place within the first 7 months in 2017 than in 2016, w.r.t all levels of severity of accidents. 
# - Most number of incidents are recorded for Country_01 contributing to almost half of total incidents. Also, the most severe accidents of level ‘V’ took place only in Country_01. 
# - Though only 44 incidents were reported for Country_03, the severity of accident level ‘IV’ is more in Country_03 with 4.5% than in Country_02 which is 3.8%. And, it is 9.1% for Country_01. Thus, more severe accidents took place in Country_01.
# - The employee type of category ‘Employee’ has never caused accidents of severity level ’V’. ‘Third Party’ and ‘Third Party (Remote)’ employees are the victims of accidents with severity level ‘V’, which are 6 and 2 respectively.
# - Country_03 doesn’t have ‘Third Party (Remote)’ employees.
# - Only males are involved in severe accidents.
# - The reasons for most severe accidents are due to fall, pressed, power lock, vehicles, and mobile equipment and remains of choco.
# - Mining industry has resulted in more severe accidents
# - Most affected parts during accidents are right and left hand, face, head, foot, finger, and eye.
# - One important difference noted in word cloud is that ‘Operator’ is mostly used in accident levels, whereas ‘Employee’ in less severe accidents. Accident level V has very few frequently used words to describe the incident.
# 
