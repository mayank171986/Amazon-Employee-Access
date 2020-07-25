#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# # Amazon Employee Access Challenge

# ## Overview
# 
# When an employee at any company starts work, they first need to obtain the computer access necessary to fulfill their role. This access may allow an employee to read/manipulate resources through various applications or web portals. It is assumed that employees fulfilling the functions of a given role will access the same or similar resources. It is often the case that employees figure out the access they need as they encounter roadblocks during their daily work (e.g. not able to log into a reporting portal). A knowledgeable supervisor then takes time to manually grant the needed access in order to overcome access obstacles. As employees move throughout a company, this access discovery/recovery cycle wastes a nontrivial amount of time and money.
# 
# There is a considerable amount of data regarding an employee’s role within an organization and the resources to which they have access. Given the data related to current employees and their provisioned access, models can be built that automatically determine access privileges as employees enter and leave roles within a company. These auto-access models seek to minimize the human involvement required to grant or revoke employee access.

# ## Objective
# 
# The objective of this competition is to build a model, learned using historical data, that will determine an employee's access needs, such that manual access transactions (grants and revokes) are minimized as the employee's attributes change over time. The model will take an employee's role information and a resource code and will return whether or not access should be granted.

# ## ML Problem
# 
# So our aim is to develop a Machine Learning model that takes an employee’s access request as input which contains details about the employee’s attributes like role, department etc.. and the model has to decide whether to provide access or not. Here the dataset provided by Amazon contains real historic data collected from 2010 and 2011.The Performance metric used in this case study is AUC score.

# ## Data Information

# https://www.kaggle.com/c/amazon-employee-access-challenge/data

# ### Data Description
# 
# The data consists of real historical data collected from 2010 & 2011.  Employees are manually allowed or denied access to resources over time. You must create an algorithm capable of learning from this historical data to predict approval/denial for an unseen set of employees. 

# ### File Descriptions
# 
# train.csv - The training set. Each row has the ACTION (ground truth), RESOURCE, and information about the employee's role at the time of approval
# 
# test.csv - The test set for which predictions should be made.  Each row asks whether an employee having the listed characteristics should have access to the listed resource.

# ### Column Descriptions
# 
# 
# <table>
#     <tr>
#         <td><b>Column Name</b></td>
#         <td><b>Description</b></td>
#     </tr>
#     <tr>
#         <td>ACTION</td>
#         <td>ACTION is 1 if the resource was approved, 0 if the resource was not</td>
#     </tr>
#     <tr>
#         <td>RESOURCE</td>
#         <td>An ID for each resource</td>
#     </tr>
#     <tr>
#         <td>MGR_ID</td>
#         <td>The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time</td>
#     </tr>
#     <tr>
#         <td>ROLE_ROLLUP_1</td>
#         <td>Company role grouping category id 1 (e.g. US Engineering)</td>
#     </tr>
#     <tr>
#         <td>ROLE_ROLLUP_2</td>
#         <td>Company role grouping category id 2 (e.g. US Retail)</td>
#     </tr>
#     <tr>
#         <td>ROLE_DEPTNAME</td>
#         <td>Company role department description (e.g. Retail)</td>
#     </tr>
#     <tr>
#         <td>ROLE_TITLE</td>
#         <td>Company role business title description (e.g. Senior Engineering Retail Manager)</td>
#     </tr>
#     <tr>
#         <td>ROLE_FAMILY_DESC</td>
#         <td>Company role family extended description (e.g. Retail Manager, Software Engineering)</td>
#     </tr>
#     <tr>
#         <td>ROLE_FAMILY</td>
#         <td>Company role family description (e.g. Retail Manager)</td>
#     </tr>
#     <tr>
#         <td>ROLE_CODE</td>
#         <td>Company role code; this code is unique to each role (e.g. Manager)</td>
#     </tr>
# </table>

# # Data Analysis

# In[2]:


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[3]:


train.shape


# In[4]:


test.shape


# #### Train Data Analysis

# In[5]:


train.columns


# In[6]:


train.info()


# In[7]:


train.head()


# In[8]:


train.describe()


# In[9]:


# unique values
for i in train:
    print(i, len(train[i].unique()))


# ROLE_TITLE and ROLE_CODE columns has same no. of entries, In other words we can say both columns are same. ACTION is our class label.

# In[10]:


train.isna().sum()


# In[11]:


train.duplicated().sum()


# There is no duplicated and missing values in the train dataset

# #### Test Data Analysis

# In[12]:


test.columns


# In[13]:


test.info()


# In[14]:


test.head()


# In[15]:


test.describe()


# In[16]:


# unique value
for i in test:
    print(i, len(test[i].unique()))


# In[17]:


test.isna().sum()


# In[18]:


test.duplicated().sum()


# There is no duplicated and missing values in the test dataset

# ### Analysing Individual Columns

# ACTION

# In[19]:


train['ACTION'].value_counts()


# In[20]:


approved_actions = train[train.ACTION==1]


# In[21]:


rejected_actions = train[train.ACTION==0]


# In[22]:


approved_actions.shape


# In[23]:


rejected_actions.shape


# In[24]:


plt.figure(figsize=(9,6));
sb.countplot(x='ACTION',data=train);
plt.title('Count of values for ACTION variable');


# As per the graph we have imbalanced data set, frequency of approved requests are much greater than rejected one. So we have to find out some ways to make this dataset balance.

# RESOURCE

# In[25]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['RESOURCE'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['RESOURCE'],label='Rejected',shade=True);
plt.title('Distribution of RESOURCE variable');
plt.xlabel('RESOURCE');
plt.ylabel('Probability Density');


# In[26]:


# Top five approved requests
approved_actions['RESOURCE'].value_counts()[:5]


# In[27]:


# Another Top five approved requests
approved_actions['RESOURCE'].value_counts()[5:10]


# In[28]:


# Top five rejected requests
rejected_actions['RESOURCE'].value_counts()[:5]


# In[29]:


# Another Top five rejected requests
rejected_actions['RESOURCE'].value_counts()[5:10]


# In[30]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['RESOURCE'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['RESOURCE'],label='Rejected',shade=True);
plt.title('Distribution of RESOURCE variable');
plt.xlim(0,100000)
plt.xlabel('RESOURCE');
plt.ylabel('Probability Density');


# Looking at above KDE plot we can say that b/w 70K-90K Approved requests are higher than the rejected ones

# MGR_ID

# In[31]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['MGR_ID'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['MGR_ID'],label='Rejected',shade=True);
plt.title('Distribution of MGR_ID variable');
plt.xlabel('MGR_ID');
plt.ylabel('Probability Density');


# In[32]:


# Top 5 Approved Actions for attribute MGR_ID
approved_actions['MGR_ID'].value_counts()[:5]


# In[33]:


# Top 5 Rejected Actions for attribute MGR_ID
rejected_actions['MGR_ID'].value_counts()[:5]


# In[34]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['MGR_ID'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['MGR_ID'],label='Rejected',shade=True);
plt.title('Distribution of MGR_ID variable');
plt.xlim(0,100000)
plt.xlabel('MGR_ID');
plt.ylabel('Probability Density');


# Looking at above KDE plot we can say that b/w 0-20K Approved requests are higher than the rejected ones

# ROLE_ROLLUP_1

# In[35]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_ROLLUP_1'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_ROLLUP_1'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_ROLLUP_1 variable');
plt.xlabel('ROLE_ROLLUP_1');
plt.ylabel('Probability Density');


# In[36]:


# Top 5 Approved Actions for attribute ROLE_ROLLUP_1
approved_actions['ROLE_ROLLUP_1'].value_counts()[:5]


# In[37]:


# Top 5 Rejected Actions for attribute ROLE_ROLLUP_1
rejected_actions['ROLE_ROLLUP_1'].value_counts()[:5]


# In[38]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_ROLLUP_1'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_ROLLUP_1'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_ROLLUP_1 variable');
plt.xlim(100000,150000)
plt.xlabel('ROLE_ROLLUP_1');
plt.ylabel('Probability Density');


# Looking at above KDE plot we can say that trends are almost similar

# ROLE_ROLLUP_2

# In[39]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_ROLLUP_2'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_ROLLUP_2'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_ROLLUP_2 variable');
plt.xlabel('ROLE_ROLLUP_2');
plt.ylabel('Probability Density');


# In[40]:


# Top 5 Approved Actions for attribute ROLE_ROLLUP_2
approved_actions['ROLE_ROLLUP_2'].value_counts()[:5]


# In[41]:


# Top 5 Rejected Actions for attribute ROLE_ROLLUP_2
rejected_actions['ROLE_ROLLUP_2'].value_counts()[:5]


# In[42]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_ROLLUP_2'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_ROLLUP_2'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_ROLLUP_2 variable');
plt.xlim(100000, 150000)
plt.xlabel('ROLE_ROLLUP_2');
plt.ylabel('Probability Density');


# Looking at above KDE plot we can say that b/w 110K-120K Approved requests are higher than the rejected ones

# ROLE_DEPTNAME

# In[43]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_DEPTNAME'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_DEPTNAME'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_DEPTNAME variable');
plt.xlabel('ROLE_DEPTNAME');
plt.ylabel('Probability Density');


# In[44]:


# Top 5 Approved Actions for attribute ROLE_DEPTNAME
approved_actions['ROLE_DEPTNAME'].value_counts()[:5]


# In[45]:


# Top 5 Rejected Actions for attribute ROLE_DEPTNAME
rejected_actions['ROLE_DEPTNAME'].value_counts()[:5]


# In[46]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_DEPTNAME'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_DEPTNAME'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_DEPTNAME variable');
plt.xlim(100000, 150000)
plt.xlabel('ROLE_DEPTNAME');
plt.ylabel('Probability Density');


# Looking at above KDE plot we can say that b/w 110K-130K Approved requests are higher than the rejected ones

# ROLE_TITLE

# In[47]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_TITLE'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_TITLE'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_TITLE variable');
plt.xlabel('ROLE_TITLE');
plt.ylabel('Probability Density');


# In[48]:


# Top 5 Approved Actions for attribute ROLE_TITLE
approved_actions['ROLE_TITLE'].value_counts()[:5]


# In[49]:


# Top 5 Rejected Actions for attribute ROLE_TITLE
rejected_actions['ROLE_TITLE'].value_counts()[:5]


# In[50]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_TITLE'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_TITLE'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_TITLE variable');
plt.xlim(100000, 150000)
plt.xlabel('ROLE_TITLE');
plt.ylabel('Probability Density');


# Looking at above KDE plot we can say that b/w 110K-130K Approved requests are higher than the rejected ones

# ROLE_FAMILY_DESC

# In[51]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_FAMILY_DESC'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_FAMILY_DESC'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_FAMILY_DESC variable');
plt.xlabel('ROLE_FAMILY_DESC');
plt.ylabel('Probability Density');


# In[52]:


# Top 5 Approved Actions for attribute ROLE_FAMILY_DESC
approved_actions['ROLE_FAMILY_DESC'].value_counts()[:5]


# In[53]:


# Top 5 Rejected Actions for attribute ROLE_FAMILY_DESC
rejected_actions['ROLE_FAMILY_DESC'].value_counts()[:5]


# In[54]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_FAMILY_DESC'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_FAMILY_DESC'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_FAMILY_DESC variable');
plt.xlim(100000, 250000)
plt.xlabel('ROLE_FAMILY_DESC');
plt.ylabel('Probability Density');


# Looking at above KDE plot we can say that b/w 100K-140K Approved requests are higher than the rejected ones

# ROLE_FAMILY

# In[55]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_FAMILY'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_FAMILY'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_FAMILY variable');
plt.xlabel('ROLE_FAMILY');
plt.ylabel('Probability Density');


# In[56]:


# Top 5 Approved Actions for attribute ROLE_FAMILY
approved_actions['ROLE_FAMILY'].value_counts()[:5]


# In[57]:


# Top 5 Rejected Actions for attribute ROLE_FAMILY
rejected_actions['ROLE_FAMILY'].value_counts()[:5]


# In[58]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_FAMILY'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_FAMILY'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_FAMILY variable');
plt.xlim(100000, 200000)
plt.xlabel('ROLE_FAMILY');
plt.ylabel('Probability Density');


# Looking at above KDE plot we can say that b/w 100K-140K Approved requests are higher than the rejected ones

# In[59]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_FAMILY'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_FAMILY'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_FAMILY variable');
plt.xlim(200000, 300000)
plt.xlabel('ROLE_FAMILY');
plt.ylabel('Probability Density');


# Looking at above KDE plot we can say that b/w 260K-300K Approved requests are higher than the rejected ones

# ROLE_CODE

# In[60]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_CODE'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_CODE'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_CODE variable');
plt.xlabel('ROLE_CODE');
plt.ylabel('Probability Density');


# In[61]:


# Top 5 Approved Actions for attribute ROLE_CODE
approved_actions['ROLE_CODE'].value_counts()[:5]


# In[62]:


# Top 5 Rejected Actions for attribute ROLE_CODE
rejected_actions['ROLE_CODE'].value_counts()[:5]


# In[63]:


plt.figure(figsize=(15,6));
sb.kdeplot(approved_actions['ROLE_CODE'],label='Accepted',shade=True);
sb.kdeplot(rejected_actions['ROLE_CODE'],label='Rejected',shade=True);
plt.title('Distribution of ROLE_CODE variable');
plt.xlim(100000, 150000)
plt.xlabel('ROLE_CODE');
plt.ylabel('Probability Density');


# Looking at above KDE plot we can say that b/w trends are almost similar for both classes

# In[64]:


train[['ACTION', 'RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
       'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
       'ROLE_CODE']].hist(figsize=(13,10),bins=20,color='Y')


# In[65]:


plt.figure(figsize=(15,6));
sb.heatmap(train.corr(), annot=True, fmt='.2f');
plt.title('Heat Map between all features');


# ### Observation

# 1. Almost all values are 0 expect corelation b/w (ROLE_FAMILY_DESC, ROLE_TITLE) and (ROLE_CODE, ROLE_TITLE)
# 2. Corelation b/w ROLE_FAMILY_DESC and ROLE_TITLE is 0.17
# 3. Corelation b/w ROLE_CODE and ROLE_TITLE is 0.16

# In[67]:


plt.figure(figsize=(15,6))
sb.pairplot(train[['ACTION', 'RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
       'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
       'ROLE_CODE']])


# ### Observation:

# There is only relationship b/w ROLE_CODE and ROLE_TITLE

# In[ ]:




