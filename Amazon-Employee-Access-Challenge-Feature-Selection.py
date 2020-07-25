#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import category_encoders as ce
from scipy import sparse

from itertools import permutations
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# import warnings
# warnings.filterwarnings('ignore')


# # Amazon Employee Access Challenge

# In[2]:


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[3]:


train.shape


# In[4]:


test.shape


# ## One Hot Encoding

# In[5]:


# One hot encoding of RESOURCE Feature
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(train['RESOURCE'].values.reshape(-1, 1))# Fit has to happen only on train data

train_resource_ohe = ohe.transform(train['RESOURCE'].values.reshape(-1, 1))
test_resource_ohe = ohe.transform(test['RESOURCE'].values.reshape(-1, 1))

print(train_resource_ohe.shape, test_resource_ohe.shape)


# In[6]:


# One hot encoding of MGR_ID Feature
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(train['MGR_ID'].values.reshape(-1, 1))# Fit has to happen only on train data

train_mgr_id_ohe = ohe.transform(train['MGR_ID'].values.reshape(-1, 1))
test_mgr_id_ohe = ohe.transform(test['MGR_ID'].values.reshape(-1, 1))

print(train_mgr_id_ohe.shape, test_mgr_id_ohe.shape)


# In[7]:


# One hot encoding of ROLE_ROLLUP_1 Feature
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(train['ROLE_ROLLUP_1'].values.reshape(-1, 1))# Fit has to happen only on train data

train_role_rollup_1_ohe = ohe.transform(train['ROLE_ROLLUP_1'].values.reshape(-1, 1))
test_role_rollup_1_ohe = ohe.transform(test['ROLE_ROLLUP_1'].values.reshape(-1, 1))

print(train_role_rollup_1_ohe.shape, test_role_rollup_1_ohe.shape)


# In[8]:


# One hot encoding of ROLE_ROLLUP_2 Feature
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(train['ROLE_ROLLUP_2'].values.reshape(-1, 1))# Fit has to happen only on train data

train_role_rollup_2_ohe = ohe.transform(train['ROLE_ROLLUP_2'].values.reshape(-1, 1))
test_role_rollup_2_ohe = ohe.transform(test['ROLE_ROLLUP_2'].values.reshape(-1, 1))

print(train_role_rollup_2_ohe.shape, test_role_rollup_2_ohe.shape)


# In[9]:


# One hot encoding of ROLE_DEPTNAME Feature
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(train['ROLE_DEPTNAME'].values.reshape(-1, 1))# Fit has to happen only on train data

train_role_deptname_ohe = ohe.transform(train['ROLE_DEPTNAME'].values.reshape(-1, 1))
test_role_deptname_ohe = ohe.transform(test['ROLE_DEPTNAME'].values.reshape(-1, 1))

print(train_role_deptname_ohe.shape, test_role_deptname_ohe.shape)


# In[10]:


# One hot encoding of ROLE_TITLE Feature
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(train['ROLE_TITLE'].values.reshape(-1, 1))# Fit has to happen only on train data

train_role_title_ohe = ohe.transform(train['ROLE_TITLE'].values.reshape(-1, 1))
test_role_title_ohe = ohe.transform(test['ROLE_TITLE'].values.reshape(-1, 1))

print(train_role_title_ohe.shape, test_role_title_ohe.shape)


# In[11]:


# One hot encoding of ROLE_FAMILY_DESC Feature
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(train['ROLE_FAMILY_DESC'].values.reshape(-1, 1))# Fit has to happen only on train data

train_role_family_desc_ohe = ohe.transform(train['ROLE_FAMILY_DESC'].values.reshape(-1, 1))
test_role_family_desc_ohe = ohe.transform(test['ROLE_FAMILY_DESC'].values.reshape(-1, 1))

print(train_role_family_desc_ohe.shape, test_role_family_desc_ohe.shape)


# In[12]:


# One hot encoding of ROLE_FAMILY Feature
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(train['ROLE_FAMILY'].values.reshape(-1, 1))# Fit has to happen only on train data

train_role_family_ohe = ohe.transform(train['ROLE_FAMILY'].values.reshape(-1, 1))
test_role_family_ohe = ohe.transform(test['ROLE_FAMILY'].values.reshape(-1, 1))

print(train_role_family_ohe.shape, test_role_family_ohe.shape)


# In[13]:


# One hot encoding of ROLE_CODE Feature
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(train['ROLE_CODE'].values.reshape(-1, 1))# Fit has to happen only on train data

train_role_code_ohe = ohe.transform(train['ROLE_CODE'].values.reshape(-1, 1))
test_role_code_ohe = ohe.transform(test['ROLE_CODE'].values.reshape(-1, 1))

print(train_role_code_ohe.shape, test_role_code_ohe.shape)


# In[14]:


train_ohe = hstack((train_resource_ohe, train_mgr_id_ohe,  train_role_rollup_1_ohe, train_role_rollup_2_ohe, train_role_deptname_ohe, train_role_title_ohe, train_role_family_desc_ohe, train_role_family_ohe, train_role_code_ohe))


# In[15]:


test_ohe = hstack((test_resource_ohe, test_mgr_id_ohe, test_role_rollup_1_ohe, test_role_rollup_2_ohe, test_role_deptname_ohe, test_role_title_ohe, test_role_family_desc_ohe, test_role_family_ohe, test_role_code_ohe))


# In[16]:


y_train_ohe = train['ACTION']


# In[17]:


train_ohe.shape, test_ohe.shape, y_train_ohe.shape


# ## Frequency Encoding

# https://python-data-science.readthedocs.io/en/latest/preprocess.html

# In[18]:


### FREQUENCY ENCODING

# size of each category
# encoding = titanic.groupby('Embarked').size()
# get frequency of each category
# encoding = encoding/len(titanic)
# titanic['enc'] = titanic.Embarked.map(encoding)


# In[19]:


### FREQUENCY ENCODING RESOURCE

# size of each category
encoding = train.groupby('RESOURCE').size()

# get frequency of each category
encoding = encoding/len(train)
train_resource_fc = train.RESOURCE.map(encoding)
test_resource_fc = test.RESOURCE.map(encoding)

print(train_resource_fc.shape, test_resource_fc.shape, train_resource_fc.isna().sum(), test_resource_fc.isna().sum())
# fill missing values
test_resource_fc = test_resource_fc.fillna(0)
print(train_resource_fc.shape, test_resource_fc.shape, train_resource_fc.isna().sum(), test_resource_fc.isna().sum())


# In[20]:


### FREQUENCY ENCODING MGR_ID

# size of each category
encoding = train.groupby('MGR_ID').size()

# get frequency of each category
encoding = encoding/len(train)
train_mgr_id_fc = train.MGR_ID.map(encoding)
test_mgr_id_fc = test.MGR_ID.map(encoding)

print(train_mgr_id_fc.shape, test_mgr_id_fc.shape, train_mgr_id_fc.isna().sum(), test_mgr_id_fc.isna().sum())
# fill missing values
test_mgr_id_fc = test_mgr_id_fc.fillna(0)
print(train_mgr_id_fc.shape, test_mgr_id_fc.shape, train_mgr_id_fc.isna().sum(), test_mgr_id_fc.isna().sum())


# In[21]:


### FREQUENCY ENCODING ROLE_ROLLUP_1

# size of each category
encoding = train.groupby('ROLE_ROLLUP_1').size()

# get frequency of each category
encoding = encoding/len(train)
train_rollup_1_fc = train.ROLE_ROLLUP_1.map(encoding)
test_rollup_1_fc = test.ROLE_ROLLUP_1.map(encoding)

print(train_rollup_1_fc.shape, test_rollup_1_fc.shape, train_rollup_1_fc.isna().sum(), test_rollup_1_fc.isna().sum())
# fill missing values
test_rollup_1_fc = test_rollup_1_fc.fillna(0)
print(train_rollup_1_fc.shape, test_rollup_1_fc.shape, train_rollup_1_fc.isna().sum(), test_rollup_1_fc.isna().sum())


# In[22]:


### FREQUENCY ENCODING ROLE_ROLLUP_2

# size of each category
encoding = train.groupby('ROLE_ROLLUP_2').size()

# get frequency of each category
encoding = encoding/len(train)
train_rollup_2_fc = train.ROLE_ROLLUP_2.map(encoding)
test_rollup_2_fc = test.ROLE_ROLLUP_2.map(encoding)

print(train_rollup_2_fc.shape, test_rollup_2_fc.shape, train_rollup_2_fc.isna().sum(), test_rollup_2_fc.isna().sum())
# fill missing values
test_rollup_2_fc = test_rollup_2_fc.fillna(0)
print(train_rollup_2_fc.shape, test_rollup_2_fc.shape, train_rollup_2_fc.isna().sum(), test_rollup_2_fc.isna().sum())


# In[23]:


### FREQUENCY ENCODING ROLE_DEPTNAME

# size of each category
encoding = train.groupby('ROLE_DEPTNAME').size()

# get frequency of each category
encoding = encoding/len(train)
train_role_deptname_fc = train.ROLE_DEPTNAME.map(encoding)
test_role_deptname_fc = test.ROLE_DEPTNAME.map(encoding)

print(train_role_deptname_fc.shape, test_role_deptname_fc.shape, train_role_deptname_fc.isna().sum(), test_role_deptname_fc.isna().sum())
# fill missing values
test_role_deptname_fc = test_role_deptname_fc.fillna(0)
print(train_role_deptname_fc.shape, test_role_deptname_fc.shape, train_role_deptname_fc.isna().sum(), test_role_deptname_fc.isna().sum())


# In[24]:


### FREQUENCY ENCODING ROLE_TITLE

# size of each category
encoding = train.groupby('ROLE_TITLE').size()

# get frequency of each category
encoding = encoding/len(train)
train_role_title_fc = train.ROLE_TITLE.map(encoding)
test_role_title_fc = test.ROLE_TITLE.map(encoding)

print(train_role_title_fc.shape, test_role_title_fc.shape, train_role_title_fc.isna().sum(), test_role_title_fc.isna().sum())
# fill missing values
test_role_title_fc = test_role_title_fc.fillna(0)
print(train_role_title_fc.shape, test_role_title_fc.shape, train_role_title_fc.isna().sum(), test_role_title_fc.isna().sum())


# In[25]:


### FREQUENCY ENCODING ROLE_FAMILY_DESC

# size of each category
encoding = train.groupby('ROLE_FAMILY_DESC').size()

# get frequency of each category
encoding = encoding/len(train)
train_role_family_desc_fc = train.ROLE_FAMILY_DESC.map(encoding)
test_role_family_desc_fc = test.ROLE_FAMILY_DESC.map(encoding)

print(train_role_family_desc_fc.shape, test_role_family_desc_fc.shape, train_role_family_desc_fc.isna().sum(), test_role_family_desc_fc.isna().sum())
# fill missing values
test_role_family_desc_fc = test_role_family_desc_fc.fillna(0)
print(train_role_family_desc_fc.shape, test_role_family_desc_fc.shape, train_role_family_desc_fc.isna().sum(), test_role_family_desc_fc.isna().sum())


# In[26]:


### FREQUENCY ENCODING ROLE_FAMILY

# size of each category
encoding = train.groupby('ROLE_FAMILY').size()

# get frequency of each category
encoding = encoding/len(train)
train_role_family_fc = train.ROLE_FAMILY.map(encoding)
test_role_family_fc = test.ROLE_FAMILY.map(encoding)

print(train_role_family_fc.shape, test_role_family_fc.shape, train_role_family_fc.isna().sum(), test_role_family_fc.isna().sum())
# fill missing values
test_role_family_fc = test_role_family_fc.fillna(0)
print(train_role_family_fc.shape, test_role_family_fc.shape, train_role_family_fc.isna().sum(), test_role_family_fc.isna().sum())


# In[27]:


### FREQUENCY ENCODING ROLE_CODE

# size of each category
encoding = train.groupby('ROLE_CODE').size()

# get frequency of each category
encoding = encoding/len(train)
train_role_code_fc = train.ROLE_CODE.map(encoding)
test_role_code_fc = test.ROLE_CODE.map(encoding)

print(train_role_code_fc.shape, test_role_code_fc.shape, train_role_code_fc.isna().sum(), test_role_code_fc.isna().sum())
# fill missing values
test_role_code_fc = test_role_code_fc.fillna(0)
print(train_role_code_fc.shape, test_role_code_fc.shape, train_role_code_fc.isna().sum(), test_role_code_fc.isna().sum())


# In[28]:


type(test_role_code_fc[0:10])


# In[29]:


train_df_fc = pd.DataFrame({'resource_fc':train_resource_fc, 'mgr_id_fc':train_mgr_id_fc,'rollup_1_fc':train_rollup_1_fc, 'rollup_2_fc':train_rollup_2_fc, 'role_deptname_fc':train_role_deptname_fc, 'role_title_fc':train_role_title_fc, 'role_family_desc_fc':train_role_family_desc_fc, 'role_family_fc':train_role_family_fc, 'role_code_fc':train_role_code_fc})


# In[30]:


test_df_fc = pd.DataFrame({'resource_fc':test_resource_fc, 'mgr_id_fc':test_mgr_id_fc, 'rollup_1_fc':test_rollup_1_fc, 'rollup_2_fc':test_rollup_2_fc, 'role_deptname_fc':test_role_deptname_fc, 'role_title_fc':test_role_title_fc, 'role_family_desc_fc':test_role_family_desc_fc, 'role_family_fc':test_role_family_fc, 'role_code_fc':test_role_code_fc})


# In[31]:


train_df_fc.shape


# In[32]:


test_df_fc.shape


# In[33]:


train_y_fc = train['ACTION'].values


# In[34]:


train_y_fc.shape


# ## Response Encoding

# https://medium.com/analytics-vidhya/types-of-categorical-data-encoding-schemes-a5bbeb4ba02b

# In[35]:


# sample
data = pd.DataFrame({
    'color' : ['Blue', 'Black', 'Black','Blue', 'Blue'],
    'outcome' : [1,      2,        1,     1,      2,]
})
# column to perform encoding
X = data['color']
Y = data['outcome']
# create an object of the TargetEncoder
ce_TE = ce.TargetEncoder(cols=['color'])
# fit and transform and you will get the encoded data
ce_TE.fit(X,Y)
ce_TE.transform(X)


# In[36]:


### RESPONSE ENCODING RESOURCE

# column to perform encoding
X = train['RESOURCE']
Y = train['ACTION']
# create an object of the TargetEncoder
ce_TE = ce.TargetEncoder(cols=['RESOURCE'])
# fit and transform and you will get the encoded data
ce_TE.fit(X,Y)
train_resource_rc = ce_TE.transform(X)
test_resource_rc = ce_TE.transform(test['RESOURCE'])

print(train_resource_rc.shape, test_resource_rc.shape)


# In[37]:


train_resource_rc[:10]


# In[38]:


### RESPONSE ENCODING MGR_ID

# column to perform encoding
X = train['MGR_ID']
Y = train['ACTION']
# create an object of the TargetEncoder
ce_TE = ce.TargetEncoder(cols=['MGR_ID'])
# fit and transform and you will get the encoded data
ce_TE.fit(X,Y)
train_mgr_id_rc = ce_TE.transform(X)
test_mgr_id_rc = ce_TE.transform(test['MGR_ID'])

print(train_mgr_id_rc.shape, test_mgr_id_rc.shape)


# In[39]:


### RESPONSE ENCODING ROLE_ROLLUP_1

# column to perform encoding
X = train['ROLE_ROLLUP_1']
Y = train['ACTION']
# create an object of the TargetEncoder
ce_TE = ce.TargetEncoder(cols=['ROLE_ROLLUP_1'])
# fit and transform and you will get the encoded data
ce_TE.fit(X,Y)
train_rollup_1_rc = ce_TE.transform(X)
test_rollup_1_rc = ce_TE.transform(test['ROLE_ROLLUP_1'])

print(train_rollup_1_rc.shape, test_rollup_1_rc.shape)


# In[40]:


### RESPONSE ENCODING ROLE_ROLLUP_2

# column to perform encoding
X = train['ROLE_ROLLUP_2']
Y = train['ACTION']
# create an object of the TargetEncoder
ce_TE = ce.TargetEncoder(cols=['ROLE_ROLLUP_2'])
# fit and transform and you will get the encoded data
ce_TE.fit(X,Y)
train_rollup_2_rc = ce_TE.transform(X)
test_rollup_2_rc = ce_TE.transform(test['ROLE_ROLLUP_2'])

print(train_rollup_2_rc.shape, test_rollup_2_rc.shape)


# In[41]:


### RESPONSE ENCODING ROLE_DEPTNAME

# column to perform encoding
X = train['ROLE_DEPTNAME']
Y = train['ACTION']
# create an object of the TargetEncoder
ce_TE = ce.TargetEncoder(cols=['ROLE_DEPTNAME'])
# fit and transform and you will get the encoded data
ce_TE.fit(X,Y)
train_role_deptname_rc = ce_TE.transform(X)
test_role_deptname_rc = ce_TE.transform(test['ROLE_DEPTNAME'])

print(train_role_deptname_rc.shape, test_role_deptname_rc.shape)


# In[42]:


### RESPONSE ENCODING ROLE_TITLE

# column to perform encoding
X = train['ROLE_TITLE']
Y = train['ACTION']
# create an object of the TargetEncoder
ce_TE = ce.TargetEncoder(cols=['ROLE_TITLE'])
# fit and transform and you will get the encoded data
ce_TE.fit(X,Y)
train_role_title_rc = ce_TE.transform(X)
test_role_title_rc = ce_TE.transform(test['ROLE_TITLE'])

print(train_role_title_rc.shape, test_role_title_rc.shape)


# In[43]:


### RESPONSE ENCODING ROLE_FAMILY_DESC

# column to perform encoding
X = train['ROLE_FAMILY_DESC']
Y = train['ACTION']
# create an object of the TargetEncoder
ce_TE = ce.TargetEncoder(cols=['ROLE_FAMILY_DESC'])
# fit and transform and you will get the encoded data
ce_TE.fit(X,Y)
train_role_family_desc_rc = ce_TE.transform(X)
test_role_family_desc_rc = ce_TE.transform(test['ROLE_FAMILY_DESC'])

print(train_role_family_desc_rc.shape, test_role_family_desc_rc.shape)


# In[44]:


### RESPONSE ENCODING ROLE_FAMILY

# column to perform encoding
X = train['ROLE_FAMILY']
Y = train['ACTION']
# create an object of the TargetEncoder
ce_TE = ce.TargetEncoder(cols=['ROLE_FAMILY'])
# fit and transform and you will get the encoded data
ce_TE.fit(X,Y)
train_role_family_rc = ce_TE.transform(X)
test_role_family_rc = ce_TE.transform(test['ROLE_FAMILY'])

print(train_role_family_rc.shape, test_role_family_rc.shape)


# In[45]:


### RESPONSE ENCODING ROLE_CODE

# column to perform encoding
X = train['ROLE_CODE']
Y = train['ACTION']
# create an object of the TargetEncoder
ce_TE = ce.TargetEncoder(cols=['ROLE_CODE'])
# fit and transform and you will get the encoded data
ce_TE.fit(X,Y)
train_role_code_rc = ce_TE.transform(X)
test_role_code_rc = ce_TE.transform(test['ROLE_CODE'])

print(train_role_code_rc.shape, test_role_code_rc.shape)


# In[46]:


train_df_rc = pd.DataFrame ({'resource_rc':train_resource_rc['RESOURCE'],'mgr_id_rc':train_mgr_id_rc['MGR_ID'], 'rollup_1_rc':train_rollup_1_rc['ROLE_ROLLUP_1'],  'rollup_2_rc':train_rollup_2_rc['ROLE_ROLLUP_2'], 'role_deptname_rc':train_role_deptname_rc['ROLE_DEPTNAME'], 'role_title_rc':train_role_title_rc['ROLE_TITLE'], 'role_family_desc_rc':train_role_family_desc_rc['ROLE_FAMILY_DESC'], 'role_family_rc':train_role_family_rc['ROLE_FAMILY'], 'role_code_rc':train_role_code_rc['ROLE_CODE']})


# In[47]:


test_df_rc = pd.DataFrame ({'resource_rc':test_resource_rc['RESOURCE'],'mgr_id_rc':test_mgr_id_rc['MGR_ID'], 'rollup_1_rc':test_rollup_1_rc['ROLE_ROLLUP_1'],  'rollup_2_rc':test_rollup_2_rc['ROLE_ROLLUP_2'], 'role_deptname_rc':test_role_deptname_rc['ROLE_DEPTNAME'], 'role_title_rc':test_role_title_rc['ROLE_TITLE'], 'role_family_desc_rc':test_role_family_desc_rc['ROLE_FAMILY_DESC'], 'role_family_rc':test_role_family_rc['ROLE_FAMILY'], 'role_code_rc':test_role_code_rc['ROLE_CODE']})


# In[48]:


train_df_rc


# In[49]:


test_df_rc


# In[50]:


train_y_rc = train['ACTION'].values


# In[51]:


train_y_rc.shape


# # Feature Engineering

# ## Encoding with Singular Value Decomposition
# 
# Here I'll use singular value decomposition (SVD) to learn encodings from pairs of categorical features. SVD is one of the more complex encodings, but it can also be very effective. We'll construct a matrix of co-occurences for each pair of categorical features. Each row corresponds to a value in feature A, while each column corresponds to a value in feature B. Each element is the count of rows where the value in A appears together with the value in B.
# 
# You then use singular value decomposition to find two smaller matrices that equal the count matrix when multiplied.

# In[52]:


#https://www.kaggle.com/dmitrylarko/kaggledays-sf-2-amazon-unsupervised-encoding#SVD-Encoding
#https://www.kaggle.com/matleonard/encoding-categorical-features-with-svd


# In[53]:


train_data=train.drop(columns=['ACTION'],axis=1)


# In[54]:


train_data.shape


# In[55]:


train_data.nunique()


# In[56]:


test_data=test.drop(columns=['id'],axis=1)


# In[57]:


test_data.shape


# In[58]:


test_data.nunique()


# In[59]:


train_svd = pd.DataFrame()
test_svd = pd.DataFrame()


# In[60]:


temp = train_data.groupby(['ROLE_ROLLUP_1','ROLE_ROLLUP_2'])['ROLE_ROLLUP_1'].count()
temp=temp.unstack(fill_value=0)


# In[61]:


temp


# In[62]:


temp = train_data.groupby(['RESOURCE','MGR_ID'])['MGR_ID'].count()
temp=temp.unstack(fill_value=0)


# In[63]:


temp


# In[64]:


train_data.columns


# In[65]:


for col1,col2 in tqdm(permutations(train_data.columns,2)):
    res_train=(train_data.groupby([col1,col2])[col2].count()) 
    res_train=res_train.unstack(fill_value=0)

    svd=TruncatedSVD(n_components=1,random_state=42,).fit(res_train)
    val_train=svd.transform(res_train)
    val_train = pd.DataFrame(val_train)
    val_train = val_train.set_index(res_train.index)
    
    train_svd[col1+'_'+col2]=train[col1].map(val_train.iloc[:,0])
    test_svd[col1+'_'+col2]=test[col1].map(val_train.iloc[:,0])


# In[66]:


train_svd.shape,test_svd.shape


# In[67]:


train_svd.fillna(0,inplace=True)
test_svd.fillna(0,inplace=True)
print(train_svd.isna().sum().values)
print(test_svd.isna().sum().values)


# In[68]:


train_svd.head()


# ### Normalizing the data

# In[69]:


from sklearn.preprocessing import Normalizer
columns = (train_svd.columns)
x_vals1=train_svd[columns]
x_vals2=test_svd[columns]
n=Normalizer()
n.fit(x_vals1)
x_vals1 = n.transform(x_vals1)
train_svd = pd.DataFrame(x_vals1,columns=columns)
x_vals2 = n.transform(x_vals2)
test_svd = pd.DataFrame(x_vals2,columns=columns)


# In[70]:


train_svd.shape,test_svd.shape


# In[71]:


train_svd.head()


# In[72]:


test_svd.head()


# In[73]:


# Save data into csv files


# In[74]:


train_df_fc.to_csv('data/train_df_fc.csv', index=False)
test_df_fc.to_csv('data/test_df_fc.csv', index=False)

train_df_rc.to_csv('data/train_df_rc.csv', index=False)
test_df_rc.to_csv('data/test_df_rc.csv', index=False)

train_svd.to_csv('data/train_svd.csv', index=False)
test_svd.to_csv('data/test_svd.csv', index=False)


# In[75]:


# feature selection for one hot encoding
train_ohe.shape, test_ohe.shape, y_train_ohe.shape


# In[76]:


from sklearn.feature_selection import SelectKBest,chi2
ktop = SelectKBest(chi2,k=4500).fit(train_ohe,y_train_ohe)
train_ohe=ktop.transform(train_ohe)
test_ohe=ktop.transform(test_ohe)


# In[77]:


train_ohe.shape, test_ohe.shape, y_train_ohe.shape


# In[78]:


sparse.save_npz('data/train_ohe.npz', train_ohe)
sparse.save_npz('data/test_ohe.npz', test_ohe)


# In[ ]:




