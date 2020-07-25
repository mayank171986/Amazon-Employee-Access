#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import warnings
# warnings.filterwarnings('ignore')


# In[2]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pickle


# # Amazon Employee Access Challenge

# In[3]:


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[4]:


train.shape


# In[5]:


test.shape


# In[6]:


y_train = train['ACTION']


# In[7]:


y_train.shape


# In[8]:


train_data = train.drop('ACTION', axis=1)
train_data.shape


# In[9]:


test_data = test.drop('id', axis=1)
test_data.shape


# ## Common Variables

# In[10]:


# define variables
random_state = 42
cv = 5
scoring = 'roc_auc'
verbose=2


# ## Common functions

# In[11]:


def save_submission(predictions, filename):
    '''
    Save predictions into csv file
    '''
    global test
    submission = pd.DataFrame()
    submission["Id"] = test["id"]
    submission["ACTION"] = predictions
    filepath = "result/sampleSubmission_"+filename
    submission.to_csv(filepath, index = False)


# In[12]:


def print_graph(results, param1, param2, xlabel, ylabel, title='Plot showing the ROC_AUC score for various hyper parameter values'):
    '''
    Plot the graph
    '''
    plt.plot(results[param1],results[param2]);
    plt.grid();
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.title(title);  


# In[13]:


def get_rf_params():
    '''
    Return dictionary of parameters for random forest
    '''
    params = {
         'n_estimators':[10,20,50,100,200,500,700,1000],
         'max_depth':[1,2,5,10,12,15,20,25],
         'max_features':[1,2,3,4,5],
         'min_samples_split':[2,5,7,10,20]
    }

    return params


# In[14]:


def get_xgb_params():
    '''
    Return dictionary of parameters for xgboost
    '''
    params = {
        'n_estimators': [10,20,50,100,200,500,750,1000],
        'learning_rate': uniform(0.01, 0.6),
        'subsample': uniform(),
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'colsample_bytree': uniform(),
        'min_child_weight': [1, 2, 3, 4]
    }
    
    return params


# ### We will try following models
# 
# 1. KNN
# 2. SVM
# 3. Logistic Regression
# 4. Random Forest
# 5. Xgboost

# ## Build Models on the raw data

# ## 1.1 KNN with raw features

# In[15]:


parameters={'n_neighbors':np.arange(1,100, 5)}
clf = RandomizedSearchCV(KNeighborsClassifier(n_jobs=-1),parameters,random_state=random_state,cv=cv,verbose=verbose,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_data,y_train)


# In[16]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_n_neighbors')
results


# In[17]:


print_graph(results, 'param_n_neighbors', 'mean_test_score', 'Hyperparameter - No. of neighbors', 'Test score')  


# In[18]:


best_c=best_model.best_params_['n_neighbors']
best_c


# In[19]:


model = KNeighborsClassifier(n_neighbors=best_c,n_jobs=-1)
model.fit(train_data,y_train)


# In[20]:


predictions = model.predict_proba(test_data)[:,1]
save_submission(predictions, "knn_raw.csv")


# ![knn-raw](images/knn-raw-new.png)

# ## 1.2 SVM with raw feature

# In[21]:


C_val = uniform(loc=0, scale=4)
model= LinearSVC(verbose=verbose,random_state=random_state,class_weight='balanced',max_iter=2000)
parameters={'C':C_val}
clf = RandomizedSearchCV(model,parameters,random_state=random_state,cv=cv,verbose=verbose,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_data,y_train)


# In[22]:


best_c=best_model.best_params_['C']
best_c


# In[23]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_C')
results


# In[24]:


print_graph(results, 'param_C', 'mean_test_score', 'Hyperparameter - C', 'Test score')


# In[25]:


#https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
model = LinearSVC(C=best_c,verbose=verbose,random_state=random_state,class_weight='balanced',max_iter=2000)
model = CalibratedClassifierCV(model)
model.fit(train_data,y_train)


# In[26]:


predictions = model.predict_proba(test_data)[:,1]
save_submission(predictions, 'svm_raw.csv')


# ![svm-raw](images/svm-raw.png)

# ## 1.3 Logistic Regression with Raw Feature

# In[27]:


C_val = uniform(loc=0, scale=4)
lr= LogisticRegression(verbose=verbose,random_state=random_state,class_weight='balanced',solver='lbfgs',max_iter=500,n_jobs=-1)
parameters={'C':C_val}
clf = RandomizedSearchCV(lr,parameters,random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_data,y_train)


# In[28]:


best_c=best_model.best_params_['C']
best_c


# In[29]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_C')
results


# In[30]:


print_graph(results, 'param_C', 'mean_test_score', 'Hyperparameter - C', 'Test score')


# In[31]:


model = LogisticRegression(C=best_c,verbose=verbose,n_jobs=-1,random_state=random_state,class_weight='balanced',solver='lbfgs')
model.fit(train_data,y_train)


# In[32]:


predictions = model.predict_proba(test_data)[:,1]
save_submission(predictions, 'lr_raw.csv')


# ![lr-raw](images/lr-raw.png)

# ## 1.4 Random Forest with Raw Feature

# In[33]:


rfc = RandomForestClassifier(random_state=random_state,class_weight='balanced',n_jobs=-1)
clf = RandomizedSearchCV(rfc,get_rf_params(),random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_data,y_train)


# In[34]:


results = pd.DataFrame(best_model.cv_results_)
results.sort_values('mean_test_score',ascending=False,inplace=True)
param_keys=['param_'+str(each) for each in get_rf_params().keys()]
param_keys.append('mean_test_score')
results[param_keys].head(10)


# In[35]:


n_estimators=clf.best_params_['n_estimators']
max_features=clf.best_params_['max_features']
max_depth=clf.best_params_['max_depth']
min_samples_split=clf.best_params_['min_samples_split']
n_estimators,max_features,max_depth,min_samples_split


# In[36]:


model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,
                             min_samples_split=min_samples_split,
                             random_state=random_state,class_weight='balanced',n_jobs=-1)

model.fit(train_data,y_train)


# In[37]:


features=train_data.columns
importance=model.feature_importances_
features=pd.DataFrame({'features':features,'value':importance})
features=features.sort_values('value',ascending=False)
sns.barplot('value','features',data=features);
plt.title('Feature Importance');


# ## Features Observations:
# 
# 1. MGR_ID is the most important feature followed by RESOURCE and ROLE_DEPTNAME

# In[38]:


predictions = model.predict_proba(test_data)[:,1]
save_submission(predictions, 'rf_raw.csv')


# ![rf-raw](images/rf-raw.png)

# ## 1.5 Xgboost with Raw Feature

# In[39]:


xgb = XGBClassifier()
clf = RandomizedSearchCV(xgb,get_xgb_params(),random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model=clf.fit(train_data,y_train)


# In[40]:


results = pd.DataFrame(best_model.cv_results_)
results.sort_values('mean_test_score',ascending=False,inplace=True)
param_keys=['param_'+str(each) for each in get_xgb_params().keys()]
param_keys.append('mean_test_score')
results[param_keys].head(10)


# In[41]:


colsample_bytree = clf.best_params_['colsample_bytree']
learning_rate=clf.best_params_['learning_rate']
max_depth=clf.best_params_['max_depth']
min_child_weight=clf.best_params_['min_child_weight']
n_estimators=clf.best_params_['n_estimators']
subsample=clf.best_params_['subsample']
colsample_bytree,learning_rate,max_depth,min_child_weight,n_estimators,subsample


# In[42]:


model = XGBClassifier(colsample_bytree=colsample_bytree,learning_rate=learning_rate,max_depth=max_depth,
                     min_child_weight=min_child_weight,n_estimators=n_estimators,subsample=subsample,n_jobs=-1)

model.fit(train_data,y_train)


# In[43]:


features=train_data.columns
importance=model.feature_importances_
features=pd.DataFrame({'features':features,'value':importance})
features=features.sort_values('value',ascending=False)
sns.barplot('value','features',data=features);
plt.title('Feature Importance');


# In[44]:


predictions = model.predict_proba(test_data)[:,1]
save_submission(predictions, 'xgb_raw.csv')


# ![xgb-raw](images/xgb-raw.png)

# ![kaggle-submission-raw](images/kaggle-submission-raw.png)

# In[45]:


from prettytable import PrettyTable

x = PrettyTable(['Model', 'Feature', 'Private Score', 'Public Score'])
x.add_row(['KNN','Raw', 0.67224, 0.68148])
x.add_row(['SVM', 'Raw', 0.50286, 0.51390])
x.add_row(['Logistic Regression', 'Raw', 0.53857, 0.53034])
x.add_row(['Random Forest', 'Raw', 0.87269, 0.87567])
x.add_row(['Xgboost', 'Raw', 0.86988, 0.87909])

print(x)


# # Observations:
# 
# 1. Xgboost perform best on the raw features
# 2. Random forest also perform good on raw features
# 3. Tree based models performs better than linear models for raw features

# ## Build model on one hot encoded features

# ### 2.1 KNN with one hot encoded features

# In[46]:


train_ohe = sparse.load_npz('data/train_ohe.npz')
test_ohe = sparse.load_npz('data/test_ohe.npz')

train_ohe.shape, test_ohe.shape, y_train.shape


# In[47]:


parameters={'n_neighbors':np.arange(1,100, 5)}
clf = RandomizedSearchCV(KNeighborsClassifier(n_jobs=-1),parameters,random_state=random_state,cv=cv,verbose=verbose,scoring=scoring,n_jobs=4)
best_model = clf.fit(train_ohe,y_train)


# In[48]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_n_neighbors')
results


# In[49]:


print_graph(results, 'param_n_neighbors', 'mean_test_score', 'Hyperparameter - No. of neighbors', 'Test score')  


# In[50]:


best_c=best_model.best_params_['n_neighbors']
best_c


# In[51]:


model = KNeighborsClassifier(n_neighbors=best_c,n_jobs=-1)
model.fit(train_ohe,y_train)


# In[52]:


predictions = model.predict_proba(test_ohe)[:,1]
save_submission(predictions, "knn_ohe.csv")


# ![knn-ohe](images/knn-ohe.png)

# ## 2.2 SVM with one hot encoded features

# In[53]:


C_val = uniform(loc=0, scale=4)
model= LinearSVC(verbose=verbose,random_state=random_state,class_weight='balanced',max_iter=2000)
parameters={'C':C_val}
clf = RandomizedSearchCV(model,parameters,random_state=random_state,cv=cv,verbose=verbose,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_ohe,y_train)


# In[54]:


best_c=best_model.best_params_['C']
best_c


# In[55]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_C')
results


# In[56]:


print_graph(results, 'param_C', 'mean_test_score', 'Hyperparameter - C', 'Test score')


# In[57]:


#https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
model = LinearSVC(C=best_c,verbose=verbose,random_state=random_state,class_weight='balanced',max_iter=2000)
model = CalibratedClassifierCV(model)
model.fit(train_ohe,y_train)


# In[58]:


predictions = model.predict_proba(test_ohe)[:,1]
save_submission(predictions, 'svm_ohe.csv')


# ![svm-ohe](images/svm-ohe.png)

# ## 2.3 Logistic Regression with one hot encoded features

# In[59]:


C_val = uniform(loc=0, scale=4)
lr= LogisticRegression(verbose=verbose,random_state=random_state,class_weight='balanced',solver='lbfgs',max_iter=500,n_jobs=-1)
parameters={'C':C_val}
clf = RandomizedSearchCV(lr,parameters,random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_ohe,y_train)


# In[60]:


best_c=best_model.best_params_['C']
best_c


# In[61]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_C')
results


# In[62]:


print_graph(results, 'param_C', 'mean_test_score', 'Hyperparameter - C', 'Test score')


# In[63]:


model = LogisticRegression(C=best_c,verbose=verbose,n_jobs=-1,random_state=random_state,class_weight='balanced',solver='lbfgs')
model.fit(train_ohe,y_train)


# In[64]:


predictions = model.predict_proba(test_ohe)[:,1]
save_submission(predictions, 'lr_ohe.csv')


# ![lr-ohe](images/lr-ohe.png)

# ## 2.4 Random Forest with one hot encoded features

# In[65]:


rfc = RandomForestClassifier(random_state=random_state,class_weight='balanced',n_jobs=-1)
clf = RandomizedSearchCV(rfc,get_rf_params(),random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_ohe,y_train)


# In[66]:


results = pd.DataFrame(best_model.cv_results_)
results.sort_values('mean_test_score',ascending=False,inplace=True)
param_keys=['param_'+str(each) for each in get_rf_params().keys()]
param_keys.append('mean_test_score')
results[param_keys].head(10)


# In[67]:


n_estimators=clf.best_params_['n_estimators']
max_features=clf.best_params_['max_features']
max_depth=clf.best_params_['max_depth']
min_samples_split=clf.best_params_['min_samples_split']
n_estimators,max_features,max_depth,min_samples_split


# In[68]:


model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,
                             min_samples_split=min_samples_split,
                             random_state=random_state,class_weight='balanced',n_jobs=-1)

model.fit(train_ohe,y_train)


# In[69]:


# features=train_ohe.columns
# importance=model.feature_importances_
# features=pd.DataFrame({'features':features,'value':importance})
# features=features.sort_values('value',ascending=False)
# sns.barplot('value','features',data=features);
# plt.title('Feature Importance');


# In[70]:


predictions = model.predict_proba(test_ohe)[:,1]
save_submission(predictions, 'rf_ohe.csv')


# ![rf-ohe](images/rf-ohe.png)

# ## 2.5 Xgboost with one hot encoded features

# In[71]:


xgb = XGBClassifier()
clf = RandomizedSearchCV(xgb,get_xgb_params(),random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model=clf.fit(train_ohe,y_train)


# In[72]:


results = pd.DataFrame(best_model.cv_results_)
results.sort_values('mean_test_score',ascending=False,inplace=True)
param_keys=['param_'+str(each) for each in get_xgb_params().keys()]
param_keys.append('mean_test_score')
results[param_keys].head(10)


# In[73]:


colsample_bytree = clf.best_params_['colsample_bytree']
learning_rate=clf.best_params_['learning_rate']
max_depth=clf.best_params_['max_depth']
min_child_weight=clf.best_params_['min_child_weight']
n_estimators=clf.best_params_['n_estimators']
subsample=clf.best_params_['subsample']
colsample_bytree,learning_rate,max_depth,min_child_weight,n_estimators,subsample


# In[74]:


model = XGBClassifier(colsample_bytree=colsample_bytree,learning_rate=learning_rate,max_depth=max_depth,
                     min_child_weight=min_child_weight,n_estimators=n_estimators,subsample=subsample,n_jobs=-1)

model.fit(train_ohe,y_train)


# In[75]:


# features=train_ohe.columns
# importance=model.feature_importances_
# features=pd.DataFrame({'features':features,'value':importance})
# features=features.sort_values('value',ascending=False)
# sns.barplot('value','features',data=features);
# plt.title('Feature Importance');


# In[76]:


predictions = model.predict_proba(test_ohe)[:,1]
save_submission(predictions, 'xgb_ohe.csv')


# ![xgb-ohe](images/xgb-ohe.png)

# ![kaggle-submission-ohe](images/kaggle-submission-ohe.png)

# In[77]:


from prettytable import PrettyTable

x = PrettyTable(['Model', 'Feature', 'Private Score', 'Public Score'])
x.add_row(['KNN','ohe', 0.81657, 0.81723])
x.add_row(['SVM', 'ohe', 0.87249, 0.87955])
x.add_row(['Logistic Regression', 'ohe', 0.87436, 0.88167])
x.add_row(['Random Forest', 'ohe', 0.84541, 0.84997])
x.add_row(['Xgboost', 'ohe', 0.84717, 0.85102])

print(x)


# # Observations:
# 
# 1. One hot encoding features performs better than other encoding technique
# 2. Linear models (Logistic Regression and SVM) performs better on higher dimension

# # 3 Build Model on frequency encoding feature

# ## 3.1 KNN with frequency encoding

# In[78]:


train_df_fc = pd.read_csv('data/train_df_fc.csv')
test_df_fc = pd.read_csv('data/test_df_fc.csv')


# In[79]:


train_df_fc.shape, test_df_fc.shape, y_train.shape


# In[80]:


parameters={'n_neighbors':np.arange(1,100, 5)}
clf = RandomizedSearchCV(KNeighborsClassifier(n_jobs=-1),parameters,random_state=random_state,cv=cv,verbose=verbose,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_df_fc,y_train)


# In[81]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_n_neighbors')
results


# In[82]:


print_graph(results, 'param_n_neighbors', 'mean_test_score', 'Hyperparameter - No. of neighbors', 'Test score')  


# In[83]:


best_c=best_model.best_params_['n_neighbors']
best_c


# In[84]:


model = KNeighborsClassifier(n_neighbors=best_c,n_jobs=-1)
model.fit(train_df_fc,y_train)


# In[85]:


predictions = model.predict_proba(test_df_fc)[:,1]
save_submission(predictions, "knn_fc.csv")


# ![knn-fc](images/knn-fc.png)

# ## 3.2 SVM with frequency encoding

# In[86]:


C_val = uniform(loc=0, scale=4)
model= LinearSVC(verbose=verbose,random_state=random_state,class_weight='balanced',max_iter=2000)
parameters={'C':C_val}
clf = RandomizedSearchCV(model,parameters,random_state=random_state,cv=cv,verbose=verbose,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_df_fc,y_train)


# In[87]:


best_c=best_model.best_params_['C']
best_c


# In[88]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_C')
results


# In[89]:


print_graph(results, 'param_C', 'mean_test_score', 'Hyperparameter - C', 'Test score')


# In[90]:


#https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
model = LinearSVC(C=best_c,verbose=verbose,random_state=random_state,class_weight='balanced',max_iter=2000)
model = CalibratedClassifierCV(model)
model.fit(train_df_fc,y_train)


# In[91]:


predictions = model.predict_proba(test_df_fc)[:,1]
save_submission(predictions, 'svm_fc.csv')


# ![svm-fc](images/svm-fc.png)

# ## 3.3 Logistic Regression with frequency encoding

# In[92]:


C_val = uniform(loc=0, scale=4)
lr= LogisticRegression(verbose=verbose,random_state=random_state,class_weight='balanced',solver='lbfgs',max_iter=500,n_jobs=-1)
parameters={'C':C_val}
clf = RandomizedSearchCV(lr,parameters,random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_df_fc,y_train)


# In[93]:


best_c=best_model.best_params_['C']
best_c


# In[94]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_C')
results


# In[95]:


print_graph(results, 'param_C', 'mean_test_score', 'Hyperparameter - C', 'Test score')


# In[96]:


model = LogisticRegression(C=best_c,verbose=verbose,n_jobs=-1,random_state=random_state,class_weight='balanced',solver='lbfgs')
model.fit(train_df_fc,y_train)


# In[97]:


predictions = model.predict_proba(test_df_fc)[:,1]
save_submission(predictions, 'lr_fc.csv')


# ![lr-fc](images/lr-fc.png)

# ## 3.4 Random Forest with frequency encoding

# In[98]:


rfc = RandomForestClassifier(random_state=random_state,class_weight='balanced',n_jobs=-1)
clf = RandomizedSearchCV(rfc,get_rf_params(),random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_df_fc,y_train)


# In[99]:


results = pd.DataFrame(best_model.cv_results_)
results.sort_values('mean_test_score',ascending=False,inplace=True)
param_keys=['param_'+str(each) for each in get_rf_params().keys()]
param_keys.append('mean_test_score')
results[param_keys].head(10)


# In[100]:


n_estimators=clf.best_params_['n_estimators']
max_features=clf.best_params_['max_features']
max_depth=clf.best_params_['max_depth']
min_samples_split=clf.best_params_['min_samples_split']
n_estimators,max_features,max_depth,min_samples_split


# In[101]:


model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,
                             min_samples_split=min_samples_split,
                             random_state=random_state,class_weight='balanced',n_jobs=-1)

model.fit(train_df_fc,y_train)


# In[103]:


features=train_df_fc.columns
importance=model.feature_importances_
features=pd.DataFrame({'features':features,'value':importance})
features=features.sort_values('value',ascending=False)
sns.barplot('value','features',data=features);
plt.title('Feature Importance');


# In[106]:


predictions = model.predict_proba(test_df_fc)[:,1]
save_submission(predictions, 'rf_fc.csv')


# ![rf-fc](images/rf-fc.png)

# ## 3.5 Xgboost with frequency encoding

# In[107]:


xgb = XGBClassifier()
clf = RandomizedSearchCV(xgb,get_xgb_params(),random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model=clf.fit(train_df_fc,y_train)


# In[108]:


results = pd.DataFrame(best_model.cv_results_)
results.sort_values('mean_test_score',ascending=False,inplace=True)
param_keys=['param_'+str(each) for each in get_xgb_params().keys()]
param_keys.append('mean_test_score')
results[param_keys].head(10)


# In[109]:


colsample_bytree = clf.best_params_['colsample_bytree']
learning_rate=clf.best_params_['learning_rate']
max_depth=clf.best_params_['max_depth']
min_child_weight=clf.best_params_['min_child_weight']
n_estimators=clf.best_params_['n_estimators']
subsample=clf.best_params_['subsample']
colsample_bytree,learning_rate,max_depth,min_child_weight,n_estimators,subsample


# In[110]:


model = XGBClassifier(colsample_bytree=colsample_bytree,learning_rate=learning_rate,max_depth=max_depth,
                     min_child_weight=min_child_weight,n_estimators=n_estimators,subsample=subsample,n_jobs=-1)

model.fit(train_df_fc,y_train)


# In[111]:


features=train_df_fc.columns
importance=model.feature_importances_
features=pd.DataFrame({'features':features,'value':importance})
features=features.sort_values('value',ascending=False)
sns.barplot('value','features',data=features);
plt.title('Feature Importance');


# In[112]:


predictions = model.predict_proba(test_df_fc)[:,1]
save_submission(predictions, 'xgb_fc.csv')


# ![xgb-fc](images/xgb-fc.png)

# ![kaggle-submission-fc](images/kaggle-submission-fc.png)

# In[113]:


from prettytable import PrettyTable

x = PrettyTable(['Model', 'Feature', 'Private Score', 'Public Score'])
x.add_row(['KNN','fc', 0.79715, 0.79125])
x.add_row(['SVM', 'fc', 0.60085, 0.59550])
x.add_row(['Logistic Regression', 'fc', 0.59896, 0.59778])
x.add_row(['Random Forest', 'fc', 0.87299, 0.87616])
x.add_row(['Xgboost', 'fc', 0.86987, 0.86944])

print(x)


# # Observations:
# 
# 1. Tree based models performs better for this feature than linear models
# 2. KNN is doing good for every feature

# # 4 Build Model using response encoding feature

# In[114]:


train_df_rc = pd.read_csv('data/train_df_rc.csv')
test_df_rc = pd.read_csv('data/test_df_rc.csv')


# In[115]:


train_df_rc.shape, test_df_rc.shape, y_train.shape


# ## 4.1 KNN with response encoding

# In[116]:


parameters={'n_neighbors':np.arange(1,100, 5)}
clf = RandomizedSearchCV(KNeighborsClassifier(n_jobs=-1),parameters,random_state=random_state,cv=cv,verbose=verbose,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_df_rc,y_train)


# In[117]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_n_neighbors')
results


# In[118]:


print_graph(results, 'param_n_neighbors', 'mean_test_score', 'Hyperparameter - No. of neighbors', 'Test score')  


# In[119]:


best_c=best_model.best_params_['n_neighbors']
best_c


# In[120]:


model = KNeighborsClassifier(n_neighbors=best_c,n_jobs=-1)
model.fit(train_df_rc,y_train)


# In[121]:


predictions = model.predict_proba(test_df_rc)[:,1]
save_submission(predictions, "knn_rc.csv")


# ![knn-rc](images/knn-rc.png)

# ## 4.2 SVM with response encoding

# In[122]:


C_val = uniform(loc=0, scale=4)
model= LinearSVC(verbose=verbose,random_state=random_state,class_weight='balanced',max_iter=2000)
parameters={'C':C_val}
clf = RandomizedSearchCV(model,parameters,random_state=random_state,cv=cv,verbose=verbose,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_df_rc,y_train)


# In[123]:


best_c=best_model.best_params_['C']
best_c


# In[124]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_C')
results


# In[125]:


print_graph(results, 'param_C', 'mean_test_score', 'Hyperparameter - C', 'Test score')


# In[126]:


#https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
model = LinearSVC(C=best_c,verbose=verbose,random_state=random_state,class_weight='balanced',max_iter=2000)
model = CalibratedClassifierCV(model)
model.fit(train_df_rc,y_train)


# In[127]:


predictions = model.predict_proba(test_df_rc)[:,1]
save_submission(predictions, 'svm_rc.csv')


# ![svm-rc](images/svm-rc.png)

# ## 4.3 Logistic Regression with response encoding

# In[128]:


C_val = uniform(loc=0, scale=4)
lr= LogisticRegression(verbose=verbose,random_state=random_state,class_weight='balanced',solver='lbfgs',max_iter=500,n_jobs=-1)
parameters={'C':C_val}
clf = RandomizedSearchCV(lr,parameters,random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_df_rc,y_train)


# In[129]:


best_c=best_model.best_params_['C']
best_c


# In[130]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_C')
results


# In[131]:


print_graph(results, 'param_C', 'mean_test_score', 'Hyperparameter - C', 'Test score')


# In[132]:


model = LogisticRegression(C=best_c,verbose=verbose,n_jobs=-1,random_state=random_state,class_weight='balanced',solver='lbfgs')
model.fit(train_df_rc,y_train)


# In[133]:


predictions = model.predict_proba(test_df_rc)[:,1]
save_submission(predictions, 'lr_rc.csv')


# ![lr-rc](images/lr-rc.png)

# ## 4.4 Random Forest with response encoding

# In[134]:


rfc = RandomForestClassifier(random_state=random_state,class_weight='balanced',n_jobs=-1)
clf = RandomizedSearchCV(rfc,get_rf_params(),random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_df_rc,y_train)


# In[135]:


results = pd.DataFrame(best_model.cv_results_)
results.sort_values('mean_test_score',ascending=False,inplace=True)
param_keys=['param_'+str(each) for each in get_rf_params().keys()]
param_keys.append('mean_test_score')
results[param_keys].head(10)


# In[136]:


n_estimators=clf.best_params_['n_estimators']
max_features=clf.best_params_['max_features']
max_depth=clf.best_params_['max_depth']
min_samples_split=clf.best_params_['min_samples_split']
n_estimators,max_features,max_depth,min_samples_split


# In[137]:


model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,
                             min_samples_split=min_samples_split,
                             random_state=random_state,class_weight='balanced',n_jobs=-1)

model.fit(train_df_rc,y_train)


# In[138]:


features=train_df_rc.columns
importance=model.feature_importances_
features=pd.DataFrame({'features':features,'value':importance})
features=features.sort_values('value',ascending=False)
sns.barplot('value','features',data=features);
plt.title('Feature Importance');


# In[139]:


predictions = model.predict_proba(test_df_rc)[:,1]
save_submission(predictions, 'rf_rc.csv')


# ![rf-rc](images/rf-rc.png)

# ## 4.5 Xgboost with response encoding

# In[140]:


xgb = XGBClassifier()
clf = RandomizedSearchCV(xgb,get_xgb_params(),random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model=clf.fit(train_df_rc,y_train)


# In[141]:


results = pd.DataFrame(best_model.cv_results_)
results.sort_values('mean_test_score',ascending=False,inplace=True)
param_keys=['param_'+str(each) for each in get_xgb_params().keys()]
param_keys.append('mean_test_score')
results[param_keys].head(10)


# In[142]:


colsample_bytree = clf.best_params_['colsample_bytree']
learning_rate=clf.best_params_['learning_rate']
max_depth=clf.best_params_['max_depth']
min_child_weight=clf.best_params_['min_child_weight']
n_estimators=clf.best_params_['n_estimators']
subsample=clf.best_params_['subsample']
colsample_bytree,learning_rate,max_depth,min_child_weight,n_estimators,subsample


# In[143]:


model = XGBClassifier(colsample_bytree=colsample_bytree,learning_rate=learning_rate,max_depth=max_depth,
                     min_child_weight=min_child_weight,n_estimators=n_estimators,subsample=subsample,n_jobs=-1)

model.fit(train_df_rc,y_train)


# In[144]:


features=train_df_rc.columns
importance=model.feature_importances_
features=pd.DataFrame({'features':features,'value':importance})
features=features.sort_values('value',ascending=False)
sns.barplot('value','features',data=features);
plt.title('Feature Importance');


# In[145]:


predictions = model.predict_proba(test_df_rc)[:,1]
save_submission(predictions, 'xgb_rc.csv')


# ![xgb-rc](images/xgb-rc.png)

# ![kaggle-submission-rc](images/kaggle-submission-rc.png)

# In[146]:


from prettytable import PrettyTable

x = PrettyTable(['Model', 'Feature', 'Private Score', 'Public Score'])
x.add_row(['KNN','rc', 0.84352, 0.85351])
x.add_row(['SVM', 'rc', 0.85160, 0.86031])
x.add_row(['Logistic Regression', 'rc', 0.85322, 0.86180])
x.add_row(['Random Forest', 'rc', 0.83136, 0.83892])
x.add_row(['Xgboost', 'rc', 0.84135, 0.84190])

print(x)


# # Observations:
# 
# 1. Every model performs good for this feature
# 2. Linear models performs better than Tree based models

# # 5 Build model on SVD feature

# In[147]:


train_svd = pd.read_csv('data/train_svd.csv')
test_svd = pd.read_csv('data/test_svd.csv')


# In[148]:


train_svd.shape, test_svd.shape, y_train.shape


# ## 5.1 KNN with SVD

# In[149]:


parameters={'n_neighbors':np.arange(1,100, 5)}
clf = RandomizedSearchCV(KNeighborsClassifier(n_jobs=-1),parameters,random_state=random_state,cv=cv,verbose=verbose,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_svd,y_train)


# In[150]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_n_neighbors')
results


# In[151]:


print_graph(results, 'param_n_neighbors', 'mean_test_score', 'Hyperparameter - No. of neighbors', 'Test score')  


# In[152]:


best_c=best_model.best_params_['n_neighbors']
best_c


# In[153]:


model = KNeighborsClassifier(n_neighbors=best_c,n_jobs=-1)
model.fit(train_svd,y_train)


# In[154]:


predictions = model.predict_proba(test_svd)[:,1]
save_submission(predictions, "knn_svd.csv")


# ![knn-svd](images/knn-svd.png)

# ## 5.2 SVM with SVD

# In[155]:


C_val = uniform(loc=0, scale=4)
model= LinearSVC(verbose=verbose,random_state=random_state,class_weight='balanced',max_iter=2000)
parameters={'C':C_val}
clf = RandomizedSearchCV(model,parameters,random_state=random_state,cv=cv,verbose=verbose,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_svd,y_train)


# In[156]:


best_c=best_model.best_params_['C']
best_c


# In[157]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_C')
results


# In[158]:


print_graph(results, 'param_C', 'mean_test_score', 'Hyperparameter - C', 'Test score')


# In[159]:


#https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
model = LinearSVC(C=best_c,verbose=verbose,random_state=random_state,class_weight='balanced',max_iter=2000)
model = CalibratedClassifierCV(model)
model.fit(train_svd,y_train)


# In[160]:


predictions = model.predict_proba(test_svd)[:,1]
save_submission(predictions, 'svm_svd.csv')


# ![svm-svd](images/svm-svd.png)

# ## 5.3 Logistic Regression with SVD

# In[161]:


C_val = uniform(loc=0, scale=4)
lr= LogisticRegression(verbose=verbose,random_state=random_state,class_weight='balanced',solver='lbfgs',max_iter=500,n_jobs=-1)
parameters={'C':C_val}
clf = RandomizedSearchCV(lr,parameters,random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_svd,y_train)


# In[162]:


best_c=best_model.best_params_['C']
best_c


# In[163]:


results = pd.DataFrame.from_dict(best_model.cv_results_)
results=results.sort_values('param_C')
results


# In[164]:


print_graph(results, 'param_C', 'mean_test_score', 'Hyperparameter - C', 'Test score')


# In[165]:


model = LogisticRegression(C=best_c,verbose=verbose,n_jobs=-1,random_state=random_state,class_weight='balanced',solver='lbfgs')
model.fit(train_svd,y_train)


# In[166]:


predictions = model.predict_proba(test_svd)[:,1]
save_submission(predictions, 'lr_svd.csv')


# ![lr-svd](images/lr-svd.png)

# ## 5.4 Random Forest with SVD

# In[167]:


rfc = RandomForestClassifier(random_state=random_state,class_weight='balanced',n_jobs=-1)
clf = RandomizedSearchCV(rfc,get_rf_params(),random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model = clf.fit(train_svd,y_train)


# In[168]:


results = pd.DataFrame(best_model.cv_results_)
results.sort_values('mean_test_score',ascending=False,inplace=True)
param_keys=['param_'+str(each) for each in get_rf_params().keys()]
param_keys.append('mean_test_score')
results[param_keys].head(10)


# In[169]:


n_estimators=clf.best_params_['n_estimators']
max_features=clf.best_params_['max_features']
max_depth=clf.best_params_['max_depth']
min_samples_split=clf.best_params_['min_samples_split']
n_estimators,max_features,max_depth,min_samples_split


# In[170]:


model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,
                             min_samples_split=min_samples_split,
                             random_state=random_state,class_weight='balanced',n_jobs=-1)

model.fit(train_svd,y_train)


# In[171]:


features=train_svd.columns
importance=model.feature_importances_
features=pd.DataFrame({'features':features,'value':importance})
features=features.sort_values('value',ascending=False)
sns.barplot('value','features',data=features);
plt.title('Feature Importance');


# In[172]:


predictions = model.predict_proba(test_svd)[:,1]
save_submission(predictions, 'rf_svd.csv')


# ![rf-svd](images/rf-svd.png)

# ## 5.5 Xgboost with SVD

# In[173]:


xgb = XGBClassifier()
clf = RandomizedSearchCV(xgb,get_xgb_params(),random_state=random_state,cv=cv,verbose=verbose,n_iter=100,scoring=scoring,n_jobs=-1)
best_model=clf.fit(train_svd,y_train)


# In[174]:


results = pd.DataFrame(best_model.cv_results_)
results.sort_values('mean_test_score',ascending=False,inplace=True)
param_keys=['param_'+str(each) for each in get_xgb_params().keys()]
param_keys.append('mean_test_score')
results[param_keys].head(10)


# In[175]:


colsample_bytree = clf.best_params_['colsample_bytree']
learning_rate=clf.best_params_['learning_rate']
max_depth=clf.best_params_['max_depth']
min_child_weight=clf.best_params_['min_child_weight']
n_estimators=clf.best_params_['n_estimators']
subsample=clf.best_params_['subsample']
colsample_bytree,learning_rate,max_depth,min_child_weight,n_estimators,subsample


# In[176]:


model = XGBClassifier(colsample_bytree=colsample_bytree,learning_rate=learning_rate,max_depth=max_depth,
                     min_child_weight=min_child_weight,n_estimators=n_estimators,subsample=subsample,n_jobs=-1)

model.fit(train_svd,y_train)


# In[177]:


features=train_svd.columns
importance=model.feature_importances_
features=pd.DataFrame({'features':features,'value':importance})
features=features.sort_values('value',ascending=False)
sns.barplot('value','features',data=features);
plt.title('Feature Importance');


# In[178]:


predictions = model.predict_proba(test_svd)[:,1]
save_submission(predictions, 'xgb_svd.csv')


# ![xgb-svd](images/xgb-svd.png)

# ![kaggle-submission-svd](images/kaggle-submission-svd.png)

# In[179]:


from prettytable import PrettyTable

x = PrettyTable(['Model', 'Feature', 'Private Score', 'Public Score'])
x.add_row(['KNN','svd', 0.79245, 0.78572])
x.add_row(['SVM', 'svd', 0.63648, 0.63806])
x.add_row(['Logistic Regression', 'svd', 0.63255, 0.63314])
x.add_row(['Random Forest', 'svd', 0.87119, 0.86924])
x.add_row(['Xgboost', 'svd', 0.86909, 0.86664])

print(x)


# # Observations:
# 
# 1. Tree based models works better than linear model
# 2. KNN is performing overall good

# # We have to improve our model to reach into 5-10% on kaggle

# In[180]:


# https://www.kaggle.com/mitribunskiy/tutorial-catboost-overview


# In[181]:


# https://www.kaggle.com/prashant111/catboost-classifier-tutorial


# # https://catboost.ai/
# 
# ## CatBoost is a high-performance open source library for gradient boosting on decision trees
# 
# 
# ### About
# CatBoost is an algorithm for gradient boosting on decision trees. It is developed by Yandex researchers and engineers, and is used for search, recommendation systems, personal assistant, self-driving cars, weather prediction and many other tasks at Yandex and in other companies, including CERN, Cloudflare, Careem taxi. It is in open-source and can be used by anyone.
# 
# 
# ### Features
# 
# 1. Reduce time spent on parameter tuning, because CatBoost provides great results with default parameters
# 2. Improve your training results with CatBoost that allows you to use non-numeric factors, instead of having to pre-process your data or spend time and effort turning it to numbers. 
# 3. Reduce overfitting when constructing your models with a novel gradient-boosting scheme.
# 4. Apply your trained model quickly and efficiently even to latency-critical tasks using CatBoost's model applier

# In[182]:


params = {
            'loss_function':'Logloss',
            'eval_metric':'AUC',
            'cat_features':list(range(train_data.shape[1])),
            'verbose':100,
            'random_seed':random_state
        }


# In[183]:


clf= CatBoostClassifier(**params)
clf.fit(train_data,y_train)


# In[184]:


predictions = clf.predict_proba(test_data)[:,1]


# In[185]:


save_submission(predictions, 'catboost.csv')


# ![catboost](images/catboost.png)

# ## Catboost perform better than all our previous models and it's AUC score is much better than previous models so I am selecting this for predicting future data

# In[186]:


# Save model on disk
pickle.dump(clf, open('models/catboost_model.pkl', 'wb'))


# In[ ]:




