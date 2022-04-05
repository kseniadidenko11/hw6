#!/usr/bin/env python
# coding: utf-8

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1McNxpNrSwfqu1w-QtlOmPSmfULvkkMQV" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h4 style="color:#303030"> Python for Data Science, Homework, template: </h4> </center>
# <center> <h1 style="color:#303030">Classification of Pirate Attacks</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center style="color:#303030"><h4>Propulsion Academy, 2021</h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm">
#         <p><strong>Goal:</strong> Practice binary classification on ship and Pirate attack data</p>
#         <strong> Sections:</strong>
#         <a id="P0" name="P0"></a>
#         <ol>
#             <li> <a style="color:#303030" href="#SU">Set Up </a> </li>
#             <li> <a style="color:#303030" href="#P1">Exploratory Data Analysis</a></li>
#             <li> <a style="color:#303030" href="#P2">Modeling</a></li>
#         </ol>
#         <strong>Topics Trained:</strong> Binary Classification.
#     </div>
# </div>
# 
# 
# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/" title="momentum"> SIT Introduction to Data Science</a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/weeks/week2/day1/index.html" title="momentum">Week 2 Day 1, Applied Machine Learning</a>|
#         <a style="color:#00BAE5" href="https://colab.research.google.com/drive/1DK68oHRR2-5IiZ2SG7OTS2cCFSe-RpeE?usp=sharing" title="momentum"> Assignment, Classification of the success of pirate attacks</a>
# </strong></nav>

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# In[ ]:


#get_ipython().system('sudo apt-get install build-essential swig')
#get_ipython().system('curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
#get_ipython().system('pip install auto-sklearn')
#get_ipython().system('pip install pipelineprofiler # visualize the pipelines created by auto-sklearn')
#get_ipython().system('pip install shap')
#get_ipython().system('pip install plotly')
#get_ipython().system('pip install scipy')


# Please note you **need to restart the run after the installation** for it to take effect!

# Packages import

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import set_config
from sklearn.pipeline import Pipeline
from pandas_profiling import ProfileReport
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

from joblib import dump

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import autosklearn.regression
import PipelineProfiler
import shap
import math

import matplotlib.pyplot as plt


# Configure logging.

# In[ ]:


import datetime

import logging

timesstr = str(datetime.datetime.now()).replace(' ', '_')
log_config = {
    "version":1,
    "root":{
        "handlers" : ["console"],
        "level": "DEBUG"
    },
    "handlers":{
        "console":{
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG"
        }
    },
    "formatters":{
        "std_out": {
            "format": "%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : (Process Details : (%(process)d, %(processName)s), Thread Details : (%(thread)d, %(threadName)s))\nLog : %(message)s",
            "datefmt":"%d-%m-%Y %I:%M:%S"
        }
    },
}

logging.config.dictConfig(log_config)


# Mount google drive and apply settings

# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)
data_path = "/content/drive/MyDrive/Introduction2DataScience/exercises/hw6/data/raw/"
model_path = "/content/drive/MyDrive/Introduction2DataScience/exercises/hw6/models"
set_config(display='diagram')


# Please Download the data from [this source](https://drive.google.com/file/d/1uMM8qdQSiHHjIiYPd45EPzXH7sqIiQ9t/view?usp=sharing), and upload it on your introduction2DS/data google drive folder.

# <a id='P1' name="P1"></a>
# ## [Exploratory Data Analysis](#P0)
# 

# ### Understand the Context

# **What type of problem are we trying to solve?**
# 
# With this data set, we want to build a classifier that would predict if a given Pirate attack will be successful.
# 
# **_This is a binary classification problem_**

# **How was the data collected?/ Is there documentation on the Data?**
# 
# You, the High Commander of the 7 Seas, are tasked to predict whether a Pirate attack is likely to be successful. It just so happens that you are also a talented Data scientist! You are provided with a dataset on vessels that have been attacked by pirates. Your dataset describes the attributes of a successful (1) and an unsuccessful (0) attack.
# The columns are described as follow:
# 
#   -   DATETIME (LOCAL):  The local date and time of when the attack occurred 
#   -   DATETIME (UTC): The date and time in UTC format of when the attack occurred
#   -   DATE (LT): The date in local time of when the attack occurred
#   -   DATE (UTC): The date in UTC format of when the attack occurred
#   -   REGION: Region in which the attack took place
#   -   COUNTRY: Country of the occurrence 
#   -   INCIDENT TYPE: Type of the attack
#   -   MAERSK: Indicates whether the attacked vessel belonged to the MAERSK shipping company
#   -   VESSEL ACTIVITY LOCATION: Location of the vessel during attack
#   -   LAT: Latitude coordinate of attack
#   -   LONG: Longitude coordinate of attack
#   -   TIMEZONE: Time zone in which the attack occurred 
#   -   TIME OF DAY: General time of day when the attack occurred
#   -   ATTACKS: Indicate whether the attack occurred (1: attack occurred, 0: attack did not occur)
#   -   ATTACK SUCCESS: Indicate whether the attack was successful or not.
# 
# 

# **Do we have assumption about the data?**

# - longitude and lattitude should be numbers corresponding to the longitude and latitude of water part of the world
# - we expect to see more frequent attack in the following areas: the Gulf of Guinea and the Straits of Malacca and Singapore.

# **Can we foresee any challenge related to this data set?**

# We may have redundant columns - for example the DATETIME (LOCAL) can be calculated from datetime utc knowing the region of attack. Timezone can also be calculated from the region.

# ### Data Structure and types

# **Load the csv file as a DataFrame using Pandas**

# In[ ]:


df = pd.read_csv(f'{data_path}pirate_data.csv', delimiter=",", index_col=0)


# **Perform the necessary type transformations**

# In[ ]:


df['DATETIME (LOCAL)'] = pd.to_datetime(df['DATETIME (LOCAL)'])
df['DATETIME (UTC)'] = pd.to_datetime(df['DATETIME (UTC)'])
df['DATE (LT)'] = pd.to_datetime(df['DATE (LT)'])
df['DATE (UTC)'] = pd.to_datetime(df['DATE (UTC)'])


# In[ ]:


for column in df.select_dtypes(include=['object']):
    print(f'Column {column} has {len(df[column].unique())} categories: {df[column].unique()}\n')
    df[column] = df[column].astype("category")
df.dtypes


# **Perform test/train split here**
# 
# !!! Please think about it!!! How should the data be splitted?

# We want to keep the same proportion, so we will use 'stratify' parameter for splitting.

# In[ ]:


#y = df['ATTACK SUCCESS']
#X = df.drop(['ATTACK SUCCESS'], axis=1)
X, y = df.iloc[:,:-1], df.iloc[:,-1]
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# from now on, we will use the train dataframe. let's save the train and test.

# ### Missing Values and Duplicates

# In[ ]:


X_train['VESSEL ACTIVITY LOCATION'] = X_train['VESSEL ACTIVITY LOCATION'].cat.add_categories('UNKNOWN')
X_train['VESSEL ACTIVITY LOCATION'].fillna('UNKNOWN', inplace =True) 
X_train['VESSEL TYPE'].fillna('UNKNOWN', inplace =True) 


# ### Relationship between features (correlations)
# 
# **What are the relationships between features (make a pairplot)? Are they linear?**

# In[ ]:


fig = px.parallel_categories(X_train[['REGION', 'COUNTRY', 'INCIDENT TYPE', 
                                      'VESSEL TYPE', 'VESSEL ACTIVITY LOCATION', 
                                      'TIME OF DAY']], color=y_train)
fig.show()


# ### Feature Creation and Combination
# 
# 

# - **What kind of Scaling should we use/try?**
# - **Should we transform some features?**
# - **Should we drop some features?**
# - **Should we combine features?**

# In[ ]:


X_train_copy = X_train.copy()
X_train_copy = X_train_copy.drop(['INCIDENT TYPE', "ATTACKS", "DATETIME (LOCAL)", "DATETIME (UTC)", "DATE (LT)", "TIMEZONE"], axis=1)

X_test_copy = X_test.copy()
X_test_copy = X_test.drop(['INCIDENT TYPE', "ATTACKS", "DATETIME (LOCAL)", "DATETIME (UTC)", "DATE (LT)", "TIMEZONE"], axis=1)
categorical_columns = ['REGION', 'COUNTRY', 'VESSEL TYPE', 
                       'VESSEL ACTIVITY LOCATION', 'TIME OF DAY']
for column in categorical_columns:
    tempdf = pd.get_dummies(X_train_copy[column], prefix=column)
    X_train_copy = pd.merge(left=X_train_copy, right=tempdf, left_index=True, right_index=True,)
    X_train_copy = X_train_copy.drop(columns=column)
    tempdf = pd.get_dummies(X_test_copy[column], prefix=column)
    X_test_copy = pd.merge(left=X_test_copy, right=tempdf, left_index=True, right_index=True,)
    X_test_copy = X_test_copy.drop(columns=column)

X_train_copy["DATE (UTC)"] = X_train_copy["DATE (UTC)"].dt.dayofyear
X_test_copy["DATE (UTC)"] = X_test_copy["DATE (UTC)"].dt.dayofyear

X_train_copy


# We can delete the timezone column as latitude and longitude cover that information. We also need to delete the incident type column as it leaks data. Dates can be converted to day-of-the-year. Finally, we need to covert all the categorical columns into 1-hot encoded columns so they can be processed by the model.

# ### Conclusion: Experimental setup and  Possible Feature Transformations

# Let's wrap up on the exploratory data analysis and conclude. We should now be able to answer the following questions:
# 
# - **What would be our baseline for the analysis?**
# - **What kind of modelling setup should we use/try?**
# - **What kind of Scaling should we use/try?**
# - **If outliers, what kind of treatment should we apply?**
# - **Should we transform some features?**
# - **Should we drop some features?**
# - **Should we combine features?**

# The baseline for our analysis would be about 64%, as this would be our result if we always guess that attacks are successful. Accordingly, our model must do better than that. For modeling we will first try logistic regression, and then also try autoML. We can just use the standard scaler, as we don't have to many numeric variables and is should work fine for the longitude and latitude. There are no outliers so we can skip that, but we do need to transform the data. Specifically, we need to convert the categorical columns with a one-hot encoder so that they can be used by the model. We will also convert the date column into a day-of-the-year column as that is the truely relevant information when it comes to the date. We will drop a number of the features. Some, like the timezones, are because they are duplicates. Others, like the Incident Type have to be dropped because they leak data which would make the model useless. We don't need to combine any data.

# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# ### Pipeline Definition

# In[ ]:


from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


# In[ ]:


X_train = X_train.copy()
X_train = X_train.drop(['INCIDENT TYPE', "ATTACKS", "DATETIME (LOCAL)", "DATETIME (UTC)", "DATE (LT)", "TIMEZONE"], axis=1)
categorical_columns = ['REGION', 'COUNTRY', 'VESSEL TYPE', 
                       'VESSEL ACTIVITY LOCATION', 'TIME OF DAY']


# In[ ]:


ohe_variables = ['REGION', 'COUNTRY', 'VESSEL TYPE', 'VESSEL ACTIVITY LOCATION', 'TIME OF DAY']

oe_variables = []
#categories=['Month-to-month', 'One year', 'Two year']

num_variables = ["LAT", "LONG"]


# In[ ]:


numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                      ('scaler', StandardScaler())])


# In[ ]:


ohe_transformer = OneHotEncoder(handle_unknown='ignore')


# In[ ]:


oe_transformer = OrdinalEncoder(categories=[['1', '0'],])


# In[ ]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_variables),
        ('ohe', ohe_transformer, ohe_variables),
        ('oe', oe_transformer, oe_variables)])


# In[ ]:


from sklearn.pipeline import Pipeline
classification_model = Pipeline(steps=[('preprocessor', preprocessor), 
                                       ('classifier', LogisticRegression())])


# In[ ]:


classification_model


# In[ ]:


cross_val_score(classification_model, X_train, y_train)


# ### Model Training

# In[ ]:


y_train_encoded = y_train.astype(float)


# In[ ]:


total_time = 300
per_run_time_limit = 30


# In[ ]:


import autosklearn.classification
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=total_time,
    per_run_time_limit=per_run_time_limit,
)
automl.fit(X_train_copy, y_train_encoded)


# In[ ]:


logging.info(f'Ran autosklearn classifier for a total time of {total_time} seconds, with a maximum of {per_run_time_limit} seconds per model run')


# In[ ]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[ ]:


logging.info(f'Saved regressor model at {model_path}model{timesstr}.pkl ')


# In[ ]:


logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# In[ ]:


#profiler_data= PipelineProfiler.import_autosklearn(automl)
#PipelineProfiler.plot_pipeline_matrix(profiler_data)


# ### Model Evaluation

# In[ ]:


for column in X_train_copy.columns:
  if not column in X_test_copy.columns:
    loc = X_train_copy.columns.get_loc(column)
    X_test_copy.insert(loc, column, 0) 
    print(f'Column {column} added')


# In[ ]:


for column in X_test_copy.columns:
  if not column in X_train_copy.columns:
    print(f'Column {column} added')


# In[ ]:





# In[ ]:





# In[ ]:


y_pred = automl.predict(X_test_copy)


# In[ ]:


logging.info(f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test_copy, y_test)}")


# In[ ]:


print(f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test_copy, y_test)}")


# we can also plot the y_test vs y_pred scatter:

# In[ ]:



actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
actual_vs_predicted
#df = pd.DataFrame(np.concatenate((X_test_copy, y_test.to_numpy().reshape(-1,1), y_pred.reshape(-1,1)),  axis=1))


# In[ ]:


df = pd.DataFrame(np.concatenate((X_test_copy, y_test.to_numpy().reshape(-1,1), y_pred.reshape(-1,1)),  axis=1))


# In[ ]:


cols = list(X_test_copy)
cols.append('True Target')
cols.append('Predicted Target')


# In[ ]:


df.columns = cols


# In[ ]:


fig = px.scatter(df, x='Predicted Target', y='True Target')
fig.write_html(f"{model_path}residualfig_{timesstr}.html")


# In[ ]:


logging.info(f"Figure of residuals saved as {model_path}residualfig_{timesstr}.html")


# ### Model Explainability

# In[ ]:


explainer = shap.KernelExplainer(model = automl.predict, data = X_test_copy.iloc[:50, :], link = "identity")


# In[ ]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test_copy.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test_copy.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = X_test_copy.iloc[X_idx:X_idx+1,:], 
                show=False,
                matplotlib=True
                )
plt.savefig(f"{model_path}shap_example_{timesstr}.png")
logging.info(f"Shapley example saved as {model_path}shap_example_{timesstr}.png")


# In[ ]:


shap_values = explainer.shap_values(X = X_test_copy.iloc[0:50,:], nsamples = 100)


# In[ ]:


# print the JS visualization code to the notebook
shap.initjs()
fig = shap.summary_plot(shap_values = shap_values,
                  features = X_test_copy.iloc[0:50,:],
                  show=False)
plt.savefig(f"{model_path}shap_summary_{timesstr}.png")
logging.info(f"Shapley summary saved as {model_path}shap_summary_{timesstr}.png")


# _Your Comments here_

# --------------
# # End of This Notebook
