#!/usr/bin/env python
# coding: utf-8

# # CKD Prediction
# ---
# ![image%20of%20Chronic%20Kidney%20Disease.jpeg](attachment:image%20of%20Chronic%20Kidney%20Disease.jpeg)
# 
# In this project I will show you how to create predictive machine learning model using python program to predict and classify patience as having chronic kidney disease (ckd) or not. 
# 
# Chronic kidney disease, also called chronic kidney failure, describes the gradual loss of kidney function. Your kidneys filter wastes and excess fluids from your blood, which are then excreted in your urine. When chronic kidney disease reaches an advanced stage, dangerous levels of fluid, electrolytes and wastes can build up in your body. -Mayo Clinic
# 
# In the early stages of chronic kidney disease, you may have few signs or symptoms. Chronic kidney disease may not become apparent until your kidney function is significantly impaired. -Mayo Clinic
# 
# Treatment for chronic kidney disease focuses on slowing the progression of the kidney damage, usually by controlling the underlying cause. Chronic kidney disease can progress to end-stage kidney failure, which is fatal without artificial filtering (dialysis) or a kidney transplant. -Mayo Clinic
# 
# ### Table of Contents
# 1. Data Pre Processing
# 2. EDA
# 3. Feature Encoding
# 4. Model Building
#     * Knn
#     * Decision Tree Classifier
#     * Random Forest Classifier
#     * Ada Boost Classifier
#     * Gradient Boosting Classifier
#     * Stochastic Gradient Boosting (SGB)
#     * XgBoost
#     * Cat Boost Classifier
#     * Extra Trees Classifier
#     * LGBM Classifier
# 5. Models Comparison
# 
# ### Data Set Information
# acknowledgements: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease
# * Age(numerical) - age in years
# * Blood Pressure(numerical) - bp in mm/Hg
# * Specific Gravity(nominal) - sg - (1.005,1.010,1.015,1.020,1.025)
# * Albumin(nominal) - al - (0,1,2,3,4,5)
# * Sugar(nominal) - su - (0,1,2,3,4,5)
# * Red Blood Cells(nominal) - rbc - (normal,abnormal)
# * Pus Cell (nominal) - pc - (normal,abnormal)
# * Pus Cell clumps(nominal) - pcc - (present,notpresent)
# * Bacteria(nominal) - ba - (present,notpresent)
# * Blood Glucose Random(numerical) - bgr in mgs/dl
# * Blood Urea(numerical) -bu in mgs/dl
# * Serum Creatinine(numerical) - sc in mgs/dl
# * Sodium(numerical) - sod in mEq/L
# * Potassium(numerical) - pot in mEq/L
# * Hemoglobin(numerical) - hemo in gms
# * Packed Cell Volume(numerical)
# * White Blood Cell Count(numerical) - wc in cells/cumm
# * Red Blood Cell Count(numerical) - rc in millions/cmm
# * Hypertension(nominal) - htn - (yes,no)
# * Diabetes Mellitus(nominal) - dm - (yes,no)
# * Coronary Artery Disease(nominal) - cad - (yes,no)
# * Appetite(nominal) - appet - (good,poor)
# * Pedal Edema(nominal) - pe - (yes,no)
# * Anemia(nominal) - ane - (yes,no)
# * Class (nominal)- class - (ckd,notckd)

# ## Data Pre Processing

# In[1]:


# necessary imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 26)


# In[2]:


# import dataset 
data_file = pd.read_csv("kidney_disease.csv")
# view dataset
data_file


# In[3]:


# get a concise summary of the dataset
data_file.info()


# In[4]:


# check the null value for each column 
data_file.isnull().sum()


# In[5]:


# Percentage of missing values
((data_file.isnull().sum()/data_file.shape[0])*100).sort_values(ascending=False)


# In[6]:


# check if the dataset holds duplicate values
data_file.duplicated().any()


# From the above we can learn that some columns of th edataset have high percentage of missing values and the dataset has no duplicated row of data. We would drop some columns that they won't help us to find deep insights from the dataset.

# In[7]:


#drop id column because it is a unique identifier for each row and won't help us to find any insights from the data
data_file.drop(["id"],axis=1,inplace=True) 


# In[ ]:





# In[8]:


# rename column names to make it more user-friendly

data_file.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']


# In[9]:


data_file


# In[10]:


# get a concise summary of the dataset
data_file.info()


# In[11]:


# change some variables to numerical dtype, helping ther model to better learn the dataset
data_file['packed_cell_volume'] = pd.to_numeric(data_file['packed_cell_volume'], errors='coerce')
data_file['white_blood_cell_count'] = pd.to_numeric(data_file['white_blood_cell_count'], errors='coerce')
data_file['red_blood_cell_count'] = pd.to_numeric(data_file['red_blood_cell_count'], errors='coerce')


# In[12]:


data_file.info()


# In[13]:


# extract the categorical columns
cat_cols = [col for col in data_file.columns if data_file[col].dtype == 'object']
cat_cols


# In[14]:


# extract the categorical columns
num_cols = [col for col in data_file.columns if data_file[col].dtype != 'object']
num_cols


# In[15]:


# looking at unique values in categorical columns
for col in cat_cols:
    print(f"{col}: {data_file[col].unique()} \n")


# In[16]:


# some columns have unclear lables, so we need to clearify them or remove them 
data_file['diabetes_mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)

data_file['coronary_artery_disease'] = data_file['coronary_artery_disease'].replace(to_replace = '\tno', value='no')

data_file['class'] = data_file['class'].replace(to_replace = {'ckd\t': 'ckd', 'notckd': 'not ckd'})


# In[17]:


# change "class" column into boolean value
data_file['class'] = data_file['class'].map({'ckd': 0, 'not ckd': 1})
data_file['class'] = pd.to_numeric(data_file['class'], errors='coerce')


# In[18]:


# looking at unique values in categorical columns
for col in cat_cols:
    print(f"{col}: {data_file[col].unique()} \n")


# ## Exploratory Data Analysis (EDA)

# In[19]:


# checking numerical features distribution
plt.figure(figsize = (20, 15))
plotnumber = 1

for column in num_cols:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.distplot(data_file[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# ### Skewness is present in some of the columns.
#         Observations:
#         1.age looks a bit left skewed
#         2.Blood gluscose random is right skewed
#         3.Blood Urea is also a bit right skewed
#         4.Rest of the features are lightly skewed

# In[20]:


# looking at categorical columns

plt.figure(figsize = (20, 15))
plotnumber = 1

for column in cat_cols:
    if plotnumber <= 11:
        ax = plt.subplot(3, 4, plotnumber)
        sns.countplot(data_file[column], palette = 'rocket')
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[21]:


# create heatmap graph to learn the relationships between two variables

plt.figure(figsize = (15, 8))

sns.heatmap(data_file.corr(), annot = True, linewidths = 2, linecolor = 'lightgrey')
plt.show()


# Positive Correlation:
# * Specific gravity -> Red blood cell count, Packed cell volume and Hemoglobin 
# * Sugar -> Blood glucose random
# * Blood Urea -> Serum creatinine
# * Hemoglobin -> Red Blood cell count <- packed cell volume
# 
# 
# Negative Correlation:
# * Albumin, Blood urea -> Red blood cell count, packed cell volume, Hemoglobin
# * Serum creatinine -> Sodium

# In[22]:


# defining functions to create plot

def violin(col):
    fig = px.violin(data_file, y=col, x="class", color="class", box=True, template = 'plotly_dark')
    return fig.show()

def kde(col):
    grid = sns.FacetGrid(data_file, hue="class", height = 6, aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
    
def scatter(col1, col2):
    fig = px.scatter(data_file, x=col1, y=col2, color="class", template = 'plotly_dark')
    return fig.show()


# In[23]:


violin('red_blood_cell_count')


# In[24]:


kde('red_blood_cell_count')


# In[25]:


violin('white_blood_cell_count')


# In[26]:


kde('white_blood_cell_count')


# In[27]:


violin('packed_cell_volume')


# In[28]:


kde('packed_cell_volume')


# In[29]:


violin('haemoglobin')


# In[30]:


kde('haemoglobin')


# In[31]:


violin('albumin')


# In[32]:


kde('albumin')


# In[33]:


violin('blood_glucose_random')


# In[34]:


kde('blood_glucose_random')


# In[35]:


kde('sodium')


# In[36]:


violin('blood_urea')


# In[37]:


kde('blood_urea')


# In[38]:


violin('specific_gravity')


# In[39]:


kde('specific_gravity')


# In[40]:


scatter('haemoglobin', 'packed_cell_volume')


# In[41]:


scatter('red_blood_cell_count', 'packed_cell_volume')


# In[42]:


scatter('red_blood_cell_count', 'albumin')


# In[43]:


scatter('sugar', 'blood_glucose_random')


# In[44]:


scatter('packed_cell_volume','blood_urea')


# In[45]:


px.bar(data_file, x="specific_gravity", y="packed_cell_volume", color='class', barmode='group', template = 'plotly_dark', height = 400)


# In[46]:


px.bar(data_file, x="specific_gravity", y="albumin", color='class', barmode='group', template = 'plotly_dark', height = 400)


# In[47]:


px.bar(data_file, x="blood_pressure", y="packed_cell_volume", color='class', barmode='group', template = 'plotly_dark', height = 400)


# In[48]:


px.bar(data_file, x="blood_pressure", y="haemoglobin", color='class', barmode='group', template = 'plotly_dark', height = 400)


# In[ ]:





# ## Feature Encoding

# In[49]:


# checking for null values
data_file.isna().sum().sort_values(ascending = False)


# In[50]:


# filling null values, we will use two methods, random sampling for higher null values and 
# mean/mode sampling for lower null values

def random_value_imputation(feature):
    random_sample = data_file[feature].dropna().sample(data_file[feature].isna().sum())
    random_sample.index = data_file[data_file[feature].isnull()].index
    data_file.loc[data_file[feature].isnull(), feature] = random_sample
    
def impute_mode(feature):
    mode = data_file[feature].mode()[0]
    data_file[feature] = data_file[feature].fillna(mode)


# In[51]:


# filling num_cols null values using random sampling method
for col in num_cols:
    random_value_imputation(col)


# In[52]:


data_file[num_cols].isnull().sum()


# In[53]:


data_file


# In[54]:


for col in cat_cols:
    print(f"{col} has {data_file[col].nunique()} categories\n")


# In[55]:


# apply label encoder to change the categorical variabel
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_cols:
    data_file[col] = le.fit_transform(data_file[col])


# In[56]:


data_file


# ## Model Building

# In[57]:


ind_col = [col for col in data_file.columns if col != 'class']
dep_col = 'class'

X = data_file[ind_col]
y = data_file[dep_col]


# In[58]:


# splitting data intp training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# ### KNN

# In[59]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of knn

knn_acc = accuracy_score(y_test, knn.predict(X_test))

print(f"Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}")
print(f"Test Accuracy of KNN is {knn_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, knn.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, knn.predict(X_test))}")


# In[ ]:





# ### Decision Tree Classifier

# In[60]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of decision tree

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")


# ### Random Forest Classifier

# In[61]:


from sklearn.ensemble import RandomForestClassifier

rd_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 11, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)
rd_clf.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of random forest
rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))

print(f"Training Accuracy of Random Forest Classifier is {accuracy_score(y_train, rd_clf.predict(X_train))}")
print(f"Test Accuracy of Random Forest Classifier is {rd_clf_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, rd_clf.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, rd_clf.predict(X_test))}")


# In[ ]:





# ### Ada Boost Classifier

# In[62]:


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(base_estimator = dtc)
ada.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of ada boost

ada_acc = accuracy_score(y_test, ada.predict(X_test))

print(f"Training Accuracy of Ada Boost Classifier is {accuracy_score(y_train, ada.predict(X_train))}")
print(f"Test Accuracy of Ada Boost Classifier is {ada_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, ada.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, ada.predict(X_test))}")


# In[ ]:





# ### Gradient Boosting Classifier

# In[63]:


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of gradient boosting classifier

gb_acc = accuracy_score(y_test, gb.predict(X_test))

print(f"Training Accuracy of Gradient Boosting Classifier is {accuracy_score(y_train, gb.predict(X_train))}")
print(f"Test Accuracy of Gradient Boosting Classifier is {gb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, gb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, gb.predict(X_test))}")


# In[ ]:





# ### Stochastic Gradient Boosting (SGB)

# In[64]:


sgb = GradientBoostingClassifier(max_depth = 4, subsample = 0.90, max_features = 0.75, n_estimators = 200)
sgb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of stochastic gradient boosting classifier

sgb_acc = accuracy_score(y_test, sgb.predict(X_test))

print(f"Training Accuracy of Stochastic Gradient Boosting is {accuracy_score(y_train, sgb.predict(X_train))}")
print(f"Test Accuracy of Stochastic Gradient Boosting is {sgb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, sgb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, sgb.predict(X_test))}")


# In[ ]:





# ### XgBoost

# In[65]:


from xgboost import XGBClassifier

xgb = XGBClassifier(objective = 'binary:logistic', learning_rate = 0.5, max_depth = 5, n_estimators = 150)
xgb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of xgboost

xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

print(f"Training Accuracy of XgBoost is {accuracy_score(y_train, xgb.predict(X_train))}")
print(f"Test Accuracy of XgBoost is {xgb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, xgb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, xgb.predict(X_test))}")


# In[ ]:





# ### Cat Boost Classifier

# In[66]:


from catboost import CatBoostClassifier

cat = CatBoostClassifier(iterations=10)
cat.fit(X_train, y_train)


# In[67]:


# accuracy score, confusion matrix and classification report of cat boost

cat_acc = accuracy_score(y_test, cat.predict(X_test))

print(f"Training Accuracy of Cat Boost Classifier is {accuracy_score(y_train, cat.predict(X_train))}")
print(f"Test Accuracy of Cat Boost Classifier is {cat_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, cat.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, cat.predict(X_test))}")


# In[ ]:





# ### Extra Trees Classifier

# In[68]:


from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of extra trees classifier

etc_acc = accuracy_score(y_test, etc.predict(X_test))

print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(y_train, etc.predict(X_train))}")
print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, etc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, etc.predict(X_test))}")


# In[ ]:





# ## Models Comparison

# In[69]:


models = pd.DataFrame({
    'Model' : [ 'KNN', 'Decision Tree Classifier', 'Random Forest Classifier','Ada Boost Classifier',
             'Gradient Boosting Classifier', 'Stochastic Gradient Boosting', 'XgBoost', 'Cat Boost', 'Extra Trees Classifier'],
    'Score' : [knn_acc, dtc_acc, rd_clf_acc, ada_acc, gb_acc, sgb_acc, xgb_acc, cat_acc, etc_acc]
})


models.sort_values(by = 'Score', ascending = False)


# In[70]:


px.bar(data_frame = models, x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', 
       title = 'Models Comparison')

