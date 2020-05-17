#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt 


# In[2]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.head()


# In[3]:


train_df.shape


# In[4]:


train_df.head()


# #### 1. Decision/Random - MSSubClass, MSZoning, Street, Alley, LotFrontage, etc
# #### 2. Linear - LotArea,   

# In[5]:


n = list(train_df.isna().sum())
n1 = pd.DataFrame(n)
n1.columns = ['na']
col_d = pd.DataFrame(train_df.columns)
col_d['e'] = n1
col_d[0:10]


# In[6]:


#Categorical Values
featu = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
         'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl','Exterior1st', 'Exterior2nd',
         'MasVnrType', 'ExterQual','ExterCond', 'Foundation','BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1',
        'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType',
        'GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition' ]

LF_mean = train_df.LotFrontage.mean()
train_df.LotFrontage = train_df.LotFrontage.fillna(train_df.LotFrontage.mean())
for x in featu:
    train_df.dropna(subset=[x], inplace=True)
    train_df

train_df.shape


# ### One Hot Encoding for Categorical Values

# In[7]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(handle_unknown = 'ignore')

def v(c1):    
    M = pd.DataFrame(onehotencoder.fit_transform(train_df[[c1]]).toarray())
    # get length of df's columns
    num_cols = len(list(M))
    rng = range(0, num_cols)
    new_cols = [c1 + str(i) for i in rng] 
    M.columns = new_cols[:num_cols]
    return M



MS = pd.concat([v(i) for i in featu], axis=1)

MS.head()


# In[8]:


MS.shape


# In[9]:


MS.isna().sum()


# In[10]:


MS = MS.reset_index(drop=True)
MS


# In[11]:


# Cleaning and selecting numerical columns
numer = train_df.copy()
num_sel = numer[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
               'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
               'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
               'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]
num_sel.head()


# In[12]:


num_sel.shape


# In[13]:


num_sel = num_sel.reset_index(drop=True)
num_sel


# In[14]:


num_sel.isna().sum()


# In[15]:


train_clean_df = pd.concat([MS, num_sel], axis=1)
train_clean_df.head()


# In[16]:


train_clean_df.shape


# In[17]:


train_clean_df.isna().sum()


# In[18]:


sales_train = train_df.SalePrice
sales_train_df = pd.DataFrame(sales_train)
sales_train_df = sales_train_df.reset_index(drop=True)
sales_train_df


# ### Test File

# In[19]:



featu_test = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
         'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl','Exterior1st', 'Exterior2nd',
         'MasVnrType', 'ExterQual','ExterCond', 'Foundation','BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1',
        'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType',
        'GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition' ]

#Preparing test dataset
LF_mean = test_df.LotFrontage.mean()
test_df.LotFrontage = test_df.LotFrontage.fillna(test_df.LotFrontage.mean())
for x in featu_test:
    test_df.dropna(subset=[x], inplace=True)
    
test_df


# In[20]:


test_df.shape


# In[21]:


test_df.isna().sum()


# In[22]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(handle_unknown = 'ignore')

def v_test(c1):    
    M = pd.DataFrame(onehotencoder.fit_transform(test_df[[c1]]).toarray())
    # get length of df's columns
    num_cols = len(list(M))
    rng = range(0, num_cols)
    new_cols = [c1 + str(i) for i in rng] 
    M.columns = new_cols[:num_cols]
    return M


MS_test = pd.concat([v_test(i) for i in featu], axis=1)
MS_test
MS_test = MS_test.reset_index(drop=True)
MS_test


# In[23]:


MS_test.shape


# In[24]:


numer_t = test_df.copy()
num_sel_t = numer_t[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
               'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
               'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
               'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]
num_sel_t = num_sel_t.reset_index(drop=True)
num_sel_t.head()


# In[25]:


num_sel_t.shape


# In[26]:


num_sel_t.isna().sum()


# In[27]:


test_clean_df = pd.concat([MS_test, num_sel_t], axis=1)
test_clean_df.head()


# In[28]:


test_clean_df.shape


# In[29]:


test_clean_df.isna().sum()


# ### Training and Validation

# In[30]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
y = sales_train_df.SalePrice
features = [x for x in train_clean_df.columns]
X = train_clean_df[features]
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)
model_r = RandomForestRegressor(random_state = 0)
model_r.fit(train_X, train_y)
predict_r = model_r.predict(val_X)
predict_r = pd.DataFrame(predict_r, columns=['SalePrice'])
predict_r


# In[31]:


mae = mean_absolute_error(val_y, predict_r)
mae


# In[32]:


val_y1 = pd.DataFrame(val_y)
val_y1 = val_y1.reset_index(drop=True)
val_y1


# In[33]:


score = model_r.score(val_X, val_y)
score


# In[34]:


val_X = val_X.sort_values('GarageArea')


# In[35]:


plt.rcParams["figure.figsize"] = [14, 8]
plt.plot(val_X.GarageArea, val_y)
plt.plot(val_X.GarageArea, predict_r)
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.legend(['Original', 'Predicted'])
plt.show()


# ### If test file has less features than train

# In[36]:


# Get missing columns in the training test
missing_cols = set( train_clean_df.columns ) - set( test_clean_df.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test_clean_df[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test_clean_df = test_clean_df[train_clean_df.columns]


# ### Predicting for test file

# In[37]:


cols = [x for x in test_clean_df.columns]
test_X = test_clean_df[cols]
predict_r_t = model_r.predict(test_X)
predict_r_t = pd.DataFrame(predict_r_t, columns=['SalePrice'])
predict_r_t


# In[ ]:




