"""
**********Plan of action*************

~~~~Understanding the Problem~~~
"Ask a home buyer to describe their dream house, and they probably won't begin 
with the height of the basement ceiling or the proximity to an east-west railroad. 
But this playground competition's dataset proves that much more influences price 
negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential 
homes in Ames, Iowa, this competition challenges you to predict the final price 
of each home."

~~~Initial understanding of the data~~~
1. Describe the data:
    Understand the type of data in the dataset. Is it numerical or categorical?
2. Identify possible algorithms to be used
3. Identify how many null values there are in comparison to the data points:
    If there are too many null values to non-null values, might be worth dropping the column
4. Check trends against features:
    Check how each feature affects the price of the house
    Check correlation.
5. Identify what features I will keep and what features I can 'engineer'

~~~Data Cleansing and Engineering~~~
1. Clean the data for the features selected
2. Engineer the features

~~~Data Encoding~~~
1. Decide on OneHotEncoding and LabelEncoding
2. Prepare the data to be encoded
3. Normalise the data

~~~Data Training~~~
1. Train the data from chosen algorithms
2. Do grid search and ROC

~~~Submission~~~
1. Submit my predictions

"""

'''
IMPORT LIBRARIES
'''

#data analysis and visualisation
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sn
import scipy.stats

'''
UNDERSTANDING THE PROBLEM AND DATA
'''

#describe the data
train = pd.read_csv('train.csv')
#traditionally call the test data 'test' but as we don't know the answers to the test data
#I'll be splitting the train data to test on, so to avoid confusion I called it
#submission data
submission_test = pd.read_csv('test.csv')

#see the columns and the type of information in the train and submission data
print(train.columns.values)
print(submission_test.columns.values)

'''
The submission is the same as train, just without the saleprice at the end.
There is a lot of columns, so will definitely have to reduce the dimensionality
'''
train.info()
'''
There is a big distribution between text and numerical data.
There are 80 features. We want to reduce this down to about 5/6.
The training data has 3 float64, 35 int64 and 43 text based features
There are 1460 entries.
'''
submission_test.info()
'''
Submission data seems to have more floating points than the training data - 
This will need to be unified.
There are 1459 entries.
'''
train_null_values = train.isnull().sum().sort_values(ascending=False)
submission_test_null_values = submission_test.isnull().sum().sort_values(ascending=False)
'''
Initially, lets drop the values that have around 50% of the values missing.
This means dropping:
    PoolQC
    MiscFeature
    Alley
    Fence
    FireplaceQu
We can also drop ID
Right now there are still 74 features left. 
To help reduce this, let's plot the house price against the numerical feature and calculate the
correlation.
'''
train = train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1)
submission_test = submission_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1)

corr_df = pd.DataFrame(columns=['Feature', 'Pearsons R'])
dfx = train['MSSubClass'].dropna()
dfy = train['SalePrice'].dropna()
pearson_r = scipy.stats.pearsonr(dfx, dfy)[0]
corr_df = corr_df.append({'Feature' : 'MSSubClass', 'Pearsons R' : pearson_r}, ignore_index = True)

#if (np.issubdtype(train.dropna().iloc[0,0], !np.number))
