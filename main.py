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

#machine learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

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
train = train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'Id'], axis = 1)
submission_test = submission_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'Id'], axis = 1)

#create this so it is easier to append the columns into the dataframe used to record the pearson R value
cols = train.columns
corr_df = pd.DataFrame(columns=['Feature', 'Pearsons R'])

#check from columns 0 -> 74 as we know there are 74 features left
for columnNo in range (0, 74):
    #checks if the first item in the row is a number
    if isinstance(train.dropna().iloc[0,columnNo], str) == False:
        #if so make a temp dataframe with the numerical values with the sale price
        df = train.iloc[:, [columnNo, 74]].dropna()
        #calculate the R value and only take the R value, as this function outputs a P-value too
        pearson_r = scipy.stats.pearsonr(df.iloc[:,0], df.iloc[:,1])[0]
        #add it to the column to compare
        corr_df = corr_df.append({'Feature' : cols[columnNo], 'Pearsons R' : pearson_r}, ignore_index = True)
        
'''
We want to keep features that have a strong negative or positive correlation, 
and ignore the features that don't have a strong correlation. Using this logic
I will keep only the features that have R value ranging between -1 -> -0.6 and 
0.6 -> 1.
This now means we can keep:
    OverallQual
    GrLivArea
    GarageCars
    GarageArea
    TotalBsmtSF
We've now narrowed down 36 numerical values down to 5. Depending on the analysis
of the non numerical values, I will drop GarageCars and GarageArea too and the
basement square foot can be combined with the basement quality. Some of these 
features can be engineered and reduced. I will look at this later.

To test the non-numerical features, I will get the mean value of each category 
of each feature and see if there is a difference in value of the mean value of the sale.
I will judge the effect a feature has by checking the standard deviation of the values.
For example for 'MSZoning', there are 4 possible values, 'C', 'FV', 'RL' and 'RM'.
If the standard deviation for the sale value for each of the 4 possible values is the low, then I will 
deem that feature not having an impact on the sale price.
'''



std_df = pd.DataFrame(columns=['Feature', 'Standard Deviation', 'Percentage from Mean'])
sale_mean = train[['SalePrice']].mean()
#check from columns 0 -> 74 as we know there are 74 features left
for columnNo in range (0, 74):
    #checks if the first item in the row is a string
    if isinstance(train.dropna().iloc[0,columnNo], str) == True:
        #calculate the mean value per category of the column
        pivot = train.iloc[:, [columnNo, 74]].dropna().groupby([cols[columnNo]], as_index=False).mean().sort_values(by=cols[columnNo], ascending=False)
        #calculate the Standard deviation
        standard_dev = np.std(pivot.iloc[:, 1])
        #compare standard deviation with the Sale Mean to get a percentage difference
        mean_perc = standard_dev/sale_mean
        #add it to the column to compare
        std_df = std_df.append({'Feature' : cols[columnNo], 'Standard Deviation' : standard_dev, 'Percentage from Mean' : mean_perc[0]}, ignore_index = True)

'''
There are quite a few features here that have quite a large standard deviation.
Let's look at the features that have a standard deviation of 70k Plus:
    ExterQual - Exterior Quality
    KitchenQual - Kitchen Quality
    BsmtQual - Basement Quality
    ************************************************
    Condition2 - Other ammenities nearby (if at all)
    RoofMatl- Roof Material
For now I will use these features.
'''    
train_test = train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'ExterQual', 'KitchenQual', 'BsmtQual', 'Condition2', 'RoofMatl', 'SalePrice']]
submission = submission_test[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'ExterQual', 'KitchenQual', 'BsmtQual', 'Condition2', 'RoofMatl']]

'''
Now that we have the final few columns, lets examine them.
I can immediately see that 'GarageCars' and 'GarageArea' are linked. These two columns
can be combined. We can bin the area size into 4/5 bins and then multiply this 
with the amount of cars that can be in the garage. The basement quality can be 
label encoded and then combined with the square area. As GarageCars and 
GarageArea are already numerical and don't recquire encoding, so I will work
with this data first.
'''

#First check how many NA there are in those 2 features
train_null_values = train_test.isnull().sum().sort_values(ascending=False)
submission_test_null_values = submission.isnull().sum().sort_values(ascending=False)

#BsmtQual has a few NA's and the GarageArea and GarageCars have a couple of NA's
#which I will deal with after combining it in train
garage_bin = train_test['GarageArea']
garage_bin= np.array(garage_bin, dtype=float)
garage_bands = pd.qcut(garage_bin, 6)

'''
By doing qcut, we divided by the garagearea into 6 equally sized bins by value.
The bin sizes are: 
    0 -> 281
    281 -> 400
    400 -> 480
    480 -> 540
    540 -> 660
    660 -> 1418
I will now create a column encoding the garage area
'''

train_test['GarageBins'] = pd.cut(x=train_test['GarageArea'], bins=[-1, 281, 400, 480, 540, 660, 1450], labels=[1, 2, 3, 4, 5, 6])
train_test['GarageBins'] = train_test['GarageBins'].astype('int64')

train_test['GarageSize'] = train_test['GarageCars'] * train_test['GarageBins']
train_test = train_test[['OverallQual', 'GrLivArea', 'GarageSize', 'ExterQual', 'KitchenQual', 'BsmtQual', 'Condition2', 'RoofMatl', 'SalePrice']]

#There is one NA value in the submission. As both for Garage Cars and Garage Bins are NA, I will fill this row in as 0
submission['GarageCars'] = submission['GarageCars'].fillna(0)
submission['GarageArea'] = submission['GarageArea'].fillna(0)

submission['GarageBins'] = pd.cut(x=submission['GarageArea'], bins=[-1, 281, 400, 480, 540, 660, 1450], labels=[1, 2, 3, 4, 5, 6])
submission['GarageBins'] = submission['GarageBins'].astype('int64')

submission['GarageSize'] = submission['GarageCars'] * submission['GarageBins']
submission = submission[['OverallQual', 'GrLivArea', 'GarageSize', 'ExterQual', 'KitchenQual', 'BsmtQual', 'Condition2', 'RoofMatl']]

'''
Next if we look at the columns ExterQual, KitchQual and BsmtQual.
They are all rating the quality of a house part from poor - excellent. For that reason, 
I will label encode (convert from words to 1-5 rating) all three columns, and then
combine them by adding them to give a total overall quality. 
Basement quality has some NA's so will need to find the average value of the 
ExterQual and KitchQual and the corresponding BsmtQual of the known values
and then use that to fill in the information
'''
quality_mapping = {"Ex" : 5, "Gd" : 4, "TA" : 3, "Fa" : 2, "Po" : 1}
train_test['ExterQual'] = train_test['ExterQual'].map(quality_mapping)
train_test['KitchenQual'] = train_test['KitchenQual'].map(quality_mapping)
train_test['BsmtQual'] = train_test['BsmtQual'].map(quality_mapping)
train_test['TotalQual'] = train_test['ExterQual'] + train_test['KitchenQual']

'''
I'm going to make a pivot table that get's the average total quality score per
basement quality score. From there I will use that to fill in values for the NA
'''
basment_pivot = train_test[['BsmtQual', 'TotalQual']].groupby(['BsmtQual'], as_index=False).mean().sort_values(by='TotalQual', ascending=False)

'''
As we can see, anything near a 9, we can assume the basement is in excellent quality.
Anything near an 7, we can assume the basement is in good quality.
Anything near a 6 we can assume the basement is in typical to fair quality.
I will do the following:
    Above 8 = Excellent
    7 = Good
    6 = Typical/Average
'''

#fill with 0's so that I can convert the zeros into numbers
train_test['BsmtQual'] = train_test['BsmtQual'].fillna(0)

for number in range(0, 1460):        
    if train_test.iloc[number,5] == 0:
        if train_test.iloc[number, 9] <= 6:
            train_test.iloc[number, 5] = 3
        elif train_test.iloc[number, 9] == 7:
            train_test.iloc[number, 5] = 4
        else:
            train_test.iloc[number, 5] = 5
            
train_test['TotalQual'] = train_test['ExterQual'] + train_test['KitchenQual'] + train_test['BsmtQual']

train_test = train_test[['OverallQual', 'GrLivArea', 'GarageSize', 'TotalQual', 'Condition2', 'RoofMatl', 'SalePrice']]

#Now do the same thing with the submission dataset
submission['ExterQual'] = submission['ExterQual'].map(quality_mapping)
submission['KitchenQual'] = submission['KitchenQual'].map(quality_mapping)
submission['BsmtQual'] = submission['BsmtQual'].map(quality_mapping)
submission['TotalQual'] = submission['ExterQual'] + submission['KitchenQual']

#fill with 0's so that I can convert the zeros into numbers
submission['BsmtQual'] = submission['BsmtQual'].fillna(0)

for number in range(0, 1459):        
    if submission.iloc[number,5] == 0:
        if submission.iloc[number, 8] <= 6:
            submission.iloc[number, 5] = 3
        elif submission.iloc[number, 8] == 7:
            submission.iloc[number, 5] = 4
        else:
            submission.iloc[number, 5] = 5
            
submission['TotalQual'] = submission['ExterQual'] + submission['KitchenQual'] + submission['BsmtQual']

submission = train_test[['OverallQual', 'GrLivArea', 'GarageSize', 'TotalQual', 'Condition2', 'RoofMatl']]

train_null_values = train_test.isnull().sum().sort_values(ascending=False)
submission_test_null_values = submission.isnull().sum().sort_values(ascending=False)

cond2_pivot = train_test[['Condition2', 'SalePrice']].groupby(['Condition2'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
roof_pivot = train_test[['RoofMatl', 'SalePrice']].groupby(['RoofMatl'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)

'''
I've decided I CBA with the roof material and condition2 at the moment. I'm going to drive forward
with the 4 features I have at the moment.
'''

submission = train_test[['OverallQual', 'GrLivArea', 'GarageSize', 'TotalQual']]
train_test = train_test[['OverallQual', 'GrLivArea', 'GarageSize', 'TotalQual', 'SalePrice']]

'''
Time to add Feature Scaling. Need to make all the values in the tables between 1 and -1.
'''

x = train_test.iloc[:, 0:4].values
y = train_test.iloc[:, 4].values

#Reshape the Y values so it is an array
y = y.reshape(len(y), 1)

sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)
'''
Model Training
'''

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#SVM
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

prediction = regressor.predict(X_test)
accuracy_svm = r2_score(y_test, prediction)

#RandomForest
Dtree = RandomForestRegressor(n_estimators = 200, random_state=0)
Dtree.fit(X_train, y_train)
Y_Dtree = Dtree.predict(X_test)
accuracy_Dtree = r2_score(y_test, Y_Dtree)

prediction_submission = Dtree.predict(submission)
df = pd.DataFrame({'SalePrice' : prediction_submission})
df['Id'] = submission_test['Id']
df = df[['Id', 'SalePrice']]
df.to_csv('submission.csv')
print(df.columns.values)