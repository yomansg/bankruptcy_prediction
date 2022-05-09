# -*- coding: utf-8 -*-
"""
Spyder Editor

Code snippets for Loan Default/ Bankruptcy Prediction
Done by: Theng Hee Yeo
TFIP-Data Analytics 2021
NUS-ISS

"""
#########################################################
##                                                     ##
##     STAGE 1:  DATA Import/ Preparation              ##
##                                                     ##
##                                                     ##
#########################################################


# Import libraries and use chardet to identify encoding
import pandas as pd
import chardet

rawdata = open("TEJ - Normalized_2013_0430_Taiwan_data.csv", 'rb').read()
result = chardet.detect(rawdata)
charenc = result['encoding']

dataset = pd.read_csv('TEJ - Normalized_2013_0430_Taiwan_data.csv', encoding='Big5')
dataset.columns

## Half of the columns are in Chinese, 
## the data dictionary in the original UCI Machine Learning Repository is not useful because the dictionary is not in the same order.
## Therefore, head over to Google translate ... and then we get the translated list here.

translated_list = ['Flag','ROA(C) before tax and interest before depreciation','ROA(A) after tax and before interest%','ROA(B) after tax and before interest and depreciation','Operating gross profit margin',
       'Realized gross profit margin of sales','operating profit rate','net profit margin before tax','net profit margin after tax','non-industry revenue and expenditure/revenue','continuous profit rate (after tax)',
       'Operating expense ratio','Research and development expense ratio','Cash flow ratio','Interest-bearing debt interest rate','Tax rate (A)','Net value per share (B)',
       'Net value per share (A)','Net value per share (C)','Persistent EPS in the last four seasons','Cash flow per share','Return per share (yuan)','Operating profit per share ( Yuan)',
       'Net profit per share before tax (yuan)','Realized gross profit growth rate of sales','Operating profit growth rate','After-tax net profit growth rate','Regular net profit growth rate','Continuous net profit growth rate' ,
       'Total Asset Growth Rate','Net Worth Growth Rate','Return on Total Assets Growth Rate','Cash Reinvestment %','Current Ratio','Quick Ratio','Interest Expenditure Rate',
       'Total liabilities/total net worth','debt ratio%','net worth/assets','long-term fund suitability ratio (A)','borrowing dependence','contingent liabilities/net worth',
       'Operating profit/paid-in capital','pre-tax net profit/paid-up capital','inventory and accounts receivable/net value','total asset turnover times','accounts receivable turnover times',
       'Average collection days','Inventory turnover rate (times)','Fixed asset turnover times','Net worth turnover rate (times)','Revenue per person','Operating profit per person',
       'Equipment rate per person','working capital to total assets','Quick asset/Total asset',
       'current assets/total assets','cash / total assets',
       'Quick asset/current liabilities','cash / current liability',
       'current liability to assets','operating funds to liability',
       'Inventory/working capital','Inventory/current liability',
       'current liability / liability','working capital/equity',
       'current liability/equity','long-term liability to current assets',
       'Retained Earnings/Total assets','total income / total expense',
       'total expense /assets','current asset turnover rate','quick asset turnover rate','working capitcal turnover rate',
       'Cash flow rate','Cash flow to Sales','fix assets to assets',
       'current liability to liability','current liability to equity',
       'equity to long-term liability','Cash flow to total assets',
       'cash flow to liability','CFO to ASSETS','cash flow to equity',
       'current liabilities to current assets',
       'one if total liabilities exceeds total assets zero otherwise',
       'net income to total assets','total assets to GNP price',
       'No-credit interval','Gross profit to Sales',
       'Net income to stockholder\'s Equity','liability to equity',
       'Degree of financial leverage (DFL)',
       'Interest coverage ratio( Interest expense to EBIT )',
       'one if net income was negative for the last two year zero otherwise',
       'equity to liability'] 

# Make your column casing consistent
translated_list_title = []
for i in range(len(translated_list)):
    translated_list_title.append(translated_list[i].title())

dataset.columns = translated_list_title
dataset.to_csv('dataset.csv', index=False)      # Export the DataFrame as CSV



#########################################################
##                                                     ##
##     STAGE 2:  Exploratory Data Analysis             ##
##                                                     ##
##                                                     ##
#########################################################

# Import libraries for Data Exploration
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
pd.options.display.max_rows = 999       # Set pandas to display all columns

dataset = pd.read_csv('dataset.csv')
dataset.describe

# We found one row with all rows identical, hence we drop that column
dataset.drop(columns=["One If Net Income Was Negative For The Last Two Year Zero Otherwise"], inplace=True)


# Plot a barplot for "Flag"
ax = sns.countplot(x="Flag", data=dataset)

# Split Dataframe into normal and bankrupt
normal = dataset[dataset.Flag == 0]
bankrupt = dataset[dataset.Flag == 1]

# Data exploration for 'Debt Ratio%'
sns.histplot(data=dataset, x="Debt Ratio%", hue="Flag")     # Plot histogram
sns.boxplot(x='Flag', y='Debt Ratio%', data=dataset, showfliers = False)    # plot a boxplot 
from scipy.stats import mannwhitneyu
results = mannwhitneyu(normal['Debt Ratio%'], bankrupt['Debt Ratio%'])  # Perform a Mann-Whitney test
results

# Data exploration for 'Current ratio%'
sns.histplot(data=dataset, x="Current Ratio", hue="Flag")
sns.boxplot(x='Flag', y='Current Ratio', data=dataset, showfliers = False)
results = mannwhitneyu(normal['Current Ratio'], bankrupt['Current Ratio'])

# Data exploration for 'Quick ratio%'
sns.histplot(data=dataset, x="Quick Ratio", hue="Flag")
sns.boxplot(x='Flag', y='Quick Ratio', data=dataset, showfliers = False)
results = mannwhitneyu(normal['Quick Ratio'], bankrupt['Quick Ratio'])

# Data exploration for 'Net Profit Margin Before Tax'
sns.histplot(data=dataset, x="Net Profit Margin Before Tax", hue="Flag")
sns.boxplot(x='Flag', y='Net Profit Margin Before Tax', data=dataset, showfliers = False)
results = mannwhitneyu(normal['Net Profit Margin Before Tax'], bankrupt['Net Profit Margin Before Tax'])



# Create a crosstab between Flag and One If Total Liabilities Exceeds Total Assets Zero Otherwise
cross_tab = pd.crosstab(dataset['Flag'], dataset['One If Total Liabilities Exceeds Total Assets Zero Otherwise'])
from scipy.stats import chi2_contingency 
stat, p, dof, expected = chi2_contingency(cross_tab)
print('p-value is: ', p)
## Drop also that column that is Not independent of flag
dataset.drop(columns=['One If Total Liabilities Exceeds Total Assets Zero Otherwise'], inplace=True)


## Perform complete Data Exploration on all the features
## Loop through everything to inspect all the boxplots, including  mannwhitneyu test of difference
from scipy.stats import mannwhitneyu
Flag = dataset['Flag']
dataset_features = dataset.drop(columns=['Flag'])
features = dataset_features.columns
for i, col in enumerate(features):
    plt.figure(i)
    result = mannwhitneyu(normal[features[i]], bankrupt[features[i]])[1]    ## get p-value of the mannwhitneyu test of diff
    title = 'Flag vs ' + features[i] + '\n p-value: {:.2E}'.format(result)          ## print p-value in scientific notation
    sns.boxplot(x='Flag', y=features[i], data=dataset, showfliers = False).set(title = title)  ## output boxplot and title


## list of selected features and then export to CSV file for next steps
selected_features = ['Roa(C) Before Tax And Interest Before Depreciation',
'Roa(A) After Tax And Before Interest%',
'Roa(B) After Tax And Before Interest And Depreciation',
'Operating Gross Profit Margin',
'Operating Profit Rate',
'Net Profit Margin Before Tax',
'Non-Industry Revenue And Expenditure/Revenue',
'Continuous Profit Rate (After Tax)',
'Cash Flow Ratio',
'Interest-Bearing Debt Interest Rate',
'Tax Rate (A)',
'Persistent Eps In The Last Four Seasons',
'Cash Flow Per Share',
'Net Worth Growth Rate',
'Current Ratio',
'Quick Ratio',
'Interest Expenditure Rate',
'Total Liabilities/Total Net Worth',
'Debt Ratio%',
'Net Worth/Assets',
'Borrowing Dependence',
'Pre-Tax Net Profit/Paid-Up Capital',
'Working Capital To Total Assets',
'Cash / Current Liability',
'Current Liability To Assets',
'Net Income To Total Assets',
'Liability To Equity',
'Degree Of Financial Leverage (Dfl)']

dataset_ML = pd.concat([Flag, dataset[selected_features]], axis=1)
dataset_ML.to_csv('dataset_ML.csv', index=False)


#########################################################
##                                                     ##
##     STAGE 3:  Machine Learning                      ##
##                                                     ##
##                                                     ##
#########################################################

# Steps 1 and 2: Import libraries and read datafile
import pandas as pd
pd.set_option("display.precision", 4)
dataset = pd.read_csv('dataset_ML.csv')

# Step 3: Prepare your independent and dependent variables
X = dataset.drop(columns=['Flag'])
y = dataset['Flag']


# Step 4: Import the machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Step 5: Split your data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Step 6a: Train a DummyClassifier
dummy_model = DummyClassifier()
dummy_model.fit(X_train, y_train)
# Step 6b: Make predictions with the DummyClassifier on Training Data
y_train_pred = dummy_model.predict(X_train)

# Step 6c: Get the f1_score and show the confusion matrix between test and prediction
print(f'The f1_score is {f1_score(y_train, y_train_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_train, y_train_pred)

# Step 6d: Repeat the above steps on Testing Data
dummy_model_pred = dummy_model.predict(X_test)
print(f'The f1_score is {f1_score(y_test, dummy_model_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, dummy_model_pred)

# Step 6e: Print the classification report between test and prediction
print(classification_report(y_test, dummy_model_pred))


# Step 7a: Train LogisticRegression model
lr = LogisticRegression(n_jobs=-1)
lr.fit(X_train, y_train)

# Step 7b: Make predictions with LogisticRegression model on Training Data
y_train_pred = lr.predict(X_train)

# Step 7c: Get the f1_score and show the confusion matrix between test and prediction
print(f'The f1_score is {f1_score(y_train, y_train_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_train, y_train_pred)

# Step 7d: Repeat the above steps on Testing Data
lr_pred = lr.predict(X_test)
print(f'The f1_score is {f1_score(y_test, lr_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, lr_pred)

# Step 7e: Print the classification report
print(classification_report(y_test, lr_pred))


# Step 8a: Train a RandomForestClassifier model
rf = RandomForestClassifier(n_estimators = 10, random_state = 0)
rf.fit(X_train, y_train)

# Step 8b: Make predictions with RandomForestClassifier model on Training Data
y_train_pred = rf.predict(X_train)

# Step 8c: Get the f1_score and show the confusion matrix between test and prediction
print(f'The f1_score is {f1_score(y_train, y_train_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_train, y_train_pred)

# Step 8d: Repeat the above steps on Testing Data
rf_pred = rf.predict(X_test)
print(f'The f1_score is {f1_score(y_test, rf_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, rf_pred)

# Step 8e: Print the classification report
print(classification_report(y_test, rf_pred))


# Step 9a: Train a GradientBoostingClassifier model
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=0)
clf.fit(X_train, y_train)

# Step 9b: Make predictions with GradientBoostingClassifier model on Training Data
y_train_pred = clf.predict(X_train)

# Step 9c: Get the f1_score and show the confusion matrix between test and prediction
print(f'The f1_score is {f1_score(y_train, y_train_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_train, y_train_pred)

# Step 9d: Repeat the above steps on Testing Data
clf_pred = clf.predict(X_test)
print(f'The f1_score is {f1_score(y_test, clf_pred):0.2f}')
print('The confusion matrix is ')
print(confusion_matrix(y_test, clf_pred))

# Step 10: Get the feature importances of the model
feature_importance = list(zip(X_train.columns, clf.feature_importances_))
feature_importance.sort(key=lambda x: x[1], reverse=True)

feature_importance = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
#feature_importance['importance'] = round(feature_importance['importance'], 5)
feature_importance.head(5)


#########################################################
##                                                     ##
##     STAGE 4:  Advanced Modelling                    ##
##                Re-sampling                          ##
##                                                     ##
#########################################################

# Step 1: Import your libraries
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek

# Step 2: Read CSV from Part II
dataset = pd.read_csv('dataset_ML.csv')

# Step 3: Prepare your independent and dependent variables
X = dataset.drop(columns=['Flag'])
y = dataset['Flag']

# Step 4: Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Step 5: Upsample your train data 
## SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

## ADASYN
sm_ADASYN = ADASYN(random_state=42)
X_res, y_res = sm_ADASYN.fit_resample(X_train, y_train)

## SMOTEENN
sm_SMOTEENN = SMOTEENN(random_state=42)
X_res, y_res = sm_SMOTEENN.fit_resample(X_train, y_train)

## SMOTETomek
sm_SMOTETomek = SMOTETomek(random_state=42)
X_res, y_res = sm_SMOTETomek.fit_resample(X_train, y_train)


#### The following code is for the various models - to reuse with each upsampler above #### 

###############  LOGISTIC REGRESSION  ###############
# Step 7a: Train a LogisticRegression model
lr = LogisticRegression(n_jobs=-1)
lr.fit(X_res, y_res)

# Step 7b: Assess LogisticRegression model on Training Data
y_train_pred = lr.predict(X_res)
print(f'The f1_score is {f1_score(y_res, y_train_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_res, y_train_pred)

# Step 7c: Assess LogisticRegression model on Testing Data
lr_pred = lr.predict(X_test)
print(f'The f1_score is {f1_score(y_test, lr_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, lr_pred)

# get feature importance, display and plot
importance = lr.coef_[0]

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

importance = lr.coef_[0]
feature_importance = list(zip(X_train.columns, importance, abs(importance)))
feature_importance.sort(key=lambda x: x[2], reverse=True)

feature_importance = pd.DataFrame(feature_importance, columns=['feature', 'importance', 'rank'])
feature_importance = feature_importance[['feature', 'importance']]
feature_importance.head(10)

###############  RANDOM FOREST CLASSIFIER ###############
# Step 8a: Train a RandomForestClassifier model
rf = RandomForestClassifier(n_estimators = 10, random_state=42)
rf.fit(X_res, y_res)

# Step 8b: Assess model performance on Training Data
y_train_pred = rf.predict(X_res)
print(f'The f1_score is {f1_score(y_res, y_train_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_res, y_train_pred)

# Step 8c: Assess model performance on Testing Data
rf_pred = rf.predict(X_test)
print(f'The f1_score is {f1_score(y_test, rf_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, rf_pred)


###############  GRADIENT BOOSTING CLASSIFIER ###############
# Step 9a: Train a GradientBoostingClassifier model
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
clf.fit(X_res, y_res)

# Step 9b: Assess model performance on Training Data
y_train_pred = clf.predict(X_res)
print(f'The f1_score is {f1_score(y_res, y_train_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_res, y_train_pred)

# Step 9c: Assess model performance on Testing Data
clf_pred = clf.predict(X_test)
print(f'The f1_score is {f1_score(y_test, clf_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, clf_pred)



#########################################################
##                                                     ##
##     STAGE 4:  Advanced Modelling                    ##
##                Hyperparameter Tuning                ##
##                                                     ##
#########################################################

# Step 10: Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Step 11: Define the parameter grid
param_grid = {'n_estimators': [5, 10, 25, 50, 100],
              'max_depth': [1, 2, 3, 4, 5], 
              'max_features': [9, 11, 13, 15, 17]}

# Step 12: Declare a GridSearchCV object
gs = GridSearchCV(
    estimator = GradientBoostingClassifier(random_state=0),
    param_grid= param_grid,
    scoring = 'recall',
    n_jobs = 4,
    cv=5)

# Step 13: Fit upsampled train data with GridSearchCV object
gs.fit(X_res, y_res)

# Step 14: Get your best parameters
print(f'The best parameters are {gs.best_params_} with a score of {gs.best_score_}')

# Step 15a: Train model with the new parameters
clf = gs.best_estimator_
clf.fit(X_res, y_res)

# Step 15b: Assess model performance
pred = clf.predict(X_test)

print(f'The f1_score is {f1_score(y_test, pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, pred)


##### Try on Random forest Classifier  #####
from sklearn.model_selection import GridSearchCV 

# Define the parameter grid
param_grid = {'n_estimators': [5, 10, 50, 100, 300, 500],
              'max_depth': [1, 3, 5, 8, 15],
              'max_features': [9, 11, 13, 15, 17]}

# Declare a GridSearchCV object
gs = GridSearchCV(
    estimator = RandomForestClassifier(random_state = 0),
    param_grid= param_grid,
    scoring = 'recall',
    n_jobs = 4,
    cv=5)

# Fit upsampled train data with GridSearchCV object
gs.fit(X_res, y_res)
print(f'The best parameters are {gs.best_params_} with a score of {gs.best_score_}')

# Train model with the new parameters
rf = gs.best_estimator_
rf.fit(X_res, y_res)

# Assess model performance
rf_pred = rf.predict(X_test)
print(f'The f1_score is {f1_score(y_test, rf_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, rf_pred)


##### Try on Logistic Regression #####
from sklearn.model_selection import GridSearchCV 

# Define the parameter grid
param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'penalty': ['l2'],
              'C': [1000, 100, 10, 1.0, 0.1, 0.01]}

# Declare a GridSearchCV object
gs = GridSearchCV(
    estimator = LogisticRegression(),
    param_grid= param_grid,
    scoring = 'recall',
    n_jobs = 4,
    cv=5)

# Fit upsampled train data with GridSearchCV object
gs.fit(X_res, y_res)
print(f'The best parameters are {gs.best_params_} with a score of {gs.best_score_}')

# Train model with the new parameters
#lr = rs.best_estimator_
lr = gs.best_estimator_
lr.fit(X_res, y_res)

# Assess model performance
lr_pred = lr.predict(X_test)
print(f'The f1_score is {f1_score(y_test, lr_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, lr_pred)


#########################################################
##                                                     ##
##     STAGE 4:  Advanced Modelling                    ##
##                Feature Engineering                  ##
##                                                     ##
#########################################################

# Step 1: Import your libraries
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek


## Step 2: Use only TOP 8 features from Feature Importance study
dataset = pd.read_csv('dataset_ML.csv')

selected_features = ['Roa(C) Before Tax And Interest Before Depreciation',
'Roa(A) After Tax And Before Interest%',
'Roa(B) After Tax And Before Interest And Depreciation',
'Persistent Eps In The Last Four Seasons',
'Debt Ratio%',
'Net Worth/Assets',
'Working Capital To Total Assets',
'Net Income To Total Assets']

Flag = dataset['Flag']
dataset_ML_new = pd.concat([Flag, dataset[selected_features]], axis=1)
dataset_ML_new.to_csv('dataset_ML_new.csv', index=False)

# Step 3: Read from subset file, then decide your independent and dependent variables
dataset = pd.read_csv('dataset_ML_new.csv')
X = dataset.drop(columns=['Flag'])
y = dataset['Flag']

# Step 4: Split your data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Step 5: Upsample your train data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

## ADASYN
sm_ADASYN = ADASYN(random_state=42)
X_res, y_res = sm_ADASYN.fit_resample(X_train, y_train)

## SMOTEENN
sm_SMOTEENN = SMOTEENN(random_state=42)
X_res, y_res = sm_SMOTEENN.fit_resample(X_train, y_train)

## SMOTETomek
sm_SMOTETomek = SMOTETomek(random_state=42)
X_res, y_res = sm_SMOTETomek.fit_resample(X_train, y_train)


#### The following code is for the various models - to reuse with each upsampler above #### 

###############  LOGISTIC REGRESSION  ###############
# Step 7a: Train a LogisticRegression model
lr = LogisticRegression(n_jobs=-1)
lr.fit(X_res, y_res)

# Step 7b: Assess LogisticRegression model on Training Data
y_train_pred = lr.predict(X_res)
print(f'The f1_score is {f1_score(y_res, y_train_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_res, y_train_pred)

# Step 7c: Assess LogisticRegression model on Testing Data
lr_pred = lr.predict(X_test)
print(f'The f1_score is {f1_score(y_test, lr_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, lr_pred)

# get feature importance, display and plot
importance = lr.coef_[0]

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

importance = lr.coef_[0]
feature_importance = list(zip(X_train.columns, importance, abs(importance)))
feature_importance.sort(key=lambda x: x[2], reverse=True)

feature_importance = pd.DataFrame(feature_importance, columns=['feature', 'importance', 'rank'])
feature_importance = feature_importance[['feature', 'importance']]
feature_importance.head(10)

###############  RANDOM FOREST CLASSIFIER ###############
# Step 8a: Train a RandomForestClassifier model
rf = RandomForestClassifier(n_estimators = 10, random_state=42)
rf.fit(X_res, y_res)

# Step 8b: Assess model performance on Training Data
y_train_pred = rf.predict(X_res)
print(f'The f1_score is {f1_score(y_res, y_train_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_res, y_train_pred)

# Step 8c: Assess model performance on Testing Data
rf_pred = rf.predict(X_test)
print(f'The f1_score is {f1_score(y_test, rf_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, rf_pred)


###############  GRADIENT BOOSTING CLASSIFIER ###############
# Step 9a: Train a GradientBoostingClassifier model
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
clf.fit(X_res, y_res)

# Step 9b: Assess model performance on Training Data
y_train_pred = clf.predict(X_res)
print(f'The f1_score is {f1_score(y_res, y_train_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_res, y_train_pred)

# Step 9c: Assess model performance on Testing Data
clf_pred = clf.predict(X_test)
print(f'The f1_score is {f1_score(y_test, clf_pred):0.2f}')
print('The confusion matrix is ')
confusion_matrix(y_test, clf_pred)

