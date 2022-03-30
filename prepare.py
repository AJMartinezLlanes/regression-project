import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from env import get_db_url
from sklearn.model_selection import train_test_split    
from acquire import get_zillow_data

import warnings
warnings.filterwarnings("ignore")

# Remove Outliers Function
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df

# Clean Zillow Data Function
def clean_zillow_data(df):
    
    '''This function will clean Zillow data'''

    # drop duplicates
    df = df.drop_duplicates()
    
    # drop null values
    df = df.dropna()
    
    # change types
    df[['bedrooms', 'year_built']] = df[['bedrooms', 'year_built']].astype(int)
    df.fips = df.fips.astype(object)

    # rename fips column 
    df = df.rename(columns={'fips':'county'})
    
    # replace values for county names
    df = df.replace({'county':{6111:'Ventura', 6059:'Orange', 6037:'Los_Angeles'}})
    
    return df

# Split Data Function
def split_zillow_data(df):

    ''' this function will take your raw data frame, clean it and split it'''
    
    # cleans data using function
    df = clean_zillow_data(df)
    
    # split the data
    train_validate, test = train_test_split(df, test_size=.2, random_state=177)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=177)
    
    # show the split
    print(f'Dataframe has been split: ')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    # return train validate and test
    return train, validate, test

# Exploration Function
def exploration_proc(df):

    '''This function will give the explotation charts used'''
    
    plt.title('Splitted Dataframe Heatmap')
    sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
    
    sns.pairplot(df, kind= 'hist')

# Question 1 Function
def get_q1(df):

    '''This function is going to cover the charts for question 1'''

    # bathroom vs tax value
    plt.title('Tax Value vs Bathroom')
    sns.boxenplot(y= 'tax_value', x= 'bathrooms', data = df)
    plt.show()

    # bedroom vs tax value
    plt.title('Tax Value vs Bedroom')
    sns.boxplot(y= 'tax_value', x= 'bedrooms', data = df)
    plt.show()
    
    # tax value vs bathroom and bedrooms
    plt.title('Tax Value vs Bathroom with Bedrooms as hue')
    sns.barplot(x= 'bathrooms', hue= 'bedrooms', y='tax_value', data = df)
    plt.show()

# Question 2 Function
def get_q2(df):

    '''This function is going to cover the charts for question 2'''

    # Tax Value vs Age
    sns.relplot(y='tax_value', x= 'year_built', data=df, kind='line').set(title='Tax Value vs Year Built')
    plt.show()

# Question 2 Stat test Function
def q2_stat_test(df):
    # setting up the alpha
    alpha = .05
    
    # state the hypothesis
    print('HO: There is no relation between Tax Value and Year Built')
    print('H⍺: There is a relation between Tax Value and Year Built')
    print('')
    
    # statistical test  
    observed = pd.crosstab(df.tax_value, df.year_built)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print('Observed\n:')
    print(observed.values)
    print('------------------------\nExpected: \n')
    print(expected.astype(int))
    print('------------------------\n')
    print(f'chi2 = {chi2:.2f}')
    print(f'p value: {p:.4f}')
    print(f'degrees of freedom: {degf}')
    if (p < alpha):
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")

# Question 3 Function
def get_q3(df):
    
    '''This function is going to cover the charts for question 3'''

    # Tax Value vs sqft
    sns.displot(data= df, x='sqft', y='tax_value',cbar=True).set(title='Tax Value vs Square Footage')
    plt.show()
    sns.jointplot(data= df, x='sqft', y='tax_value', kind ='hist')
    plt.show()

# Question 4 Function
def get_q4(df):
    
    '''This function is going to cover the charts for question 4'''

    # Tax Value vs sqft
    plt.title('Tax Value vs County')
    sns.violinplot(y='tax_value', x= 'county', data=df)
    sns.catplot(y='tax_value', x= 'county', data=df)

# Question 4 Stat Test Function
def q4_stat_test(df):
    # setting up the alpha
    alpha = .05
    
    # state the hypothesis
    print('HO: There is no relation between Tax Value and County')
    print('H⍺: There is a relation between Tax Value and County')
    print('')
    
    # statistical test  
    observed = pd.crosstab(df.tax_value, df.county)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print('Observed\n:')
    print(observed.values)
    print('------------------------\nExpected: \n')
    print(expected.astype(int))
    print('------------------------\n')
    print(f'chi2 = {chi2:.2f}')
    print(f'p value: {p:.4f}')
    print(f'degrees of freedom: {degf}')
    if (p < alpha):
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")