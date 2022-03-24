import pandas as pd
import numpy as np
import os
from env import get_db_url
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split    
import warnings
warnings.filterwarnings("ignore")


# Function to clean Zillow data
def clean_zillow_data(df):
    # drop duplicates
    df = df.drop_duplicates()
    # drop null values
    df = df.dropna()
    # change types
    df[['bedrooms', 'year_built']] = df[['bedrooms', 'year_built']].astype(int)
    df.fips = df.fips.astype(object)
    return df


def wrangle_zillow():
    
    df = get_zillow_data()
    df = clean_zillow_data(df)

    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])       
    
    return train, validate, test   