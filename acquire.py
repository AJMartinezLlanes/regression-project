import pandas as pd
import numpy as np
import os
from env import get_db_url

# Function to get zillow data
def get_zillow_data():    

    '''This function will acquire data from zillow using env file and rename the columns before saving it as CSV'''

    filename = 'zillow.csv'
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
      
    query = '''
        SELECT bedroomcnt, 
                bathroomcnt, 
                calculatedfinishedsquarefeet, 
                taxvaluedollarcnt, 
                yearbuilt, 
                taxamount,
                fips 
        FROM properties_2017
        JOIN propertylandusetype USING (propertylandusetypeid)
        JOIN predictions_2017 USING(parcelid)
        WHERE propertylandusedesc IN ("Single Family Residential", 
                                        "Inferred Single Family Residential")
            AND transactiondate LIKE "2017%%";
        '''
    print('Getting a fresh copy from SQL database...')

    df = pd.read_sql(query, get_db_url('zillow'))
    print('Saving to csv...')
    
    df = df.rename(columns = {'bedroomcnt':'bedrooms',
                          'bathroomcnt':'bathrooms',
                          'calculatedfinishedsquarefeet':'sqft',
                          'taxvaluedollarcnt':'tax_value',
                          'yearbuilt':'year_built'})
    df.to_csv(filename, index=False)
    return df