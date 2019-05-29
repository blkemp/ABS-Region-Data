import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import os
from textwrap import wrap
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import eli5

nb_path = os.getcwd()
nb_path

def removeaggregation(df):
    df.drop(df[df['CODE'].astype(str).map(len) <= 8].index, inplace=True)
    return df

def load_merge_clean(nb_path = os.getcwd()):
    df_income = pd.read_csv('{}\CSV\Income_ASGS_Final.csv'.format(nb_path), na_values='-', thousands=',')
    df_pop = pd.read_csv('{}\CSV\Population and People_ASGS.csv'.format(nb_path), na_values='-', thousands=',')
    df_solar = pd.read_csv('{}\CSV\Land and Environment_ASGS.csv'.format(nb_path), na_values='-', thousands=',')
    df_fam = pd.read_csv('{}\CSV\Family and Community_ASGS.csv'.format(nb_path), na_values='-', thousands=',')
    
    df_list = [df_income, df_solar, df_fam, df_pop]
    df_income, df_solar, df_fam, df_pop = [df.pipe(removeaggregation) for df in df_list]

    df = pd.merge(df_income, df_solar, how='inner', left_on=['CODE','YEAR','LABEL'], right_on=['CODE','YEAR','LABEL'])

    cols_to_use = df_pop.columns.difference(df.columns).tolist()
    cols_to_use.extend(['CODE','YEAR','LABEL'])

    df = pd.merge(df, df_pop[cols_to_use], how='inner', left_on=['CODE','YEAR','LABEL'], right_on=['CODE','YEAR','LABEL'])

    cols_to_use = df_fam.columns.difference(df.columns).tolist()
    cols_to_use.extend(['CODE','YEAR','LABEL'])

    df = pd.merge(df, df_fam, how='inner', left_on=['CODE','YEAR','LABEL'], right_on=['CODE','YEAR','LABEL'])

    df.set_index(['CODE', 'LABEL', 'YEAR'], inplace=True)

    df = clean_data(df, 'YEAR')

    return df


def clean_data(df, fill_mean_subset = None):
    '''
    A function to clean a dataframe and return X & y values for further processing. 
    Rows are removed where NaNs are present in response vector records.
    NaN values for all other features are filled with the mean of the feature.
    
    INPUT
    df - pandas dataframe 
    y_column - String. Name of column to be used as the response vector
    fill_mean_subset - String, column name. Allows the input of a column to "subset" when first completing
                        imputing missing numerical values with a series mean. E.g. if there is a categorical 
                        field of "year", allows imputing of null values with the mean of each year, rather 
                        than the mean of the overall series.  
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    '''    
    # Remove duplicate columns
    drop_cols = []
    check_cols = df.columns.tolist()
    check_cols.sort()
    w_end = len(check_cols)
    i = 0
    
    # Cycle through each column name
    while i < w_end:
        # assign a Check variable the the column name as a string
        # that name string should only include characters up to 1 character after the final space
        # e.g. "* %" or "* n"
        check_str = check_cols[i]
        check_str = check_str[:(check_str.rfind(" ")+2)]
        
        for col in check_cols[(i+1):]:
            # look forward in the list of column names for any other items matching CheckString & "*"
            # add any matches to a list to drop, drop from the "check" list as well so make further searches more efficient.
            # I'm almost certain there is a more efficient way to do this list/dict-wise
            if col.startswith(check_str):
                drop_cols.append(col)
                check_cols.remove(col)
                w_end -= 1
        i += 1  
    
    df.drop(drop_cols, axis = 1, inplace=True)
    
    # Drop wmpty columns
    df = df.dropna(how = 'all', axis = 1)

    
    # Fill numeric columns with the mean    
    num_vars = df.select_dtypes(include=['float', 'int']).columns
    # First, fill with the mean of the subset based on given category
    if fill_mean_subset != None:
        index_reset = False
        index_names = list(df.index.names)
        
        # Filtering sucks with multi-indexing so temporarily reset the indexes for this action
        if fill_mean_subset in index_names:
            index_reset = True
            df.reset_index(inplace=True)

        #Check if subset variable is an index item
        for subset_item in df[fill_mean_subset].unique().tolist():
            for col in num_vars:
                subset_mean = df[df[fill_mean_subset] == subset_item][col].mean() 
                df.loc[(df[fill_mean_subset] == subset_item) & (df[col].isnull()), col] = subset_mean
        
        if index_reset:
            df.set_index(index_names, inplace=True)

    # For any remaining nulls, fill with the mean of the overall series
    for col in num_vars:
        df[col].fillna((df[col].mean()), inplace=True)
        
    # OHE the categorical variables
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    for var in  cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
    
    # Fill OHE NaNs with 0
    # Get list of columns after OHE that were not in the "numeric" list from earlier, using set function for speed.
    cat_vars = list(set(df.columns.tolist()) - set(num_vars.tolist()))
    for var in cat_vars:
        df[var].fillna(0, inplace=True)
    
    
    return df
    