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
import re
import geocoder
import operator



nb_path = os.getcwd()
nb_path
def sort_series_abs(S):
    'Takes a pandas Series object and returns the series sorted by absolute value'
    temp_df = pd.DataFrame(S)
    temp_df['abs'] = temp_df.iloc[:,0].abs()
    temp_df.sort_values('abs', ascending = False, inplace = True)
    return temp_df.iloc[:,0]
    
def clean_xls(file):
    df = pd.read_excel(file,skiprows=5, header = [0, 1, 2, 3], skipfooter = 6)
    unnamed_str = 'Unnamed: [0-9]+_level_[0-9]+'
    df.columns = [re.sub(unnamed_str, '', ' '.join(col)).strip() for col in df.columns.values]
    return df

def remove_aggregation(df):
    df.drop(df[df['CODE'].astype(str).map(len) <= 8].index, inplace=True)
    return df

def load_merge_clean(nb_path = os.getcwd()):
    # Create list of all files in CSV directory
    files = []
    for (dirpath, dirnames, filenames) in os.walk('{}\CSV'.format(nb_path)):
        files.extend(filenames)
        break
    for f in files:
        if f[-4:] != '.csv':
            files.remove(f)

    # Read in csv files and merge into 1 dataframe
    # Initialise dataframe with first csv file
    df = pd.read_csv('{}\CSV\{}'.format(nb_path, files[0]), na_values='-', thousands=',')
    # loop through the remainder of CSVs and load them in
    for file in range(1,len(files)):
        df_temp = pd.read_csv('{}\CSV\{}'.format(nb_path, files[file]), na_values='-', thousands=',')
        df_temp = remove_aggregation(df_temp)
        df = pd.merge(df, df_temp, how='inner', left_on=['CODE','YEAR','LABEL'], right_on=['CODE','YEAR','LABEL'])

    df.set_index(['CODE', 'LABEL', 'YEAR'], inplace=True)

    df = clean_data(df, 'CODE')
    
    df = df.xs(2016, level = 'YEAR')

    latlng = pd.read_csv('latlng.csv')

    df = pd.merge(df, latlng, how = 'left', left_on=['LABEL'], right_on = ['LABEL'])
    df = df.set_index(['LABEL'])

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
    
    # Drop empty columns
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
    
def buildlatlng(df):
    '''
    buildlatlng
    A function to build a database of latitudes and longitudes for the australian census regions
    Inputs - df: a dataframe with index.levels[1] of the names of all the suburbs
    Outputs - latlng: a dataframe with the corresponding latitudes and longitudes looked up from openstreetmaps
    '''
    addresses = df.index.levels[1].tolist()
    lat = []
    lng = []
    for address in addresses:
        #I found OSM struggles with the word region, so best to remove
        address = re.sub('Region', '', address)
        
        #Try a variety of different spellings of the word, check that they exist and are in Australia then append to the lat/long lists and continue to next cycle once one works
        try:
            callstr = address
            print(callstr)
            g = geocoder.osm(callstr)
            assert g.country == 'Australia'
            lat.append(g.latlng[0])
            lng.append(g.latlng[1])
            print(g)
            continue
        except:
            pass
        
        try:
            callstr = address.split('-', 1)[0]+' Australia'
            print(callstr)
            g = geocoder.osm(callstr)
            assert g.country == 'Australia'
            lat.append(g.latlng[0])
            lng.append(g.latlng[1])
            print(g)
            continue
        except:
            pass
        
        try:
            callstr = address.split(' ', 1)[0]+' Australia'
            print(callstr)
            g = geocoder.osm(callstr)
            assert g.country == 'Australia'
            lat.append(g.latlng[0])
            lng.append(g.latlng[1])
            print(g)
            continue
        except:
            pass
        
        try:
            callstr = address.split('-', 1)[0]
            print(callstr)
            g = geocoder.osm(callstr)
            assert g.country == 'Australia'
            lat.append(g.latlng[0])
            lng.append(g.latlng[1])
            print(g)
            continue
        except:
            pass
        
        
        lat.append(None)
        lng.append(None)

    #because of multiple spelling of Woolaware/Wooloware we have a single error to fix
    callstr = 'woolooware'
    print(callstr)
    g = geocoder.osm(callstr)
    lat[550] = g.latlng[0]
    lng[550] = g.latlng[1]

    latlng = pd.DataFrame({'lat':lat, 'long':lng}, index = X.index.levels[1].tolist())
    return latlng

def feature_impact_plot(model, X, n_features, y_label):
    '''
    Takes a trained model and training dataset and synthesises the impacts of the top n features
    to show their relationship to the response vector (i.e. how a change in the feature changes
    the prediction). Returns n plots showing the variance for min, max, median, 1Q and 3Q.
    
    INPUTS
    model = trained supervised learning model
    X_train = feature set the training was completed using
    n_features = top n features you would like to plot
    y_label - description of response variable for axis labelling
    '''
    # Display the n most important features
    indices = np.argsort(model.feature_importances_)[::-1]
    columns = X.columns.values[indices[:n_features]]
    
    ### THIS NEEDS UPDATING to integrate subplot mechanisms for line charts
    #simulations = [[]] #deprecated
    sim_var = [[]]
    # For top 5 features
    for col in columns:
        base_pred = model.predict(X)
        #add percentiles of base predictions to a df for use in reporting
        base_percentiles = [np.percentile(base_pred, pc) for pc in range(0,101,25)]
        #simulations.append(['base',col]+base_percentiles) #deprecated

        # Create new predictions based on tweaking the parameter
        # copy X, resetting values to align to the base information through different iterations
        df_copy = X.copy()

        for val in np.arange(-X[col].std(), X[col].std(), X[col].std()/50):
            df_copy[col] = X[col] + val
            # add new predictions based on changed database
            predictions = model.predict(df_copy)
            #add percentiles of these predictions to a df for use in reporting
            percentiles = [np.percentile(predictions, pc) for pc in range(0,101,25)]
            #simulations.append([val, col] + percentiles) #deprecated
            # add variances between percentiles of these predictions and the base prediction to a df for use in reporting
            percentiles = list(map(operator.sub, percentiles, base_percentiles))
            percentiles = list(map(operator.truediv, percentiles, base_percentiles))
            sim_var.append([val, col] + percentiles)

    # Plot a line chart based on the "describe()" function applied to this database
    # Showing percentiles (min, 25th, 50th, 75th, max) over the series of values
    df_predictions = pd.DataFrame(sim_var,columns = ['Value','Feature']+[0,25,50,75,100])
    num_cols = 2

    fig, axs = plt.subplots(nrows = (int(n_features/num_cols) + int(n_features%num_cols)),
                            ncols = num_cols, sharey = True, figsize=(15,15))


    for i in range(axs.shape[0]*axs.shape[1]):
        if i < len(columns):
            axs[int(i/num_cols),int(i%num_cols)].plot(df_predictions[df_predictions['Feature'] == columns[i]]['Value'],
                                    df_predictions[df_predictions['Feature'] == columns[i]][50])
            axs[int(i/num_cols),int(i%num_cols)].set_title("\n".join(wrap(columns[i], int(100/num_cols))))

            #format the y-axis as %
            if int(i%num_cols) == 0:
                vals = axs[int(i/num_cols),int(i%num_cols)].get_yticks()
                axs[int(i/num_cols),int(i%num_cols)].set_yticklabels(['{:,.2%}'.format(x) for x in vals])
                axs[int(i/num_cols),int(i%num_cols)].set_ylabel('% change to {}'.format(y_label))

        else:
            axs[int(i/num_cols),int(i%num_cols)].axis('off')

    plt.tight_layout()    
    plt.show()
