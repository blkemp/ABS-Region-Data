# ABS-Region-Data Cycling to Work
Explore data compiled from the Australian Bureau of Statistics (ABS) compiled as part of their AUSSTATS documentation, looking at trends across a wide variety of "key measures" by geographical region within Australia.

A write-up of my results can be found on medium https://medium.com/@chriscarter_95766/cycle-commuting-in-australia-b82db3b9a3ab


I am using these files to predict which regions are more likely to cycle and the factors that drive this. I am motivated to do this as I love cycle commuting and would love to help encourage others to do the same.

Note all files were originally ".xls" format, and have had incredibly minor alterations as follows:
* Removing "branding" and "page header" rows (rows 1-5 in source document)
* Removing "Copyright" and empty "footer" columns at the end of the dataset
* Consolidating all the details outlined in the column headers into 1 header per column (removing merged formatting and cross filling for empty cells)
* Saving each document as .csv

All files are based on the ASGS (Australian Statistical Geography Standard) location methodology. There is potential of further exploration based on documents located here.

# System Requirements:
All exploration done on this dataset is currently being undertaking using Jupyter Notebooks using Python 3.7 with the below modules (list will surely expand over time):
* NumPy
* Pandas
* MatPlotLib
* SKLearn
* Seaborn
* os
* TextWrap
* eli5
* re
* geocoder
* bokeh

To install these files simply use !pip install <library> in your Jupyter notebook
    

# Key Files
The 2 main files required for this project are: 
* Chris Assignment.ipynb
Contains all my working as I try to fit the best model to my data, using R2 as a metric
* cleanfunc.py
This file aggregates the cleaning function that was used to pull disperate data sources together into a single file, as well as some of the evaluation and visualisation functions used thorughout the notebook

# Results
We were able to fit a model on the data with an R2 of ~0.57. This is a reasonable model to generate some insights from. The primary insights were that distance to work and jobs in media tended to have strong predictive power for whether people would cycle to work. There was a significant section of outliers in inner sydney that drove a lot of the results, so the model was split into a high and low variant to model the two different effects. 

# Future Improvements
I have done some work taking the log of the primary metric (% of people cycling to work). This work could be better flowed through the rest of the notebook as it had a strong impact on the R2 value.
Other work which would be interesting would be to build the multi-level random forest into a proper sklearn regressor class using the mixin class available. 

# Copyright
Note all data sourced from the ABS is subject to copyright under Creative Commons Attribution 4.0 International, as outlined [here](https://www.abs.gov.au/copyright).
