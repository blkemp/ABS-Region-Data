# ABS-Region-Data
Explore data compiled from the Australian Bureau of Statistics (ABS) compiled as part of their AUSSTATS documentation, looking at trends across a wide variety of "key measures" by geographical region within Australia.

Preliminarily I am interested in exploring ties between income and age (for potential use in later analysis on tax effects on different generations), but there are also some interesting factoids regarding solar panel installations which have piqued my curiousity.

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
