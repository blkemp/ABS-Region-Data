# ABS-Region-Data
Explore data compiled from the Australian Bureau of Statistics (ABS) compiled as part of their AUSSTATS documentation, looking at trends across a wide variety of "key measures" by geographical region within Australia.

Preliminarily I am interested in exploring some interesting factoids regarding solar panel installations which have piqued my curiousity. Though others may choose to build on top of this and investigate other interesting features.

Note all files were originally ".xls" format, and have had incredibly minor alterations as follows:
* Removing "branding" and "page header" rows (rows 1-5 in source document)
* Removing "Copyright" and empty "footer" columns at the end of the dataset
* Consolidating all the details outlined in the column headers into 1 header per column (removing merged formatting and cross filling for empty cells)
* Saving each document as .csv

All files are based on the ASGS (Australian Statistical Geography Standard) location methodology. For the purposes of exploration, the Latitude and Longitude of ABS statisical areas have been approximated using the geocoder library. There is a big oppotunity to leverage ABS region documentation to increase precision of these records, as well as build some mappign chart features from public shapefile records.

# Summary of findings
By applying machine learning techniques on the merged dataset, I find a fairly robust Random Forest model (r2 score of 0.745) which demostrates the main predictors of household installation of solar panels are largely inversely tied to how "urban" a given statistical region is, as reflected by: types of employment, population density, types of household buildings and distance of commute. I then look at growth trends of solar installations when splitting the regions quartiles defined by population density which show continued (and accelerating) high growth in solar installations in the 'non-urban' regions which has implications at the entreprenuerial, public sector and energy sector levels.

A more accessible write-up can be found in my Medium blog post published [here](https://medium.com/@brian.l.kemp/the-5-keys-to-a-great-solar-demographic-581d1b07a215).

# System Requirements:
All exploration done on this dataset is currently being undertaking using Jupyter Notebooks using Python 3.7 with the below modules (list will surely expand over time):
* NumPy
* Pandas
* MatPlotLib
* SKLearn
* os
* TextWrap
* re
* operator

# Files
* Anything under the /CSV/ directory - ABS data by region files coverted as outlined in the introduction.
* Anything under the /Images/ directory - Images of representative towns identified within the analysis, to be potentially used in the blog post.
* ABS Region Data.ipynb - the main Jupyter notebook I used in analysing the data. 
* latlng.csv - a file generated from geocoder used for mapping ABS region labels to their geographic coordinates.
* medians by code.csv - a pre-cleaned file with all ABS files merged which fills any "n/a" value (by column) with the median of the region given in the 'CODE' column. While it only contains values for 2016, medians were calculated on the basis of all years available (2011-2018).

# Copyright
Note all data sourced from the ABS is subject to copyright under Creative Commons Attribution 4.0 International, as outlined [here](https://www.abs.gov.au/copyright).
