
# Oil Well Analysis #
The goal of doing this project is to find meaningful relationships
between oil wells features, in order to find similar wells and cluster them, which falls into the following phases:
1- Handel missing data 
2- Normalizing features values
3- plot several figures such as Linear Correlation, Non-linear correlation, ... (available in "../figures/second dataset figures")
4- finding the most similar oil wells to the targeted one
5- finding the best number of clusters and clustering

##Prerequisites##

1. Python 3.x
2. Seaborn
3. Numpy
4. Matplotlib
5. Sklearn
6. Pandas

### Runnning the tests ###

This repository is just a part of bidder project and published for proving codding skills in a data science project.



### ***Functions definition*** ###
## preprocessing_and_descriptive_analysis.py ##

###normalizeValues(df)###
This function recieve a data frame and map its columns values to range [0,1]

### plot_corr(df, size = 10) ###
This function calculate and plot linear correlation matrix

### featureCorrelationRanking(df) ###
This function return a xlsx file include features and their correlations values

### PCA_(X, n) ###
This function get X matrix and extract its top n principle components using PCA algirithm.

### draw_pair_wise_scatter_plots(df) ###
 This function plot pair wise scatter plots of given df.
 
### number_of_optimal_k_means_classes(df) ###
This function draw a plot that shows how SSE error rate changed by changing number of clusters.

### k_means(df) ###
This function plots k-means algorithm result on reduced df dimensions.

### plotByLocation(df, label) ###
The goal of this function is show how wells features scattered in different 
locations.


## preprocessing_and_descriptive_analysis.py ##

### main() ###
Main function calls other functions existed in this file and calculate 
non-linear relationship between features.



## vector_similarity.py ##

This file has one main function named "plot_k_most_similar_to_well"


### plot_k_most_similar_to_well(dataSet, columns, well_name, k) ###

This function finds top "k" wells that are most similar to "well_name"
based on given "dataset" and "columns".
The metrics to find similarity are Cosine, Manhattan, Euclidean













