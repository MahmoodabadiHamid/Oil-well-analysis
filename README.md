
# Oil Well Analysis #
The goal of this project is to find meaningful relationships
between oil wells features, in order to find similar wells and cluster them, which falls into the following phases:
1- Handel missing data 
2- Normalizing features values
3- Plot several figures such as Linear Correlation, Non-linear correlation, ... (available in "../figures/second dataset figures")
4- Finding the most similar oil wells to the targeted one
5- Finding the best number of clusters and clustering the data

##Prerequisites##

1. Python 3.x
2. Seaborn
3. Numpy
4. Matplotlib
5. Sklearn
6. Pandas

### Runnning the tests ###

This repository is just a part of bigger project and published for proving codding skills in a data science project.



### ***Functions definition*** ###
## preprocessing_and_descriptive_analysis.py ##

###normalizeValues(df)###
This function recieves a data frame and maps its columns values to range [0,1]

### plot_corr(df, size = 10) ###
This function calculates and plot linear correlation matrix

### featureCorrelationRanking(df) ###
This function returns a xlsx file including features and their correlations values

### PCA_(X, n) ###
This function gets X matrix and extracts its top n principle components using PCA algirithm.

### draw_pair_wise_scatter_plots(df) ###
 This function plots pair wise scatter figures of given df.
 
### number_of_optimal_k_means_classes(df) ###
This function draws a plot that shows how SSE error rate changes by changing number of clusters.

### k_means(df) ###
This function plots k-means algorithm results on reduced df dimensions.

### plotByLocation(df, label) ###
This function shows how wells features scattered in different locations.


## mutually_information_calculator.py ##

### main() ###
This is the main function which calculates non-linear relationships between features.


## vector_similarity.py ##

This file has one main function named "plot_k_most_similar_to_well"


### plot_k_most_similar_to_well(dataSet, columns, well_name, k) ###

This function finds top "k" wells that are most similar to "well_name"
based on given "dataset" and "columns".
The metrics to find similarity are Cosine, Manhattan, Euclidean













