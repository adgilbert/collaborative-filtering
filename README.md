# Sparse Factor Analysis Collaborative Filtering

This is the directory for the CS229 project by Andy Gilbert and Andrew Hilger. 

# Summary

This project implements a sparse factor analysis collaborative filtering algorithm for building a recommendation system. 

In principle only two files need to be run. RecommenderSystem.py performs business and user clustering and saves the tourist and local dataset for a given metro area. Train.py performs the training using the sparse factor analysis and also tests at each iteration and exports testing and training results for analysis. 

PlotResults.ipynb can be used to view those results.

For other files uses see the description below.   

## Python Files:
 - canny_cf.py: contains the functions for the Sparse Factor Analysis Collaborative Filtering.
 - convert_json_to_csv.py: contains function to convery yelp json files to a csv format.
 - get_city_proportion.py: contains code to calculate feautures for users and businesses.
 - get_metro_features.py: contains additional functions to calculate features for users and businesses. 
 - plot_and_split.py: Contains code to plot clusters onto a world, EU, or US map. Also contains some splitting code for the business dataset.
 - RecomenderSystem.py: An export of RecommenderSystem.ipynb, used to run the code on the server. The code goes through the clustering and splitting of the dataset into local and tourist versions and saves the split versions for a given metro area.
 - RunCanny.py: Used to test the cann_cf.py code
 - Train.py: Used to perform the Training on the tourist and local datasets using sparse factor analysis

 ## Jupyter Notebooks:
 - PlotResults.ipynb: Notebook used to visualize the results of training.
 - RecommenderSystem.ipynb: Notebook used to perform data cleaning, feature analysis and extraction, and clustering. Ultimately, exports tourist and local split datasaets

 ## Other
 - map/: contains map data for US state map.
 - requirements.txt: the necessary python packages for the project. 
