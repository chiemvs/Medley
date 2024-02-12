# Medley
Package concerning Mediterranean drought and drivers thereof.
Package can scrape and wrangle climate data from sources such as ERA5, NCEP, oceanic reanlyses, KNMI's climate explorer, EC-Earth simulations, and the ECA&D project.
The outcome of the wrangling are many timeseries of unequal length, structured into potential drivers on the one hand and several rainfall / drought indicators on the other hand.
Subsequently, the package provides a pipeline to statistically predict one of the rainfall/drought indicators based on lagged drivers.
Temporal sampling, several algorithms, predictor-selection and hyperparameter optimization can be used to fine tune results.
For interpretation we apply Visualization is used to inspect regional differences in rainfall trends, seasonality, skill of predictions.

## Code directories and purposes

* Scripts 
Scripts consisting of sequential function calls that need to run occasionally only, with quick workloads suitable for the login node.
\_retrieve-type scripts contain the scraping and first wrangling of data.
They can be used to slightly alter or reproduce the construction of timeseries and tend to write out files with intermediate data.
All the intermediate files on drivers (climexp,amoc,iod) are loaded and gathered in prepare\_monthly\_ts\_data.py, which merges them into a complete dataframe, and writes it to a final file.
E-obs, ECAD, WP3 are all for targets.
EC-Earth data are treated differently and consist of one source providing both drivers and target.

* hpc
Contains scripts with heavy workloads that are best submitted as jobs. Each python script is accompanied by a .slurm script. 
Compute_means.py belongs to the data-wrangling scripts and is used to obtain driver timeseries from ERA5 data (pre-downloaded and existing in the cluster's data_catalogue).  
hyperparams and predictor selection are for statistical prediction part.


* Notebooks


* Medley
Source code with functions and objects that are used in all of the above. Structured into modules 
    * 

