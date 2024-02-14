# Medley
Package concerning Mediterranean drought and drivers thereof.
Package can scrape and wrangle climate data from sources such as ERA5, NCEP, oceanic reanalyses, KNMI's climate explorer, EC-Earth simulations, and the ECA&D project.
The outcome of the wrangling are many timeseries of unequal length, structured into potential drivers on the one hand and several rainfall / drought indicators on the other hand.
Time resolution is generally monthly.
Subsequently, the package provides a pipeline to statistically predict one of the rainfall/drought indicators based on lagged drivers.
Temporal sampling, several algorithms, predictor-selection and hyperparameter optimization can be used to fine tune results.
For interpreting interrelations between drivers and drought we use causal discovery. 
Furthermore, visualization is used to inspect regional differences in rainfall trends, seasonality, skill of predictions, and the discovered causal graphs.

## Code directories and purposes

* Scripts:
Scripts consisting of sequential function calls that need to run occasionally only, with quick workloads suitable for the login node.
\_retrieve-type scripts contain the scraping and first wrangling of data.
They are used to obtain data, construct timeseries and tend to write out files with intermediate data.
The retrieve-type script of (climexp, amoc, iod, monthly-era5), write out intermediate files on drivers, that are later loaded and gathered in prepare\_monthly\_ts\_data.py, which merges them into a complete dataframe.
This dataframe is written to a final file (complete.parquet) and of relevance for all other parts of the code.
The retrieve-type scripts for E-OBS, ECA&D and WP3 are all data-wrangling scripts to construct rainfall and drought indicators, though contrary to the retrieval of drivers, these do not produce timeseries yet.
The drought indicators remain spatially explicit such that geographical patterns can be analyzed (see notebooks/Trend_exploration) and regions can be flexibly adapted when building a prediction pipeline (see e.g. Medley.dataloading.prep_ecad).
EC-Earth data are treated separately because they concern a single source of data on both drivers and target, which are not observational but climate model simulations under different AMOC states.
Also this process_ec_earth_data script writes out intermediate files.
This data is only used in one part of the Causal_analysis notebook.

* hpc:
Contains scripts with heavy workloads that are best submitted as jobs to cluster nodes. Each python script is accompanied by a .slurm script specifying the job. 
Compute_means.py belongs to the data-wrangling scripts and is used to obtain driver timeseries (notably zonal-wind-based jet indices). 
It does not download the daily ERA5 data (pre-downloaded and existing in the cluster's data_catalogue).  
Since the workloads run non-interactively in jobs they write out files. 
The hyperparams and predictor selection workloads are ways to tune the statistical prediction algorithms.
The files these two write out form socalled 'experiments', with a .json containing all information needed to reproduce it and a .csv with results.

* Notebooks:
Notebooks contain lightweight workloads for data exploration, visualization and for investigation of results from predictor selection experiments.
They are useful to produce and tweak graphs interactively, and together they contain the code for most of the graphs in the XAIDA deliverable.
Poster_plots.ipynb produces trend and seasonality plots based on drivers and on the ECA&D target.
Further trend visualization for the other drought/rainfall indicators (WP3, E-OBS) is in Trend_exploration.ipynb.
Causal_investigation.ipynb is largely a stand-alone notebook, loading in the dataframe with drivers and subsequently using the tigramite for causal discovery, it also presents a stand-alone analysis of the EC-earth data.
Temporal_exploration.ipynb is the oldest notebooks, concerning some initial data exploration of simultaneous and lagged relations between drivers and drought.

* Medley:
Source code with functions and objects that are used in all of the above. Structured into modules. Functions can together form an entire statistical modeling pipeline. (Parts of this pipeline are tested in scripts/test_estimators). These are the modules:
    * dataloading. Backbone that reads in timeseries dataframe of drivers, or construct a timeseries for an ECAD-based drought indicator. Less used is the loading of EC-earth timeseries.
    * preprocessing. This mostly concerns resampling/lagging of timeseries data, standardization or removing seasonality from timeseries (possibly in a crossvalidation pipeline).
    * crossval. Code copied mostly from https://zenodo.org/records/7967133 with sklearn-compatible objects to split dataset into train-validation folds.
    * estimators. Definitions for two custom statistical forecasting algorithms, that are exposed in combination with several standard estimators from sklearn and XGbooost. An Estimator is usually applied in combination with a crossvalidation.
    * interpretation. Functionality for two things: 1) read-in and investigate experiment files written out by predictor selection or hyperparameter optimization. 2) investigate the properties of fitted estimators through socalled Explainers. This latter XAI functionality is still rudimentary.
    * analysis. Functionality for computation of trends, also on spatially explicit array data. 
    * visualization. Functionality for plotting stationdata on maps 
    * utils. Various functions, also for transforming to SPI 

