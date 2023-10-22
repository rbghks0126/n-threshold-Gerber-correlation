# n-threshold-Gerber-correlation

Extension of Gerber correlation statistic [(Gerber et al., 2022)](https://www.pm-research.com/content/iijpormgmt/48/3/87) by allowing an arbitrary number of *n* thresholds as well as incorporating the Exponentially Weighted Moving Average method.


## What's included:

### Code files
Contains the code for running backtests.
* gerber_backtest_template: Historical covariance, 1-threshold and 2-threshold Gerber covariance methods.
* EWMA_1t_gerber_backtest_template: 1-threshold Gerber covariance and 1-threshold EWMA Gerber covariance methods.
* EWMA_2t_gerber_backtest_template: 2-threshold Gerber covariance and 2-threshold EWMA Gerber covariance methods.
* tanhtanh simulation work.ipynb: Initial work for tanh-tanh continuously weighted Gerber correlation statistic. 
* gerber_utils.py: Contains helper functions for visualization and backtesting.

### data/
Contains relevant pre-processed asset returns datasets for the asset universes discussed in the thesis.
- gerber2021_9assets.parquet: Returns of 9 assets from Gerber's paper (with 2 replaced as mentioned in thesis).
- snp_22stocks.parquet: Returns of 22 stocks from SP500 with 2 stocks from each of the 11 industry sectors.
- snp_55stocks.parquet: Returns of 55 stocks from SP500 with 5 stocks from each of the 11 industry sectors.


### Thesis
- Thesis.pdf is the submitted thesis.