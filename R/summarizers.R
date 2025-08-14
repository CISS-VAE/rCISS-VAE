
## cluster_summary(data, clusters)
## Summarize data from each cluster (akin to table1 but by cluster not treatment). 
  #- Summarizes counts of categoricals, mean(sd) and/or min,med,max for continuous + proprotion of NAs

## performance_by_cluster(data, model, (other params as needed))
  # gets the de-normalized final model output and calcualtes the MSE for that vs the held-out validation data
  # gets the MSEs for any grouping (like MSE by color or MSE by race + sex)
  # uses group_by (from tidyverse)