library(reticulate)
library(rCISSVAE)

data(df_missing)
data(clusters)

dat = run_cissvae(
  data = df_missing,
  index_col = "index",
  val_proportion = 0.1, ## pass a vector for different proportions by cluster
  columns_ignore = c("Age", "Salary", "ZipCode10001", "ZipCode20002", "ZipCode30003"), ## If there are columns in addition to the index you want to ignore when selecting validation set, list them here. In this case, we ignore the 'demographic' columns because we do not want to remove data from them for validation purposes. 
  clusters = clusters$clusters, ## we have precomputed cluster labels so we pass them here
  epochs = 5,
  return_silhouettes = FALSE,
  return_history = TRUE,  # Get detailed training history
  verbose = FALSE,
  return_model = TRUE, ## Allows for plotting model schematic
  device = "cpu",  # Explicit device selection
  layer_order_enc = c("unshared", "shared", "unshared"),
  layer_order_dec = c("shared", "unshared", "shared"),
  return_validation_dataset = TRUE
)


dat$raw_data = NULL
dat$cluster_dataset = NULL

saveRDS(dat, file = "inst/extdata/demo_run.rds")


cluster_result <- cluster_on_missing(
  data = df_missing,
  cols_ignore =  c("index", "Age", "Salary", "ZipCode10001", "ZipCode20002", "ZipCode30003"),
  n_clusters = 4,  # Use KMeans with 4 clusters
  seed = 42
)

saveRDS(cluster_result, file = "inst/extdata/cluster_on_missing.rds")


## Standardize df_missing column names to feature_timepoint format
colnames(df_missing) = c('index', 'Age', 'Salary', 'ZipCode10001', 'ZipCode20002', 'ZipCode30003', 'Y1_1', 'Y1_2', 'Y1_3', 'Y1_4', 'Y1_5', 'Y2_1', 'Y2_2', 'Y2_3', 'Y2_4', 'Y2_5', 'Y3_1', 'Y3_2', 'Y3_3', 'Y3_4', 'Y3_5', 'Y4_1', 'Y4_2', 'Y4_3', 'Y4_4', 'Y4_5', 'Y5_1', 'Y5_2', 'Y5_3', 'Y5_4', 'Y5_5')

# Create and examine missingness proportion matrix
prop_matrix <- create_missingness_prop_matrix(df_missing, 
index_col = "index", 
cols_ignore = c('Age', 'Salary', 'ZipCode10001', 'ZipCode20002', 'ZipCode30003'),
repeat_feature_names = c("Y1", "Y2", "Y3", "Y4", "Y5"))


print("Missingness proportion matrix dimensions:")
print(dim(prop_matrix))
print("Sample of proportion matrix:")
print(head(prop_matrix[, 1:5]))

# Use proportion matrix with scaling for better clustering
advanced_result <- run_cissvae(
  data = df_missing,
  index_col = "index",
  clusters = NULL,  # Let function cluster using prop_matrix
  columns_ignore = c('Age', 'Salary', 'ZipCode10001', 'ZipCode20002', 'ZipCode30003'), 
  missingness_proportion_matrix = prop_matrix,
  scale_features = TRUE,  # Standardize features before clustering
  n_clusters = 4,
  leiden_resolution = 0.1,  
  epochs = 5,
  return_history = TRUE,
  return_silhouettes = TRUE,
  device = "cpu",
  verbose = FALSE,
  return_clusters = TRUE
)

saveRDS(advanced_result$training_history, file = "inst/extdata/missing_prop.rds")
