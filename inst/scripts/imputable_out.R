library(rCISSVAE)

data(dni, mock_surv)

res <- run_cissvae(
  mock_surv,
  index_col = "patient_id",
  columns_ignore = c("death_event","death_year"),
  imputable_matrix = dni,   
  val_proportion = 0.3,
  return_clusters =FALSE,
  return_history = FALSE,
  epochs = 100,
  leiden_resolution = 0.01,
  k_neighbors = 5,
  return_silhouettes = FALSE
)

saveRDS(res, file = "inst/extdata/imputable.rds")
