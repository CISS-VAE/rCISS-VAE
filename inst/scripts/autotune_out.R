library(tidyverse)
library(reticulate)
library(rCISSVAE)

data(df_missing)
data(clusters)

aut <- autotune_cissvae(
  data = df_missing,
  index_col = "index",
  clusters = clusters$clusters,
  save_model_path = NULL,
  save_search_space_path = NULL,
  n_trials = 3, ## Using low number of trials for demo
  study_name = "comprehensive_vae_autotune",
  device_preference = "cpu",
  show_progress = FALSE,  # Rich progress bars with training visualization
  load_if_exists = TRUE,
  seed = 42, 
  verbose = FALSE,
  
  # Search strategy options
  constant_layer_size = FALSE,     # Allow different sizes per layer
  evaluate_all_orders = FALSE,     # Sample layer arrangements efficiently
  max_exhaustive_orders = 100,     # Limit for exhaustive search
  
  ## Hyperparameter search space
  num_hidden_layers = c(2, 5),     # Try 2-5 hidden layers
  hidden_dims = c(64, 512),        # Layer sizes from 64 to 512
  latent_dim = c(10, 100),         # Latent dimension range
  latent_shared = c(TRUE, FALSE),
  output_shared = c(TRUE, FALSE),
  lr = 0.01,  # Learning rate range
  decay_factor = 0.99,
  beta = 0.01,  # KL weight range
  num_epochs = 5,                # Fixed epochs for demo
  batch_size = c(1000, 4000),     # Batch size options
  num_shared_encode = c(0, 1, 2, 3),
  num_shared_decode = c(0, 1, 2, 3),
  
  # Layer placement strategies - try different arrangements
  encoder_shared_placement = c("at_end", "at_start", "alternating", "random"),
  decoder_shared_placement = c("at_start", "at_end", "alternating", "random"),
  
  refit_patience = 2,        # Early stopping patience
  refit_loops = 100,                # Fixed refit loops
  epochs_per_loop = 5,   # Epochs per refit loop
  reset_lr_refit = TRUE
)

imputed <- aut$imputed
best_model <- aut$model
study <- aut$study
results <- aut$results

saveRDS(aut, file = "inst/extdata/autotune.rds")

# cat("Best validation MSE:", study$best_value, "\n")
# print("Trial results:")
# results %>% kable() %>%
#   kable_styling(font_size=12)

# plot_vae_architecture(best_model, title = "Optimized CISSVAE Architecture", save_path = "autotune_vae_arch.png")