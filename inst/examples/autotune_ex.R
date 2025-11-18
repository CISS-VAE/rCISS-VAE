library(tidyverse)
library(reticulate)
library(rCISSVAE)
reticulate::use_virtualenv("./cissvae_environment", required = TRUE)

data(df_missing)
data(clusters)

aut <- autotune_cissvae(
  data = df_missing,
  index_col = "index",
  clusters = clusters$clusters,
  n_trials = 3, ## Using low number of trials for demo
  study_name = "comprehensive_vae_autotune",
  device_preference = "cpu",
  seed = 42, 
  
  ## Hyperparameter search space
  num_hidden_layers = c(2, 5),     # Try 2-5 hidden layers
  hidden_dims = c(64, 512),        # Layer sizes from 64 to 512
  latent_dim = c(10, 100),         # Latent dimension range
  latent_shared = c(TRUE, FALSE),
  output_shared = c(TRUE, FALSE),
  lr = 0.01,  # Learning rate range
  decay_factor = 0.99,
  beta = 0.01,  # KL weight range
  num_epochs = 5,                # Limited epochs for demo
  batch_size = 4000,     # Batch size options
  num_shared_encode = c(0, 1, 2, 3),
  num_shared_decode = c(0, 1, 2, 3),
  
  # Layer placement strategies - try different arrangements
  encoder_shared_placement = c("at_end", "at_start", "alternating", "random"),
  decoder_shared_placement = c("at_start", "at_end", "alternating", "random"),
  
  refit_patience = 2,        # Early stopping patience
  refit_loops = 10,                # Fixed refit loops, limited for demo
  epochs_per_loop = 5,   # Epochs per refit loop, limited for demo
  reset_lr_refit = c(TRUE, FALSE)
)

plot_vae_architecture(aut$model, title = "Optimized CISSVAE Architecture")