# rCISSVAE/R/autotune_fun.R
library(reticulate)
library(dplyr)
#' Autotune CISS-VAE hyperparameters with Optuna
#' 
#' Performs hyperparameter optimization for CISS-VAE using Optuna, with support for
#' Rich progress bars and flexible layer arrangement strategies.
#' 
#' @param data Data frame or matrix containing the input data
#' @param index_col String name of index column to preserve (optional)
#' @param clusters Integer vector specifying cluster assignments for each row
#' @param save_model_path Optional path to save the best model's state_dict
#' @param save_search_space_path Optional path to save search space configuration  
#' @param n_trials Number of Optuna trials to run
#' @param study_name Name identifier for the Optuna study
#' @param device_preference Preferred device ("cuda" or "cpu")
#' @param show_progress Whether to display Rich progress bars during training
#' @param optuna_dashboard_db RDB storage URL/file for Optuna dashboard
#' @param load_if_exists Whether to load existing study from storage
#' @param seed Base random seed for reproducible results
#' @param verbose Whether to print detailed diagnostic information
#' @param constant_layer_size Whether all hidden layers use same dimension
#' @param evaluate_all_orders Whether to test all possible layer arrangements
#' @param max_exhaustive_orders Max arrangements to test when evaluate_all_orders=TRUE
#' @param num_hidden_layers Numeric(2) vector: (min, max) for number of hidden layers
#' @param hidden_dims Numeric vector: hidden layer dimensions to test
#' @param latent_dim Numeric(2) vector: (min, max) for latent dimension
#' @param latent_shared Logical vector: whether latent space is shared across clusters
#' @param output_shared Logical vector: whether output layer is shared across clusters
#' @param lr Numeric(2) vector: (min, max) learning rate range
#' @param decay_factor Numeric(2) vector: (min, max) LR decay factor range
#' @param beta Numeric: KL divergence weight (fixed or range)
#' @param num_epochs Integer: number of initial training epochs (fixed or range)
#' @param batch_size Integer: mini-batch size (fixed or range)
#' @param num_shared_encode Numeric vector: numbers of shared encoder layers to test
#' @param num_shared_decode Numeric vector: numbers of shared decoder layers to test
#' @param encoder_shared_placement Character vector: placement strategies for encoder shared layers
#' @param decoder_shared_placement Character vector: placement strategies for decoder shared layers
#' @param refit_patience Integer: early stopping patience for refit loops
#' @param refit_loops Integer: maximum number of refit loops
#' @param epochs_per_loop Integer: epochs per refit loop
#' @param reset_lr_refit Logical vector: whether to reset LR before refit
#' 
#' @return List containing imputed data, best model, study object, and results dataframe
#' @export
autotune_cissvae <- function(
  data,
  index_col              = NULL,
  clusters,
  save_model_path        = NULL,
  save_search_space_path = NULL,
  n_trials               = 20,
  study_name             = "vae_autotune",
  device_preference      = "cuda",
  show_progress          = FALSE,
  optuna_dashboard_db    = NULL,
  load_if_exists         = TRUE,
  seed                   = 42,
  verbose                = FALSE,
  constant_layer_size    = FALSE,
  evaluate_all_orders    = FALSE,
  max_exhaustive_orders  = 100,
  ## SearchSpace args - UPDATED with new parameters
  num_hidden_layers = c(1, 4),
  hidden_dims       = c(64, 512),
  latent_dim        = c(10, 100),
  latent_shared     = c(TRUE, FALSE),
  output_shared     = c(TRUE, FALSE),
  lr                = c(1e-4, 1e-3),
  decay_factor      = c(0.9, 0.999),
  beta              = 0.01,
  num_epochs        = 500,
  batch_size        = 4000,
  num_shared_encode = c(0, 1, 3),
  num_shared_decode = c(0, 1, 3),
  # NEW: Shared layer placement strategies
  encoder_shared_placement = c("at_end", "at_start", "alternating", "random"),
  decoder_shared_placement = c("at_start", "at_end", "alternating", "random"),
  refit_patience    = 2,
  refit_loops       = 100,
  epochs_per_loop   = 500,
  reset_lr_refit    = c(TRUE, FALSE)
) {
  # ── 1) Coerce to integers ────────────────────────────────────────────────
  n_trials          <- as.integer(n_trials)
  seed              <- as.integer(seed)
  max_exhaustive_orders <- as.integer(max_exhaustive_orders)
  num_hidden_layers <- as.integer(num_hidden_layers)
  hidden_dims       <- as.integer(hidden_dims)
  latent_dim        <- as.integer(latent_dim)
  num_epochs        <- as.integer(num_epochs)
  batch_size        <- as.integer(batch_size)
  num_shared_encode <- as.integer(num_shared_encode)
  num_shared_decode <- as.integer(num_shared_decode)
  refit_patience    <- as.integer(refit_patience)
  refit_loops       <- as.integer(refit_loops)
  epochs_per_loop   <- as.integer(epochs_per_loop)
  
  # ── 2) Validate shared layer placement strategies ───────────────────────
  valid_placements <- c("at_end", "at_start", "alternating", "random")
  if (!all(encoder_shared_placement %in% valid_placements)) {
    stop("Invalid encoder_shared_placement values. Must be one of: ", paste(valid_placements, collapse = ", "))
  }
  if (!all(decoder_shared_placement %in% valid_placements)) {
    stop("Invalid decoder_shared_placement values. Must be one of: ", paste(valid_placements, collapse = ", "))
  }
  
  # ── 3) Handle index_col ─────────────────────────────────────────────────
  if (!is.null(index_col)) {
    if (!index_col %in% colnames(data)) stop("`index_col` not found in data.")
    index_vals <- data[[index_col]]
    data       <- data[, setdiff(colnames(data), index_col), drop = FALSE]
  } else index_vals <- NULL
  
  # ── 4) Prepare matrix & Python imports ──────────────────────────────────
  mat <- if (is.data.frame(data)) as.matrix(data) else data
  auto_mod <- import("ciss_vae.training.autotune", convert = FALSE)
  SS       <- auto_mod$SearchSpace
  autotune <- auto_mod$autotune
  np       <- import("numpy", convert = FALSE)
  CD_mod   <- import("ciss_vae.classes.cluster_dataset", convert = FALSE)$ClusterDataset
  
  # ── 5) Build Python SearchSpace ─────────────────────────────────────────
  ss_py <- SS(
    num_hidden_layers = r_to_py(num_hidden_layers),
    hidden_dims       = r_to_py(hidden_dims),
    latent_dim        = r_to_py(latent_dim),
    latent_shared     = r_to_py(latent_shared),
    output_shared     = r_to_py(output_shared),
    lr                = r_to_py(lr),
    decay_factor      = r_to_py(decay_factor),
    beta              = beta,
    num_epochs        = num_epochs,
    batch_size        = batch_size,
    num_shared_encode = r_to_py(num_shared_encode),
    num_shared_decode = r_to_py(num_shared_decode),
    # NEW: Add placement strategy parameters
    encoder_shared_placement = r_to_py(encoder_shared_placement),
    decoder_shared_placement = r_to_py(decoder_shared_placement),
    refit_patience    = refit_patience,
    refit_loops       = refit_loops,
    epochs_per_loop   = epochs_per_loop,
    reset_lr_refit    = r_to_py(reset_lr_refit)
  )
  
  if (verbose) print("Built search space")
  
  # ── 6) Build ClusterDataset ──────────────────────────────────────────────
  if (missing(clusters)) stop("`clusters` is required for autotune.")
  data_py     <- np$array(mat)
  clusters_py <- np$array(as.integer(clusters))
  train_ds_py <- CD_mod(data_py, clusters_py)
  
  if (verbose) print("Built cluster dataset")
  
  # ── 7) Assemble autotune args & run ─────────────────────────────────────
  args_py <- list(
    search_space           = ss_py,
    train_dataset          = train_ds_py,
    save_model_path        = save_model_path,
    save_search_space_path = save_search_space_path,
    n_trials               = n_trials,
    study_name             = study_name,
    device_preference      = device_preference,
    show_progress          = show_progress,  # Now properly supported
    optuna_dashboard_db    = optuna_dashboard_db,
    load_if_exists         = load_if_exists,
    seed                   = seed,
    verbose                = verbose,
    # NEW: Add new parameters
    constant_layer_size    = constant_layer_size,
    evaluate_all_orders    = evaluate_all_orders,
    max_exhaustive_orders  = max_exhaustive_orders
  ) %>% keep(~ !is.null(.x))
  
  out_py      <- do.call(autotune, args_py)
  if (verbose) print("Ran autotune")
  
  out_list <- py_to_r(out_py)    # now an R list of length 4
  best_imp_py <- out_list[[1]]
  best_mod_py <- out_list[[2]]
  study_py    <- out_list[[3]]
  results_py  <- out_list[[4]]
  
  # ── 8) Convert back to R ─────────────────────────────────────────────────
  imp_df <- as.data.frame(best_imp_py, stringsAsFactors = FALSE)
  colnames(imp_df) <- colnames(mat)
  rownames(imp_df) <- rownames(mat)
  
  if (!is.null(index_vals)) {
    imp_df[[index_col]] <- index_vals
    imp_df <- imp_df[, c(index_col, setdiff(names(imp_df), index_col))]
  }
  
  results_df <- as.data.frame(results_py, stringsAsFactors = FALSE)
  
  list(
    imputed = imp_df,
    model   = best_mod_py,
    study   = study_py,
    results = results_df
  )
}