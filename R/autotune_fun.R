# rCISSVAE/R/autotune_fun.R

library(reticulate)
library(dplyr)

#' Autotune CISS-VAE hyperparameters with Optuna
#' ...
#' @param num_hidden_layers Numeric(2) vector: (min, max) for # hidden layers.
#' @param num_shared_encode Numeric vector: categorical # shared encoder layers.
#' @param num_shared_decode Numeric vector: categorical # shared decoder layers.
#' ...
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

  ## SearchSpace args
  num_hidden_layers = c(1, 4),
  hidden_dims       = c(64, 512),
  latent_dim        = c(10, 100),
  latent_shared     = c(TRUE, FALSE),
  output_shared     = c(TRUE, FALSE),
  lr                = c(1e-4, 1e-3),
  decay_factor      = c(0.9, 0.999),
  beta              = 0.01,
  num_epochs        = 10,
  batch_size        = 64,
  num_shared_encode = c(0, 1, 3),
  num_shared_decode = c(0, 1, 3),
  refit_patience    = 2,
  refit_loops       = 100,
  epochs_per_loop   = 10,
  reset_lr_refit    = c(TRUE, FALSE)
) {
  # ── 1) Coerce to integers ────────────────────────────────────────────────
  n_trials          <- as.integer(n_trials)
  seed              <- as.integer(seed)
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

  # ── 2) Validate & filter shared‐layer choices ───────────────────────────
  min_nhl <- min(num_hidden_layers)
  # encoder
  dropped_enc <- setdiff(num_shared_encode, num_shared_encode[num_shared_encode <= min_nhl])
  if (length(dropped_enc)) {
    warning("Dropping num_shared_encode > min(num_hidden_layers): ", paste(dropped_enc, collapse = ", "))
    num_shared_encode <- num_shared_encode[num_shared_encode <= min_nhl]
  }
  # decoder
  dropped_dec <- setdiff(num_shared_decode, num_shared_decode[num_shared_decode <= min_nhl])
  if (length(dropped_dec)) {
    warning("Dropping num_shared_decode > min(num_hidden_layers): ", paste(dropped_dec, collapse = ", "))
    num_shared_decode <- num_shared_decode[num_shared_decode <= min_nhl]
  }
  if (length(num_shared_encode)==0) stop("No valid num_shared_encode ≤ min(num_hidden_layers).")
  if (length(num_shared_decode)==0) stop("No valid num_shared_decode ≤ min(num_hidden_layers).")

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
    refit_patience    = refit_patience,
    refit_loops       = refit_loops,
    epochs_per_loop   = epochs_per_loop,
    reset_lr_refit    = r_to_py(reset_lr_refit)
  )

  if (verbose) print("Built search space")

  # ── 6) Build ClusterDataset ──────────────────────────────────────────────
  ## make sure that if missing clusters, all clusters is 0
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
    show_progress          = show_progress,
    optuna_dashboard_db    = optuna_dashboard_db,
    load_if_exists         = load_if_exists,
    seed                   = seed,
    verbose                = verbose
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
