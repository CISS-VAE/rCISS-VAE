#' Autotune CISS-VAE hyperparameters with Optuna
#'
#' @importFrom purrr keep
#'
#' @description Performs hyperparameter optimization for CISS-VAE using Optuna with
#'   support for both tunable and fixed parameters.
#'
#' @param data Data frame or matrix containing the input data
#' @param index_col String name of index column to preserve (optional)
#' @param clusters Integer vector specifying cluster assignments for each row.
#' @param save_model_path Optional path to save the best model's state_dict
#' @param save_search_space_path Optional path to save search space configuration
#' @param n_trials Number of Optuna trials to run
#' @param study_name Name identifier for the Optuna study
#' @param device_preference Preferred device ("cuda", "mps", "cpu")
#' @param show_progress Whether to display Rich progress bars during training
#' @param optuna_dashboard_db RDB storage URL/file for Optuna dashboard
#' @param load_if_exists Whether to load existing study from storage
#' @param seed Base random seed for reproducible results
#' @param verbose Whether to print detailed diagnostic information
#' @param constant_layer_size Whether all hidden layers use same dimension
#' @param evaluate_all_orders Whether to test all possible layer arrangements
#' @param max_exhaustive_orders Max arrangements to test when evaluate_all_orders = TRUE
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
#' @param val_proportion Proportion of non-missing data to hold out for validation.
#' @param replacement_value Numeric value used to replace missing entries before model input.
#' @param columns_ignore Character vector of column names to exclude from imputation scoring.
#' @param imputable_matrix Logical matrix indicating entries allowed to be imputed.
#' @param binary_feature_mask Logical vector marking which columns are binary.
#' @param weight_decay Weight decay (L2 penalty) used in Adam optimizer.
#' @param debug Logical; if TRUE, additional metadata is returned for debugging.
#'
#' @return List with imputed data, best model, study object, and results dataframe
#'
#' @section Tips:
#' \itemize{
#'   \item Use \code{cluster_on_missing()} or \code{cluster_on_missing_prop()} for cluster assignments.
#'   \item Use GPU computation when available; call \code{check_devices()} to see available devices.
#'   \item Adjust \code{batch_size} based on memory (larger is faster but uses more memory).
#'   \item Set \code{verbose = TRUE} or \code{show_progress = TRUE} to monitor training.
#'   \item Explore the \code{optuna-dashboard} (see vignette \code{optunadb}) for hyperparameter importance.
#'   \item For binary features, set \code{names(binary_feature_mask) <- colnames(data)}.
#' }
#'
#' @examples
#' \dontrun{
#' library(tidyverse)
#' library(reticulate)
#' library(rCISSVAE)
#'
#' reticulate::use_virtualenv("./cissvae_environment", required = TRUE)
#'
#' data(df_missing)
#' data(clusters)
#'
#' aut <- autotune_cissvae(
#'   data = df_missing,
#'   index_col = "index",
#'   clusters = clusters$clusters,
#'   n_trials = 3,
#'   study_name = "comprehensive_vae_autotune",
#'   device_preference = "cpu",
#'   seed = 42,
#'
#'   ## Hyperparameter search space
#'   num_hidden_layers = c(2, 5),
#'   hidden_dims = c(64, 512),
#'   latent_dim = c(10, 100),
#'   latent_shared = c(TRUE, FALSE),
#'   output_shared = c(TRUE, FALSE),
#'   lr = c(0.01, 0.1),
#'   decay_factor = c(0.99, 1.0),
#'   beta = c(0.01, 0.1),
#'   num_epochs = c(5, 20),
#'   batch_size = c(1000, 4000),
#'   num_shared_encode = c(0, 1, 2),
#'   num_shared_decode = c(0, 1, 2),
#'
#'   ## Placement strategies
#'   encoder_shared_placement = c(
#'     "at_end", "at_start",
#'     "alternating", "random"
#'   ),
#'   decoder_shared_placement = c(
#'     "at_start", "at_end",
#'     "alternating", "random"
#'   ),
#'
#'   refit_patience = 2,
#'   refit_loops = 10,
#'   epochs_per_loop = 5,
#'   reset_lr_refit = c(TRUE, FALSE)
#' )
#'
#' plot_vae_architecture(aut$model,
#'   title = "Optimized CISSVAE Architecture"
#' )
#' }
#'
#' @export

autotune_cissvae <- function(
  data,
  index_col              = NULL,
  val_proportion         = 0.1,
  replacement_value      = 0.0,
  columns_ignore         = NULL,
  imputable_matrix   = NULL,
  binary_feature_mask = NULL,
  clusters, ## cluster identities must be provided by user
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
  weight_decay = 0.001,
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
  reset_lr_refit    = c(TRUE, FALSE),
  ## defaults to returning helpful things
  debug = FALSE
) {

  requireNamespace('purrr')

  # -- 1) Coerce to integers ------------------------------------------------
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
  
  # -- 2) Validate shared layer placement strategies -----------------------
  valid_placements <- c("at_end", "at_start", "alternating", "random")
  if (!all(encoder_shared_placement %in% valid_placements)) {
    stop("Invalid encoder_shared_placement values. Must be one of: ", 
    paste(valid_placements, collapse = ", "))
  }
  if (!all(decoder_shared_placement %in% valid_placements)) {
    stop("Invalid decoder_shared_placement values. Must be one of: ", 
    paste(valid_placements, collapse = ", "))
  }
  
  # -- 3) Handle index_col -------------------------------------------------
  if (!is.null(index_col)) {
    if (!index_col %in% colnames(data)) stop("`index_col` not found in data.")
    index_vals <- data[[index_col]]
    data       <- data[, setdiff(colnames(data), 
      index_col), drop = FALSE]
    imputable_matrix = imputable_matrix[, setdiff(colnames(imputable_matrix), 
      index_col), drop = FALSE]
    ## handle index col in binary_feature_mask
    if(!is.null(binary_feature_mask) & !is.null(names(binary_feature_mask))){
      binary_feature_mask =  binary_feature_mask[ setdiff(names(binary_feature_mask), 
        index_col), drop = FALSE]
      if(debug){
        cat("Binary feature mask: ", 
        paste0(names(binary_feature_mask), collapse = ", "))
      }
    }
  } else index_vals <- NULL

  # -- 3.1) Ensure data and mask columns match EXACTLY (after index removal) --
if (!is.null(imputable_matrix)) {
  if (!identical(colnames(data), colnames(imputable_matrix))) {
    stop("`data` and `imputable_matrix` columns must match exactly (same names & order) after removing index_col.")
  }
}

# -- 3.2) Quick coverage diagnostics on FEATURE columns only --
feat_cols <- if (is.null(columns_ignore)) colnames(data) else setdiff(colnames(data), columns_ignore)

if (!is.null(imputable_matrix)) {
  obs_mat <- !is.na(as.matrix(data[, feat_cols, drop = FALSE]))
  imp_mat <- (as.matrix(imputable_matrix[, feat_cols, drop = FALSE]) == 1L)
  valid   <- obs_mat & imp_mat

  zero_cols <- feat_cols[colSums(valid) == 0]
  zero_rows <- which(rowSums(valid) == 0)

  if (length(zero_cols)) {
    stop("These feature columns have zero valid cells: ",
         paste(zero_cols, collapse = ", "),
         "\nDrop them or adjust the mask before running autotune.")
  }
  if (length(zero_rows)) {
    message("Rows with zero valid cells: ", length(zero_rows),
            " (they'll produce empty-loss batches unless handled in Python).")
  }
}


  # -- 4) Prepare matrix & Python imports ----------------------------------
  mat <- if (is.data.frame(data)) as.matrix(data) else data
  auto_mod <- reticulate::import("ciss_vae.training.autotune", convert = FALSE)
  SS       <- auto_mod$SearchSpace
  autotune <- auto_mod$autotune
  np       <- reticulate::import("numpy", convert = FALSE)
  pd       <- reticulate::import("pandas", convert = FALSE)
  CD_mod   <- reticulate::import("ciss_vae.classes.cluster_dataset", 
  convert = FALSE)$ClusterDataset
  
  # -- 5) Build Python SearchSpace -----------------------------------------
  ss_py <- SS(
    num_hidden_layers = reticulate::r_to_py(num_hidden_layers),
    hidden_dims       = reticulate::r_to_py(hidden_dims),
    latent_dim        = reticulate::r_to_py(latent_dim),
    latent_shared     = reticulate::r_to_py(latent_shared),
    output_shared     = reticulate::r_to_py(output_shared),
    lr                = reticulate::r_to_py(lr),
    decay_factor      = reticulate::r_to_py(decay_factor),
    beta              = beta,
    weight_decay = reticulate::r_to_py(weight_decay),
    num_epochs        = num_epochs,
    batch_size        = batch_size,
    num_shared_encode = reticulate::r_to_py(num_shared_encode),
    num_shared_decode = reticulate::r_to_py(num_shared_decode),
    # NEW: Add placement strategy parameters
    encoder_shared_placement = reticulate::r_to_py(encoder_shared_placement),
    decoder_shared_placement = reticulate::r_to_py(decoder_shared_placement),
    refit_patience    = refit_patience,
    refit_loops       = refit_loops,
    epochs_per_loop   = epochs_per_loop,
    reset_lr_refit    = reticulate::r_to_py(reset_lr_refit)
  )
  
  if (verbose) print("Built search space")
  
  # -- 6) Build ClusterDataset ----------------------------------------------
  ## Building the ClusterDataset is the point at which it will be necessary for the NAs to be good. 
  if (missing(clusters)) stop("`clusters` is required for autotune.")
  
  ## Fix for the sentinals -> make sure that, for whatever 'data' is, if it's NA the actual value in the data thing is 'NaN'
  data[is.na(data)] <- NaN
  data_py <- pd$DataFrame(data = data, dtype = "float64")

  # if(debug){
  #   print("Python data thing")
  #   print(data_py$head())
  # }
  clusters_py <- np$array(as.integer(clusters), dtype = "int64")

  # columns_ignore as a Python list (safer when convert = FALSE)
  cols_ignore_py <- if (is.null(columns_ignore)) NULL else reticulate::r_to_py(as.character(columns_ignore))

  if (!is.null(imputable_matrix)) {
      dni_py <- pd$DataFrame(imputable_matrix)
  } else dni_py <- NULL

  if(debug){
    if(!is.null(binary_feature_mask)){
      print(reticulate::r_to_py(binary_feature_mask))
      cat("Dim data_py = ", dim(data_py), "dim bfm = ", length(reticulate::r_to_py(binary_feature_mask)))
    }
  }

  ## correctly set up the cluster dataset
  train_ds_py <- CD_mod(
    data = data_py, 
    cluster_labels = clusters_py, 
    val_proportion = val_proportion, 
    replacement_value = replacement_value, 
    columns_ignore = cols_ignore_py, 
    imputable = dni_py,
    val_seed = seed,
    binary_feature_mask =  reticulate::r_to_py(binary_feature_mask))

  
  if(debug){
    print("DEBUG: BINARY FEATURE MASK")
    print(train_ds_py$binary_feature_mask)
  }
  
  if (verbose) print("Built cluster dataset")
  
  # -- 7) Assemble autotune args & run -------------------------------------
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
  ) |> keep(~ !is.null(.x))
  
  out_py      <- do.call(autotune, args_py)
  if (verbose) print("Ran autotune")
  
  out_list <-reticulate::py_to_r(out_py)    # now an R list of length 4
  best_imp_py <- out_list[[1]]
  best_mod_py <- out_list[[2]]
  study_py    <- out_list[[3]]
  results_py  <- out_list[[4]]
  
  # -- 8) Convert back to R -------------------------------------------------
  imp_df <- as.data.frame(best_imp_py, stringsAsFactors = FALSE)
  if (ncol(imp_df) == ncol(mat)) {
    colnames(imp_df) <- colnames(mat)
  }
  if (!is.null(rownames(mat)) && (nrow(imp_df) == nrow(mat))) {
    rownames(imp_df) <- rownames(mat)
  }
  
  
  if (!is.null(index_vals)) {
    imp_df[[index_col]] <- index_vals
    imp_df <- imp_df[, c(index_col, setdiff(names(imp_df), index_col))]
  }
  
  results_df <- as.data.frame(results_py, stringsAsFactors = FALSE)

  if(debug){
    print("At results")
  }
  
  out = list(
    imputed_dataset = imp_df,
    model   = best_mod_py,
    cluster_dataset = train_ds_py,
    clusters = clusters,
    study   = study_py,
    results = results_df
  )
  if(debug){
    print("made out list")
  }
  val_data = reticulate::py_to_r(
    out[["cluster_dataset"]]$val_data$detach()$cpu()$contiguous()$numpy()
  ) |>
    as.data.frame()

  val_imputed = reticulate::py_to_r(out$model$get_imputed_valdata(out$cluster_dataset)$detach()$cpu()$contiguous()$numpy()) |>
    as.data.frame()

  if (!is.null(out[["cluster_dataset"]]$feature_names)) {
    colnames(val_data) <- reticulate::py_to_r(out[["cluster_dataset"]]$feature_names)
    colnames(val_imputed) <- reticulate::py_to_r(out[["cluster_dataset"]]$feature_names)
  } 
  if (!is.null(index_col)){
    val_data[[index_col]] = index_vals
    val_data <- val_data[c(index_col, setdiff(names(val_data), index_col))]

    val_imputed[[index_col]] = index_vals
    val_imputed <- val_imputed[c(index_col, setdiff(names(val_imputed), index_col))]
  }
  # -----------------
  # If there were columns we wanted the model to ignore for validation, we want to keep them the same in the val_data  so we can filter by them for mse funct
  # ------------------
  if (!is.null(columns_ignore)){
    for(col in columns_ignore){
      val_data[[col]] = data[[col]]
      val_imputed[[col]] = data[[col]]
    }

  }

  out[["val_data"]] = val_data
  if(debug){
    print("added valdata")
  }
  out[["val_imputed"]] = val_imputed
  if(debug){
    print("added valimputed")
  }

  return(out)

}