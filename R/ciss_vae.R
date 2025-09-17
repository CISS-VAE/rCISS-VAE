#' Run the CISS-VAE pipeline for missing data imputation
#'
#' @description
#' This function wraps the Python `run_cissvae` function from the `ciss_vae` module,
#' providing a complete pipeline for missing data imputation using a Cluster-Informed
#' Shared and Specific Variational Autoencoder (CISS-VAE). The function handles data
#' preprocessing, model training, and returns imputed data along with optional
#' model artifacts.
#'
#' The CISS-VAE architecture uses cluster information to learn both shared and
#' cluster-specific representations, enabling more accurate imputation by leveraging
#' patterns within and across different data subgroups.
#'
#' @param data A data.frame or matrix (samples × features) containing the data to impute.
#'   May contain `NA` values which will be imputed.
#' @param index_col Character. Name of column in `data` to treat as sample identifier.
#'   This column will be removed before training and re-attached to results. Default `NULL`.
#' @param val_proportion Numeric. Fraction of non-missing entries to hold out for
#'   validation during training. Must be between 0 and 1. Default `0.1`.
#' @param replacement_value Numeric. Fill value for masked entries during training.
#'   Default `0.0`.
#' @param columns_ignore Character or integer vector. Columns to exclude from validation set.
#'   Can specify by name or index. Default `NULL`.
#' @param print_dataset Logical. If `TRUE`, prints dataset summary information during
#'   processing. Default `TRUE`.
#' @param clusters Optional vector or single-column data.frame of precomputed cluster
#'   labels for samples. If `NULL`, clustering will be performed automatically. Default `NULL`.
#' @param n_clusters Integer. Number of clusters for KMeans clustering when `clusters`
#'   is `NULL`. Number of clusters for KMeans clustering when 'clusters' is NULL. If `NULL`, 
#'   will use [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) for clustering.  
#'   Default `NULL`.
#' @param cluster_selection_epsilon Numeric. Epsilon parameter for HDBSCAN clustering
#'   when automatic clustering is used. Default `0.25`.
#' @param seed Integer. Random seed for reproducible results. Default `42`.
#' @param missingness_proportion_matrix Optional pre-computed missingness proportion
#'   matrix for biomarker-based clustering. If provided, clustering will be based on
#'   these proportions. Default `NULL`.
#' @param scale_features Logical. Whether to scale features when using missingness
#'   proportion matrix clustering. Default `FALSE`.
#' @param hidden_dims Integer vector. Sizes of hidden layers in encoder/decoder.
#'   Length determines number of hidden layers. Default `c(150, 120, 60)`.
#' @param latent_dim Integer. Dimension of latent space representation. Default `15`.
#' @param layer_order_enc Character vector. Sharing pattern for encoder layers.
#'   Each element should be "shared" or "unshared". Length must match `length(hidden_dims)`.
#'   Default `c("unshared", "unshared", "unshared")`.
#' @param layer_order_dec Character vector. Sharing pattern for decoder layers.
#'   Each element should be "shared" or "unshared". Length must match `length(hidden_dims)`.
#'   Default `c("shared", "shared", "shared")`.
#' @param latent_shared Logical. Whether latent space weights are shared across clusters.
#'   Default `FALSE`.
#' @param output_shared Logical. Whether output layer weights are shared across clusters.
#'   Default `FALSE`.
#' @param batch_size Integer. Mini-batch size for training. Larger values may improve
#'   training stability but require more memory. Default `4000`.
#' @param return_model Logical. If `TRUE`, returns the trained Python VAE model object.
#'   Default `TRUE`.
#' @param epochs Integer. Number of epochs for initial training phase. Default `500`.
#' @param initial_lr Numeric. Initial learning rate for optimizer. Default `0.01`.
#' @param decay_factor Numeric. Exponential decay factor for learning rate scheduling.
#'   Must be between 0 and 1. Default `0.999`.
#' @param beta Numeric. Weight for KL divergence term in VAE loss function.
#'   Controls regularization strength. Default `0.001`.
#' @param device Character. Device specification for computation ("cpu" or "cuda").
#'   If `NULL`, automatically selects best available device. Default `NULL`.
#' @param max_loops Integer. Maximum number of impute-refit loops to perform.
#'   Default `100`.
#' @param patience Integer. Early stopping patience for refit loops. Training stops
#'   if validation loss doesn't improve for this many consecutive loops. Default `2`.
#' @param epochs_per_loop Integer. Number of epochs per refit loop. If `NULL`,
#'   uses same value as `epochs`. Default `NULL`.
#' @param initial_lr_refit Numeric. Learning rate for refit loops. If `NULL`,
#'   uses same value as `initial_lr`. Default `NULL`.
#' @param decay_factor_refit Numeric. Decay factor for refit loops. If `NULL`,
#'   uses same value as `decay_factor`. Default `NULL`.
#' @param beta_refit Numeric. KL weight for refit loops. If `NULL`,
#'   uses same value as `beta`. Default `NULL`.
#' @param verbose Logical. If `TRUE`, prints detailed progress information during
#'   training. Default `FALSE`.
#' @param return_silhouettes Logical. If `TRUE`, returns silhouette scores for
#'   cluster quality assessment. Default `FALSE`.
#' @param return_history Logical. If `TRUE`, returns training history as a data.frame
#'   containing loss values and metrics over epochs. Default `FALSE`.
#' @param return_dataset Logical. If `TRUE`, returns the ClusterDataset object used
#'   during training (contains validation data, masks, etc.). Default `FALSE`.
#'
#' @details
#' The CISS-VAE method works in two main phases:
#' 
#' 1. **Initial Training**: The model is trained on the original data with validation
#'    holdout to learn initial representations and imputation patterns.
#' 
#' 2. **Impute-Refit Loops**: The model iteratively imputes missing values and
#'    retrains on the updated dataset until convergence or maximum loops reached.
#' 
#' The architecture uses both shared and cluster-specific layers to capture:
#' - **Shared patterns**: Common relationships across all clusters
#' - **Specific patterns**: Unique relationships within each cluster
#'
#' @return A list containing imputed data and optional additional outputs:
#' \describe{
#'   \item{imputed}{data.frame of imputed data with same dimensions as input.
#'     Missing values are filled with model predictions. If `index_col` was
#'     provided, it is re-attached as the first column.}
#'   \item{model}{(if `return_model=TRUE`) Python CISSVAE model object.
#'     Can be used for further analysis or predictions.}
#'   \item{dataset}{(if `return_dataset=TRUE`) Python ClusterDataset object
#'     containing validation data, masks, normalization parameters, and cluster labels.
#'     Can be used with performance_by_cluster() and other analysis functions.}
#'   \item{silhouettes}{(if `return_silhouettes=TRUE`) Numeric silhouette
#'     score measuring cluster separation quality.}
#'   \item{history}{(if `return_history=TRUE`) data.frame containing training
#'     history with columns for epoch, losses, and validation metrics.}
#' }
#'
#' @section Requirements:
#' This function requires the Python `ciss_vae` package to be installed and
#' accessible via `reticulate`. The package handles automatic device selection
#' (CPU/GPU) based on availability.
#'
#' @section Performance Tips:
#' \itemize{
#'   \item Use GPU computation when available for faster training on large datasets
#'   \item Adjust `batch_size` based on available memory (larger = faster but more memory)
#'   \item Start with default hyperparameters and adjust based on validation performance
#'   \item Use `verbose=TRUE` to monitor training progress on large datasets
#' }
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Basic usage with automatic clustering
#' result <- run_cissvae(
#'   data = my_data_with_missing,
#'   index_col = "sample_id"
#' )
#' imputed_data <- result$imputed
#' 
#' # Advanced usage with dataset for performance analysis
#' result <- run_cissvae(
#'   data = my_data,
#'   clusters = my_cluster_labels,
#'   hidden_dims = c(200, 150, 100),
#'   latent_dim = 20,
#'   epochs = 1000,
#'   return_history = TRUE,
#'   return_silhouettes = TRUE,
#'   return_dataset = TRUE,
#'   verbose = TRUE
#' )
#' 
#' # Access different outputs
#' imputed_data <- result$imputed
#' training_history <- result$history
#' cluster_quality <- result$silhouettes
#' 
#' # Use dataset for performance analysis
#' perf <- performance_by_cluster(
#'   original_data = my_data,
#'   model = result$model,
#'   dataset = result$dataset,
#'   clusters = my_cluster_labels
#' )
#' 
#' # Using pre-computed missingness matrix for clustering
#' prop_matrix <- create_missingness_prop_matrix(
#'   data = my_data, 
#'   index_col = "sample_id"
#' )
#' result <- run_cissvae(
#'   data = my_data,
#'   index_col = "sample_id",
#'   missingness_proportion_matrix = prop_matrix,
#'   scale_features = TRUE,
#'   return_dataset = TRUE
#' )
#' 
#' # Custom layer sharing patterns
#' result <- run_cissvae(
#'   data = my_data,
#'   hidden_dims = c(100, 80, 60),
#'   layer_order_enc = c("unshared", "shared", "shared"),
#'   layer_order_dec = c("shared", "shared", "unshared"),
#'   latent_shared = TRUE
#' )
#' }
#'
#' @seealso
#' \code{\link{create_missingness_prop_matrix}} for creating missingness proportion matrices
#' \code{\link{performance_by_cluster}} for analyzing model performance using the returned dataset
#' 
run_cissvae <- function(
  data,
  index_col              = NULL,
  val_proportion         = 0.1,
  replacement_value      = 0.0,
  columns_ignore         = NULL,
  print_dataset          = TRUE,
  clusters               = NULL,
  n_clusters             = NULL,
  seed                   = 42,
  missingness_proportion_matrix = NULL,
  scale_features         = FALSE,
  hidden_dims            = c(150, 120, 60),
  latent_dim             = 15,
  layer_order_enc        = c("unshared", "unshared", "unshared"),
  layer_order_dec        = c("shared", "shared", "shared"),
  latent_shared          = FALSE,
  output_shared          = FALSE,
  batch_size             = 4000,
  return_model           = TRUE,
  epochs                 = 500,
  initial_lr             = 0.01,
  decay_factor           = 0.999,
  beta                   = 0.001,
  device                 = NULL,
  max_loops              = 100,
  patience               = 2,
  epochs_per_loop        = NULL,
  initial_lr_refit       = NULL,
  decay_factor_refit     = NULL,
  beta_refit             = NULL,
  verbose                = FALSE,
  return_clusters        = FALSE,     # NEW
  return_silhouettes     = FALSE,
  return_history         = FALSE,
  return_dataset         = FALSE,
  ## NEw stuff from the ptyhon update
  do_not_impute_matrix   = NULL,
  k_neighbors            = 15L,
  leiden_resolution      = 0.5,
  leiden_objective       = c("CPM", "RB", "Modularity"),
  debug                  = FALSE
) {
  leiden_objective <- match.arg(leiden_objective)

  # ---------- helpers ----------
  `%||%` <- function(a, b) if (is.null(a)) b else a
  is_py_obj <- function(x) inherits(x, "python.builtin.object")
  accepts_arg <- function(py_callable, name) {
    inspect <- reticulate::import("inspect", convert = FALSE)
    ok <- FALSE
    try({
      sig <- inspect$signature(py_callable)
      ok <- !is.null(sig$parameters$get(name))
    }, silent = TRUE)
    ok
  }
  to_r_df_safely <- function(x) {
    if (is.data.frame(x)) return(x)
    if (inherits(x, "python.builtin.object")) {
      if (reticulate::py_has_attr(x, "to_dict")) {
        out <- tryCatch(reticulate::py_to_r(x), error = function(e) NULL)
        if (is.data.frame(out)) return(out)
        if (is.list(out) && length(out) > 0L && length(unique(lengths(out))) == 1L)
          return(as.data.frame(out, stringsAsFactors = FALSE))
      }
      if (reticulate::py_has_attr(x, "shape")) {
        arr <- tryCatch(reticulate::py_to_r(x), error = function(e) NULL)
        if (is.matrix(arr)) return(as.data.frame(arr, stringsAsFactors = FALSE))
        if (is.vector(arr)) return(data.frame(arr, check.names = FALSE))
      }
      out <- tryCatch(reticulate::py_to_r(x), error = function(e) NULL)
      if (is.data.frame(out)) return(out)
      if (is.list(out) && length(out) > 0L && length(unique(lengths(out))) == 1L)
        return(as.data.frame(out, stringsAsFactors = FALSE))
      return(out)
    }
    if (is.list(x) && length(x) > 0L && length(unique(lengths(x))) == 1L)
      return(as.data.frame(x, stringsAsFactors = FALSE))
    x
  }

  # ---------- coerce numerics/ints ----------
  seed            <- as.integer(seed)
  latent_dim      <- as.integer(latent_dim)
  batch_size      <- as.integer(batch_size)
  epochs          <- as.integer(epochs)
  max_loops       <- as.integer(max_loops)
  patience        <- as.integer(patience)
  n_clusters      <- if (!is.null(n_clusters)) as.integer(n_clusters) else NULL
  epochs_per_loop <- if (!is.null(epochs_per_loop)) as.integer(epochs_per_loop) else NULL
  hidden_dims     <- as.integer(hidden_dims)
  layer_order_enc <- as.character(layer_order_enc)
  layer_order_dec <- as.character(layer_order_dec)

  # ---------- index_col handling ----------
  if (!is.null(index_col)) {
    if (!index_col %in% colnames(data)) stop("`index_col` not found in data.")
    index_vals <- data[[index_col]]
    data       <- data[, setdiff(colnames(data), index_col), drop = FALSE]
  } else {
    index_vals <- NULL
  }
  orig_rn <- if (is.data.frame(data) || is.matrix(data)) rownames(data) else NULL
  orig_cn <- if (is.data.frame(data) || is.matrix(data)) colnames(data) else NULL

  # ---------- imports ----------
  run_mod <- reticulate::import("ciss_vae.utils.run_cissvae", convert = FALSE)
  np      <- reticulate::import("numpy", convert = FALSE)
  pd      <- reticulate::import("pandas", convert = FALSE)

  # ---------- build pandas DataFrame for `data` (fixes .isna() error) ----------
  if (is_py_obj(data)) {
    # If it's already a pandas object with .isna, pass through
    if (reticulate::py_has_attr(data, "isna")) {
      data_py <- data
    } else {
      # Best effort: wrap into pandas DataFrame
      data_py <- pd$DataFrame(reticulate::r_to_py(as.data.frame(data)))
    }
  } else if (is.data.frame(data)) {
    data_py <- pd$DataFrame(reticulate::r_to_py(data))
  } else if (is.matrix(data)) {
    # Keep column names if present
    df_tmp <- as.data.frame(data, stringsAsFactors = FALSE)
    colnames(df_tmp) <- colnames(data)
    data_py <- pd$DataFrame(reticulate::r_to_py(df_tmp))
  } else {
    # fallback
    data_py <- pd$DataFrame(reticulate::r_to_py(as.data.frame(data)))
  }

  # ---------- prepare py args ----------
  if (!is.null(clusters)) {
    if (is.data.frame(clusters)) clusters <- clusters[[1]]
    clusters <- as.vector(clusters)
    clusters_py <- np$array(as.integer(clusters))
  } else clusters_py <- NULL

  # Pass through Python object if already Python; otherwise convert to numpy/pandas
  if (!is.null(missingness_proportion_matrix)) {
    if (is_py_obj(missingness_proportion_matrix)) {
      prop_matrix_py <- missingness_proportion_matrix
    } else if (is.data.frame(missingness_proportion_matrix)) {
      prop_matrix_py <- pd$DataFrame(reticulate::r_to_py(missingness_proportion_matrix))
    } else {
      prop_matrix_py <- reticulate::r_to_py(as.matrix(missingness_proportion_matrix))
    }
  } else prop_matrix_py <- NULL

  if (!is.null(do_not_impute_matrix)) {
    if (is_py_obj(do_not_impute_matrix)) {
      dni_py <- do_not_impute_matrix
    } else {
      dni_py <- reticulate::r_to_py(as.matrix(do_not_impute_matrix))
    }
  } else dni_py <- NULL

  py_args <- list(
    data                  = data_py,
    val_proportion        = val_proportion,
    replacement_value     = replacement_value,
    columns_ignore        = if (is.null(columns_ignore)) NULL else reticulate::r_to_py(columns_ignore),
    print_dataset         = print_dataset,
    clusters              = clusters_py,
    n_clusters            = n_clusters,
    seed                  = seed,
    missingness_proportion_matrix = prop_matrix_py,
    scale_features        = scale_features,
    hidden_dims           = reticulate::r_to_py(hidden_dims),
    latent_dim            = latent_dim,
    layer_order_enc       = reticulate::r_to_py(layer_order_enc),
    layer_order_dec       = reticulate::r_to_py(layer_order_dec),
    latent_shared         = latent_shared,
    output_shared         = output_shared,
    batch_size            = batch_size,
    return_model          = return_model,
    epochs                = epochs,
    initial_lr            = initial_lr,
    decay_factor          = decay_factor,
    beta                  = beta,
    device                = device,
    max_loops             = max_loops,
    patience              = patience,
    epochs_per_loop       = epochs_per_loop,
    initial_lr_refit      = initial_lr_refit,
    decay_factor_refit    = decay_factor_refit,
    beta_refit            = beta_refit,
    verbose               = verbose,
    return_clusters       = return_clusters,
    return_silhouettes    = return_silhouettes,
    return_history        = return_history,
    return_dataset        = return_dataset
  )

  # Conditionally add NEW clustering & control args if Python actually accepts them
  if (accepts_arg(run_mod$run_cissvae, "do_not_impute_matrix")) py_args$do_not_impute_matrix <- dni_py
  if (accepts_arg(run_mod$run_cissvae, "k_neighbors"))          py_args$k_neighbors          <- as.integer(k_neighbors)
  if (accepts_arg(run_mod$run_cissvae, "leiden_resolution"))    py_args$leiden_resolution    <- as.numeric(leiden_resolution)
  if (accepts_arg(run_mod$run_cissvae, "leiden_objective"))     py_args$leiden_objective     <- as.character(leiden_objective)
  if (accepts_arg(run_mod$run_cissvae, "debug"))                py_args$debug                <- isTRUE(debug)

  # Drop NULLs
  py_args <- py_args[!vapply(py_args, is.null, logical(1))]

  # ---------- call python ----------
  res_py <- do.call(run_mod$run_cissvae, py_args)

  # ---------- parse return by flags/order ----------
  is_seq <- inherits(res_py, "python.builtin.object") &&
            reticulate::py_has_attr(res_py, "__len__") &&
            reticulate::py_has_attr(res_py, "__getitem__")
  geti <- function(x, i) if (is_seq) tryCatch(x[[i]], error = function(e) NULL) else if (i == 1L) x else NULL

  idx <- 1L
  imputed_py <- geti(res_py, idx); idx <- idx + 1L
  if (is.null(imputed_py)) stop("run_cissvae(): imputed dataset missing from Python return.")

  model_py    <- if (isTRUE(return_model))       { z <- geti(res_py, idx); idx <- idx + 1L; z } else NULL
  dataset_py  <- if (isTRUE(return_dataset))     { z <- geti(res_py, idx); idx <- idx + 1L; z } else NULL
  clusters_py <- if (isTRUE(return_clusters))    { z <- geti(res_py, idx); idx <- idx + 1L; z } else NULL
  silh_py     <- if (isTRUE(return_silhouettes)) { z <- geti(res_py, idx); idx <- idx + 1L; z } else NULL
  hist_py     <- if (isTRUE(return_history))     { z <- geti(res_py, idx); idx <- idx + 1L; z } else NULL

  # ---------- convert/repair names ----------
  imputed_df <- to_r_df_safely(imputed_py)
  imputed_df <- as.data.frame(imputed_df, stringsAsFactors = FALSE)

  rn_from_py <- NULL
  if (inherits(imputed_py, "python.builtin.object") &&
      reticulate::py_has_attr(imputed_py, "index") &&
      reticulate::py_has_attr(imputed_py$index, "tolist")) {
    rn_from_py <- tryCatch(reticulate::py_to_r(imputed_py$index$tolist()), error = function(e) NULL)
    if (!is.null(rn_from_py)) rn_from_py <- as.character(rn_from_py)
  }

  if (!is.null(rn_from_py) && length(rn_from_py) == nrow(imputed_df)) rownames(imputed_df) <- rn_from_py
  if (!is.null(orig_rn)    && length(orig_rn)    == nrow(imputed_df)) rownames(imputed_df) <- orig_rn
  if (!is.null(orig_cn)    && length(orig_cn)    == ncol(imputed_df)) colnames(imputed_df) <- orig_cn

  if (!is.null(index_vals) && length(index_vals) == nrow(imputed_df)) {
    imputed_df[[index_col]] <- index_vals
    imputed_df <- imputed_df[, c(index_col, setdiff(names(imputed_df), index_col)), drop = FALSE]
  } else if (!is.null(index_vals) && length(index_vals) != nrow(imputed_df) && isTRUE(verbose)) {
    message("run_cissvae(): index_col length mismatch; not attaching index_col.")
  }

  # ---------- assemble output ----------
  out <- list(imputed = imputed_df)
  if (isTRUE(return_model)       && !is.null(model_py))    out$model       <- model_py
  if (isTRUE(return_dataset)     && !is.null(dataset_py))  out$dataset     <- dataset_py
  if (isTRUE(return_clusters)    && !is.null(clusters_py)) out$clusters    <- tryCatch(reticulate::py_to_r(clusters_py), error = function(e) NULL)
  if (isTRUE(return_silhouettes) && !is.null(silh_py))     out$silhouettes <- tryCatch(reticulate::py_to_r(silh_py), error = function(e) NULL)
  if (isTRUE(return_history)     && !is.null(hist_py))     out$history     <- to_r_df_safely(hist_py)

  if (isTRUE(verbose)) cat("run_cissvae(): returned ->", paste(names(out), collapse = ", "), "\n")
  out
}
