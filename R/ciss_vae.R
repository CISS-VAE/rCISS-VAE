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
  cluster_selection_epsilon = 0.25,
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
  return_silhouettes     = FALSE,
  return_history         = FALSE,
  return_dataset         = FALSE  # Changed from return_valdata
) {
  
  # ── 1) Coerce integer-only args ─────────────────────────────────────────
  seed            <- as.integer(seed)
  latent_dim      <- as.integer(latent_dim)
  batch_size      <- as.integer(batch_size)
  epochs          <- as.integer(epochs)
  max_loops       <- as.integer(max_loops)
  patience        <- as.integer(patience)
  
  # Optional single values 
  n_clusters      <- if (!is.null(n_clusters)) as.integer(n_clusters) else NULL
  epochs_per_loop <- if (!is.null(epochs_per_loop)) as.integer(epochs_per_loop) else NULL
  
  initial_lr_refit    <- if (!is.null(initial_lr_refit)) as.numeric(initial_lr_refit) else NULL
  decay_factor_refit  <- if (!is.null(decay_factor_refit)) as.numeric(decay_factor_refit) else NULL
  beta_refit          <- if (!is.null(beta_refit)) as.numeric(beta_refit) else NULL
  
  # Vectors
  hidden_dims     <- as.integer(hidden_dims)
  layer_order_enc <- as.character(layer_order_enc)
  layer_order_dec <- as.character(layer_order_dec)
  
  # ── 2) Handle index_col ─────────────────────────────────────────────────
  if (!is.null(index_col)) {
    if (!index_col %in% colnames(data)) stop("`index_col` not found in data.")
    index_vals <- data[[index_col]]
    data       <- data[, setdiff(colnames(data), index_col), drop = FALSE]
  } else {
    index_vals <- NULL
  }
  
  # ── 3) Capture original row/col names ───────────────────────────────────
  orig_rn <- if (is.data.frame(data) || is.matrix(data)) rownames(data) else NULL
  orig_cn <- if (is.data.frame(data) || is.matrix(data)) colnames(data) else NULL
  
  # ── 4) Convert to matrix ────────────────────────────────────────────────
  mat <- if (is.data.frame(data)) as.matrix(data) else data
  
  # ── 5) Import Python modules ────────────────────────────────────────────
  py_mod <- reticulate::import("ciss_vae.utils", convert = FALSE)
  np     <- reticulate::import("numpy", convert = FALSE)
  
  # ── 6) Prepare clusters argument ────────────────────────────────────────
  if (!is.null(clusters)) {
    if (is.data.frame(clusters)) clusters <- clusters[[1]]
    clusters    <- as.vector(clusters)
    clusters_py <- np$array(clusters)
  } else {
    clusters_py <- NULL
  }
  
  # ── 7) Prepare missingness proportion matrix ────────────────────────────
  if (!is.null(missingness_proportion_matrix)) {
    if (is.data.frame(missingness_proportion_matrix)) {
      prop_matrix_py <- reticulate::r_to_py(as.matrix(missingness_proportion_matrix))
    } else {
      prop_matrix_py <- reticulate::r_to_py(missingness_proportion_matrix)
    }
  } else {
    prop_matrix_py <- NULL
  }
  
  # ── 8) Build argument list ──────────────────────────────────────────────
  py_args <- list(
    data                          = reticulate::r_to_py(mat),
    val_proportion                = val_proportion,
    replacement_value             = replacement_value,
    columns_ignore                = if (is.null(columns_ignore)) NULL else reticulate::r_to_py(columns_ignore),
    print_dataset                 = print_dataset,
    clusters                      = clusters_py,
    n_clusters                    = n_clusters,
    cluster_selection_epsilon     = cluster_selection_epsilon,
    seed                          = seed,
    missingness_proportion_matrix = prop_matrix_py,
    scale_features                = scale_features,
    hidden_dims                   = reticulate::r_to_py(hidden_dims),
    latent_dim                    = latent_dim,
    layer_order_enc               = reticulate::r_to_py(layer_order_enc),
    layer_order_dec               = reticulate::r_to_py(layer_order_dec),
    latent_shared                 = latent_shared,
    output_shared                 = output_shared,
    batch_size                    = batch_size,
    return_model                  = return_model,
    epochs                        = epochs,
    initial_lr                    = initial_lr,
    decay_factor                  = decay_factor,
    beta                          = beta,
    device                        = device,
    max_loops                     = max_loops,
    patience                      = patience,
    epochs_per_loop               = epochs_per_loop,
    initial_lr_refit              = initial_lr_refit,
    decay_factor_refit            = decay_factor_refit,
    beta_refit                    = beta_refit,
    verbose                       = verbose,
    return_silhouettes            = return_silhouettes,
    return_history                = return_history,
    return_dataset                = return_dataset  # Now maps directly
  )
  
  # Filter out NULLs
  py_args <- py_args[!vapply(py_args, is.null, logical(1))]
  
  # ── 9) Call Python function ─────────────────────────────────────────────
  res_py <- do.call(py_mod$run_cissvae, py_args)
  res    <- reticulate::py_to_r(res_py)
  
  # ── 10) Handle return values based on what was requested ────────────────
  # Python returns in this exact order:
  # 1. imputed_dataset (always)
  # 2. vae (if return_model=True)
  # 3. dataset (if return_dataset=True)
  # 4. silh (if return_silhouettes=True) 
  # 5. combined_history_df (if return_history=True)
  
  if (verbose) {
    cat("Received", if(is.list(res)) length(res) else 1, "return values\n")
    cat("Return flags: model =", return_model, ", dataset =", return_dataset, 
        ", silhouettes =", return_silhouettes, ", history =", return_history, "\n")
  }
  
  if (is.list(res)) {
    # Multiple return values
    imputed_data <- res[[1]]
    idx <- 2
  } else {
    # Single return value (just imputed data)
    imputed_data <- res
    idx <- NULL
  }
  
  # ── 11) Convert imputed data to data.frame ──────────────────────────────
  imputed_df <- as.data.frame(imputed_data, stringsAsFactors = FALSE)
  if (!is.null(orig_rn)) rownames(imputed_df) <- orig_rn
  if (!is.null(orig_cn)) colnames(imputed_df) <- orig_cn
  
  # ── 12) Re-attach index_col if provided ─────────────────────────────────
  if (!is.null(index_vals)) {
    imputed_df[[index_col]] <- index_vals
    imputed_df <- imputed_df[, c(index_col, setdiff(names(imputed_df), index_col))]
  }
  
  # ── 13) Assemble output based on return flags and exact Python order ────
  out <- list(imputed = imputed_df)
  
  if (!is.null(idx) && length(res) >= idx) {
    
    # Parse in the exact order Python returns them
    
    # 2. Model (if return_model=True)
    if (return_model && idx <= length(res)) {
      out$model <- res[[idx]]
      idx <- idx + 1
    }
    
    # 3. Dataset (if return_dataset=True) - Return ClusterDataset object as-is
    if (return_dataset && idx <= length(res)) {
      dataset_py <- res[[idx]]
      
      # Return the ClusterDataset object directly without conversion
      tryCatch({
        # Validate it's a ClusterDataset object
        if (reticulate::py_has_attr(dataset_py, "__class__")) {
          class_name <- reticulate::py_to_r(dataset_py$`__class__`$`__name__`)
          if (grepl("ClusterDataset", class_name)) {
            out$dataset <- dataset_py  # Return Python object directly
          } else {
            warning("Expected ClusterDataset but got ", class_name)
            out$dataset <- dataset_py  # Return anyway
          }
        } else {
          out$dataset <- dataset_py  # Return as-is
        }
      }, error = function(e) {
        warning("Could not validate ClusterDataset object: ", e$message)
        out$dataset <- dataset_py  # Return as-is if validation fails
      })
      
      idx <- idx + 1
    }
    
    # 4. Silhouettes (if return_silhouettes=True)
    if (return_silhouettes && idx <= length(res)) {
      out$silhouettes <- res[[idx]]
      idx <- idx + 1
    }
    
    # 5. History (if return_history=True)
    if (return_history && idx <= length(res)) {
      history_py <- res[[idx]]
      
      tryCatch({
        if (reticulate::py_has_attr(history_py, "to_numpy")) {
          # It's a pandas DataFrame
          out$history <- reticulate::py_to_r(history_py)
        } else if (is.data.frame(history_py)) {
          # Already converted to R data.frame
          out$history <- history_py
        } else if (is.list(history_py)) {
          # It's an R list - try to convert to data.frame
          if (length(history_py) > 0 && all(sapply(history_py, function(x) is.vector(x) && is.numeric(x)))) {
            # Check if all elements have the same length
            lengths <- sapply(history_py, length)
            if (length(unique(lengths)) == 1) {
              # All same length - can convert to data.frame
              out$history <- as.data.frame(history_py, stringsAsFactors = FALSE)
            } else {
              # Different lengths - return as list with warning
              warning("Training history has inconsistent lengths, returning as list")
              out$history <- history_py
            }
          } else {
            # Try direct conversion to data.frame
            out$history <- as.data.frame(history_py, stringsAsFactors = FALSE)
          }
        } else {
          # Try direct conversion
          out$history <- as.data.frame(reticulate::py_to_r(history_py), stringsAsFactors = FALSE)
        }
      }, error = function(e) {
        warning("Failed to convert training history to R data.frame: ", e$message)
        out$history <- history_py  # Return as-is if conversion fails
      })
      
      idx <- idx + 1
    }
  }
  
  return(out)
}