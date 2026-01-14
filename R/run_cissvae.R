#' Run the CISS-VAE pipeline for missing data imputation
#'
#' @description
#' This function wraps the Python `run_cissvae` function from the `ciss_vae` package,
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
#'   will use [Leiden](https://leidenalg.readthedocs.io/en/stable/intro.html) for clustering.  
#'   Default `NULL`.
#' @param k_neighbors Integer. Number of nearest neighbors for Leiden clustering. Defaults to 15.
#' @param leiden_resolution Float. Resolution parameter for Leiden clustering. Defaults to 0.5. 
#' @param leiden_objective Character. Objective function for Leiden clustering. One of ("CPM", "RB", "Modularity")
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
#' @param imputable_matrix Logical matrix indicating entries allowed to be imputed.
#' @param binary_feature_mask Logical vector marking which columns are binary.
#' @param weight_decay Weight decay (L2 penalty) used in Adam optimizer.
#' @param debug Logical; if TRUE, additional metadata is returned for debugging.
#' @param return_validation_dataset Logical. If `TRUE` returns validation dataset 
#' @param return_clusters Logical. If TRUE returns cluster vector
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
#'   \item{imputed_dataset}{data.frame of imputed data with same dimensions as input.
#'     Missing values are filled with model predictions. If `index_col` was
#'     provided, it is re-attached as the first column.}
#'   \item{model}{(if `return_model=TRUE`) Python CISSVAE model object.
#'     Can be used for further analysis or predictions.}
#'   \item{cluster_dataset}{(if `return_dataset=TRUE`) Python ClusterDataset object
#'     containing validation data, masks, normalization parameters, and cluster labels.
#'     Can be used with performance_by_cluster() and other analysis functions.}
#'  \item{clusters}{(if `return_clusters=TRUE`) Returns vector of cluster assignments}
#'   \item{silhouettes}{(if `return_silhouettes=TRUE`) Numeric silhouette
#'     score measuring cluster separation quality.}
#'   \item{training_history}{(if `return_history=TRUE`) data.frame containing training
#'     history with columns for epoch, losses, and validation metrics.}
#'   \item{val_data}{(if `return_validation_dataset=TRUE`) data.frame containing values held
#'    aside for validation.}
#'   \item{val_imputed}{(if `return_validation_dataset=TRUE`) data.frame containing imputed values of set held
#'    aside for validation.}
#' }
#'
#' @section Requirements:
#' This function requires the Python `ciss_vae` package to be installed and
#' accessible via `reticulate`. 
#'
#' @section Performance tips:
#' \itemize{
#'   \item If Leiden clustering yields too many clusters, consider increasing \code{k_neighbors} or reducing \code{leiden_resolution}.
#'   \item Use GPU computation when available for faster training on large datasets. Use \code{check_devices()} to see what devices are available.
#'   \item Adjust \code{batch_size} based on available memory (larger is faster but uses more memory).
#'   \item Set \code{verbose = TRUE} to monitor training progress.
#' }
#' @seealso
#' \code{\link{create_missingness_prop_matrix}} for creating missingness proportion matrices
#' \code{\link{performance_by_cluster}} for analyzing model performance using the returned dataset
#' 
#' @examples
#' \donttest{
#' library(rCISSVAE)
#'
#' data(df_missing)
#' data(clusters)
#'
#' try({
#' dat = run_cissvae(
#'  data = df_missing,
#'  index_col = "index",
#'  val_proportion = 0.1, ## pass a vector for different proportions by cluster
#'  columns_ignore = c("Age", "Salary", "ZipCode10001", "ZipCode20002", "ZipCode30003"), 
#'  clusters = clusters$clusters, ## we have precomputed cluster labels so we pass them here
#'  epochs = 5,
#'  return_silhouettes = FALSE,
#'  return_history = TRUE,  # Get detailed training history
#'  verbose = FALSE,
#'  return_model = TRUE, ## Allows for plotting model schematic
#'  device = "cpu",  # Explicit device selection
#'  layer_order_enc = c("unshared", "shared", "unshared"),
#'  layer_order_dec = c("shared", "unshared", "shared")
#')
#' })
#' }
#' @export
#' 
run_cissvae <- function(
  data,
  index_col              = NULL,
  val_proportion         = 0.1,
  replacement_value      = 0.0,
  columns_ignore         = NULL,
  imputable_matrix   = NULL,
  binary_feature_mask = NULL,
  print_dataset          = TRUE, 
  ## Cluster stuff
  clusters               = NULL,
  n_clusters             = NULL,
  seed                   = 42,
  missingness_proportion_matrix = NULL,
  scale_features         = FALSE,
  k_neighbors            = 15L,
  leiden_resolution      = 0.5,
  leiden_objective       = "CPM",
  ## Model Stuff
  hidden_dims            = c(150, 120, 60),
  latent_dim             = 15,
  layer_order_enc        = c("unshared", "unshared", "unshared"),
  layer_order_dec        = c("shared", "shared", "shared"),
  latent_shared          = FALSE,
  output_shared          = FALSE,
  batch_size             = 4000,
  epochs                 = 500,
  initial_lr             = 0.01,
  decay_factor           = 0.999,
  weight_decay = 0.001,
  beta                   = 0.001,
  device                 = NULL,
  max_loops              = 100,
  patience               = 2,
  epochs_per_loop        = NULL,
  initial_lr_refit       = NULL,
  decay_factor_refit     = NULL,
  beta_refit             = NULL,
  ## other params
  verbose                = FALSE,
  return_model           = TRUE,
  return_clusters        = FALSE,     # if clusters initiially null, will automatically return clusters. 
  return_silhouettes     = FALSE,
  return_history         = FALSE,
  return_dataset         = FALSE,
  return_validation_dataset = FALSE,
  debug                  = FALSE
){

      # Check if reticulate has initialized Python
  if (is.null(reticulate::py_config()$python)) {
      stop(
        "Python is not initialized in this session. ",
        "Please activate a reticulate Python environment before calling this function, ",
        "for example by calling:\n",
        "  reticulate::use_virtualenv(\"your/env/path\", required = TRUE)\n",
        "or\n",
        "  reticulate::use_condaenv(\"your_env_name\", required = TRUE)\n",
        "and then re-run the function.",
        call. = FALSE
      )
    } 
  
  is_py_obj <- function(x) inherits(x, "python.builtin.object")
  ## step 0: if return_validation_dataset, set return_dataset = true. If run_cissvae does your clusters, return_clusters = true
  if(return_validation_dataset){
    return_dataset = TRUE
  }
  if(is.null(clusters)){
    return_clusters = TRUE
  }

  ## preserve raw data
  data_raw = as.data.frame(data)

  ## step 0.5: make everything numeric double; keep NaN for missing -> avoids sentinel vals
  to_numeric_matrix <- function(df, debug) {
  df <- as.data.frame(df)
  # convert factor/character/logical/integer to numeric (double)
  for (nm in names(df)) {
    col <- df[[nm]]
    if (is.logical(col))       col <- as.numeric(col)       # TRUE/FALSE -> 1/0, NAs kept
    else if (is.factor(col))   col <- as.numeric(col)       # factor -> codes, NA kept
    else if (is.integer(col))  col <- as.numeric(col)       # int -> double
    else if (is.character(col)) col <- suppressWarnings(as.numeric(col))
    # ensure missing is NaN (good for pandas/NumPy)
    col[is.na(col)] <- NaN
    df[[nm]] <- col
  }
    mat = as.matrix(df)  # guarantees homogeneous double
    colnames(mat) = colnames(df)
    rownames(mat) = rownames(df)

    if(debug){
      print(colnames(mat))
    }
    return(mat)
  }


  ## step 1: coerse numerics 
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
  
  ## step 2: handle index col if passed
  if (!is.null(index_col)) {
    if (!index_col %in% colnames(data)) stop("`index_col` not found in data.")
    index_vals <- data[[index_col]]
    data       <- data[, setdiff(colnames(data), index_col), drop = FALSE]

    if (index_col %in% colnames(imputable_matrix)) {
      imputable_matrix <- imputable_matrix[, setdiff(colnames(imputable_matrix), index_col), drop = FALSE]
    }
    ## handle index col in binary_feature_mask
    if(!is.null(binary_feature_mask) & !is.null(names(binary_feature_mask))){
      binary_feature_mask =  binary_feature_mask[ setdiff(names(binary_feature_mask), index_col), drop = FALSE]
      if(debug){
        cat("Binary feature mask: ", paste0(names(binary_feature_mask), collapse = ", "))
      }
    }
  } else {
    index_vals <- NULL
  }
  ## grab original row and column names
  orig_rn <- if (is.data.frame(data) || is.matrix(data)) rownames(data) else NULL
  orig_cn <- if (is.data.frame(data) || is.matrix(data)) colnames(data) else NULL

  ## if there is a imputable_matrix make sure it has the same dimensions as data
  if (!is.null(imputable_matrix)) {
    if (!all(dim(imputable_matrix) == dim(data))) {
      stop(sprintf(
        "Dimension mismatch: data is %d x %d, imputable_matrix is %d x %d",
        nrow(data), ncol(data),
        nrow(imputable_matrix), ncol(imputable_matrix)
      ))
    }
  }
  
  ## step 3: do python imports

  run_mod <- reticulate::import("ciss_vae.training.run_cissvae", convert = FALSE)
  np      <- reticulate::import("numpy", convert = FALSE)
  pd      <- reticulate::import("pandas", convert = FALSE)


  ## step 4: coerce R data.frame -> numeric double matrix -> pandas
  ## R function should be able to accept matrix or df. Does need column names
  data_num <- to_numeric_matrix(data, debug = debug)  # has colnames/rownames
  cols_py  <- reticulate::r_to_py(colnames(data_num))
  idx_py   <- if (!is.null(rownames(data_num))) reticulate::r_to_py(rownames(data_num)) else NULL
  
  pd   <- reticulate::import("pandas", convert = FALSE)
  np   <- reticulate::import("numpy",  convert = FALSE)
  
  data_np <- reticulate::r_to_py(unclass(data_num))  # 2D ndarray (float64)
  
  if (is.null(idx_py)) {
    data_py <- pd$DataFrame(data = data_np, columns = cols_py)
  } else {
    data_py <- pd$DataFrame(data = data_np, columns = cols_py, index = idx_py)
  }
  
  if (debug) {
    print(data_py$head())   # <- call it
  }


  ## step 5: perpare python args
  if (!is.null(clusters)) { ## check for clusters
    if (is.data.frame(clusters)) clusters <- clusters[[1]]
    clusters <- as.vector(clusters)
    clusters_py <- np$array(as.integer(clusters))
  } else clusters_py <- NULL

  if (!is.null(missingness_proportion_matrix)) {
    if (is_py_obj(missingness_proportion_matrix)) {
      prop_matrix_py <- missingness_proportion_matrix
    } else if (is.data.frame(missingness_proportion_matrix)) {
      prop_matrix_py <- pd$DataFrame(reticulate::r_to_py(missingness_proportion_matrix))
    } else {
      prop_matrix_py <- reticulate::r_to_py(as.matrix(missingness_proportion_matrix))
    }
  } else prop_matrix_py <- NULL

  if (!is.null(imputable_matrix)) {
    if (is_py_obj(imputable_matrix)) {
      dni_py <- imputable_matrix
    } else {
      dni_py <- reticulate::r_to_py(imputable_matrix)
    }
  } else dni_py <- NULL

  py_args <- list(
    data                  = data_py,
    val_proportion        = val_proportion,
    replacement_value     = replacement_value,
    columns_ignore        = if (is.null(columns_ignore)) NULL else reticulate::r_to_py(list(as.character(columns_ignore))),
    print_dataset         = print_dataset,
    clusters              = clusters_py,
    n_clusters            = n_clusters,
    seed                  = seed,
    binary_feature_mask  = reticulate::r_to_py(binary_feature_mask),
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
    weight_decay = weight_decay,
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
    return_dataset        = return_dataset,
    imputable_matrix = dni_py,
    k_neighbors           = as.integer(k_neighbors),
    leiden_resolution     = as.numeric(leiden_resolution),
    leiden_objective      = as.character(leiden_objective),
    debug                 = debug
  )
  ## drop nulls from python args
  py_args <- py_args[!vapply(py_args, is.null, logical(1))]

  ## step 6: call python w/ these args

  res_py = do.call(run_mod$run_cissvae, py_args)

  
  ## res_py[0] = imputed df, res_py[1] is model
  
  ## step 7: return to r types
  res_r = reticulate::py_to_r(res_py)
  
  if(is.data.frame(res_r)){
    imputed_df = res_r
  }
  else{
    imputed_df = res_r[[1]]
  }

  ## put index back on if there was an index
  if (!is.null(index_vals) && length(index_vals) == nrow(imputed_df)) {
    imputed_df[[index_col]] <- index_vals
    imputed_df <- imputed_df[, c(index_col, setdiff(names(imputed_df), index_col)), drop = FALSE]
  } else if (!is.null(index_vals) && length(index_vals) != nrow(imputed_df) && isTRUE(verbose)) {
    message("run_cissvae(): index_col length mismatch; not attaching index_col.")
  }

  ## step 8: prepare output list
  out = list(imputed_dataset = as.data.frame(imputed_df), raw_data = data_raw)
  
  i = 2 ## starting from second entry in res_r, construct the output object
  if(return_model){
    ## keep model as python object, return rest as R object
    out[["model"]] = res_py[i-1]
    i = i+1
  }
  if(return_dataset){
    ## keep cluster_dataset as a python object aslo
    out[["cluster_dataset"]] = res_py[i - 1]
    i = i+1
  }
  if(return_clusters){
    out[["clusters"]] = as.vector(res_r[i][[1]])
    i = i+1
  }
  if(return_silhouettes){
    out[["silhouette_width"]] = as.numeric(res_r[i])
    i = i+1
  }
  if(return_history){
    out[["training_history"]] = as.data.frame(res_r[i])
    i = i+1
  }

  if(return_validation_dataset){
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
    out[["val_imputed"]] = val_imputed
  }

    
  return(out)

}