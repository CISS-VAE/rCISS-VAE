#' Run the CISS-VAE pipeline (letting Python choose device)
#'
#' @description
#' Wraps the Python `run_cissvae` function from the `ciss_vae` module,
#' handles an optional `index_col`, and returns imputed data,
#' and optionally the model, silhouette scores, and training history.
#'
#' @param data A data.frame or matrix (samples × features), may contain `NA`.
#' @param index_col Character. Column in `data` to treat as sample ID; removed before training and re-attached. Default `NULL`.
#' @param val_proportion Numeric fraction of non-missing entries to hold out. Default `0.1`.
#' @param replacement_value Numeric fill value for masked entries. Default `0.0`.
#' @param columns_ignore Character or integer vector of columns to ignore. Default `NULL`.
#' @param print_dataset Logical; if `TRUE`, prints dataset summary. Default `TRUE`.
#' @param clusters Optional vector (or single-column data.frame) of precomputed cluster labels. Default `NULL`.
#' @param n_clusters Integer for KMeans if `clusters` is `NULL`. Default `NULL`.
#' @param cluster_selection_epsilon Numeric epsilon for HDBSCAN. Default `0.25`.
#' @param seed Integer random seed. Default `42`.
#' @param missingness_proportion_matrix Optional pre-computed missingness proportion matrix for biomarker clustering. Default `NULL`.
#' @param scale_features Logical; whether to scale features in proportion matrix clustering. Default `FALSE`.
#' @param hidden_dims Integer vector of hidden layer sizes. Default `c(150,120,60)`.
#' @param latent_dim Integer latent space dimension. Default `15`.
#' @param layer_order_enc Character vector for encoder layer sharing. Default `c("unshared","unshared","unshared")`.
#' @param layer_order_dec Character vector for decoder layer sharing. Default `c("shared","shared","shared")`.
#' @param latent_shared Logical; share latent weights? Default `FALSE`.
#' @param output_shared Logical; share output weights? Default `FALSE`.
#' @param batch_size Integer batch size. Default `4000`.
#' @param return_model Logical; if `TRUE`, returns Python model. Default `TRUE`.
#' @param epochs Integer initial training epochs. Default `500`.
#' @param initial_lr Numeric initial learning rate. Default `0.01`.
#' @param decay_factor Numeric learning rate decay. Default `0.999`.
#' @param beta Numeric KL weight. Default `0.001`.
#' @param device Character device specification ("cpu" or "cuda"); auto-selects if `NULL`. Default `NULL`.
#' @param max_loops Integer max refit loops. Default `100`.
#' @param patience Integer early stop patience. Default `2`.
#' @param epochs_per_loop Integer epochs per refit loop. Default `NULL` (uses `epochs`).
#' @param initial_lr_refit Numeric LR for refit loops. Default `NULL` (uses `initial_lr`).
#' @param decay_factor_refit Numeric decay for refit loops. Default `NULL` (uses `decay_factor`).
#' @param beta_refit Numeric KL weight for refit loops. Default `NULL` (uses `beta`).
#' @param verbose Logical; if `TRUE`, prints progress. Default `FALSE`.
#' @param return_silhouettes Logical; if `TRUE`, returns silhouette scores. Default `FALSE`.
#' @param return_history Logical; if `TRUE`, returns concatenated training history. Default `FALSE`.
#'
#' @return A list with elements depending on return flags:
#'   - `imputed`: data.frame of imputed values (with `index_col` re-attached) - always returned.
#'   - `model`: Python VAE object (if `return_model = TRUE`).
#'   - `silhouettes`: numeric silhouette score (if `return_silhouettes = TRUE`).
#'   - `history`: data.frame of training history (if `return_history = TRUE`).
#' @export
#'
#' @examples
#' # Basic usage
#' result <- run_cissvae(my_data, clusters = my_clusters)
#' imputed_data <- result$imputed
#' 
#' # With training history and silhouettes
#' result <- run_cissvae(
#'   data = my_data,
#'   index_col = "sample_id",
#'   return_history = TRUE,
#'   return_silhouettes = TRUE,
#'   verbose = TRUE
#' )
#' 
#' # Using pre-computed missingness proportion matrix
#' prop_matrix <- create_missingness_prop_matrix(my_data, index_col = "sample_id")
#' result <- run_cissvae(
#'   data = my_data,
#'   index_col = "sample_id",
#'   missingness_proportion_matrix = prop_matrix,
#'   scale_features = TRUE
#' )
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
  # NEW: Missingness proportion matrix parameters
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
  # NEW: Device parameter
  device                 = NULL,
  max_loops              = 100,
  patience               = 2,
  epochs_per_loop        = NULL,
  initial_lr_refit       = NULL,
  decay_factor_refit     = NULL,
  beta_refit             = NULL,
  verbose                = FALSE,
  return_silhouettes     = FALSE,
  # NEW: Training history parameter
  return_history         = FALSE
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
  
  # FIXED: These should remain numeric, not integer
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
    # NEW: Missingness proportion matrix parameters
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
    # NEW: Device parameter
    device                        = device,
    max_loops                     = max_loops,
    patience                      = patience,
    epochs_per_loop               = epochs_per_loop,
    initial_lr_refit              = initial_lr_refit,
    decay_factor_refit            = decay_factor_refit,
    beta_refit                    = beta_refit,
    verbose                       = verbose,
    return_silhouettes            = return_silhouettes,
    # NEW: Training history parameter
    return_history                = return_history
  )
  
  # Filter out NULLs
  py_args <- py_args[!vapply(py_args, is.null, logical(1))]
  
  # ── 9) Call Python function ─────────────────────────────────────────────
  res_py <- do.call(py_mod$run_cissvae, py_args)
  res    <- reticulate::py_to_r(res_py)
  
  # ── 10) Handle return values based on what was requested ────────────────
  # The Python function returns different combinations based on flags
  # We need to parse the returned tuple correctly
  
  if (is.list(res)) {
    # Multiple return values - need to parse based on return flags
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
  
  # ── 13) Assemble output based on return flags ───────────────────────────
  out <- list(imputed = imputed_df)
  
  if (!is.null(idx) && length(res) >= idx) {
    # Parse remaining return values in the order they appear in Python function
    if (return_model) {
      out$model <- res[[idx]]
      idx <- idx + 1
    }
    
    if (return_silhouettes && !is.null(idx) && length(res) >= idx) {
      out$silhouettes <- res[[idx]]
      idx <- idx + 1
    }
    
    if (return_history && !is.null(idx) && length(res) >= idx) {
      # Convert history to R data.frame
      history_py <- res[[idx]]
      if (!is.null(history_py)) {
        out$history <- as.data.frame(history_py, stringsAsFactors = FALSE)
      } else {
        out$history <- NULL
      }
      idx <- idx + 1
    }
  }
  
  return(out)
}