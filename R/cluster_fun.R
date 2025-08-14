#' Cluster on Missingness Patterns
#'
#' Given an R data.frame or matrix with missing values, clusters on the pattern
#' of missingness and returns cluster labels plus silhouette score.
#'
#' @param data A data.frame or matrix (samples × features), may contain `NA`.
#' @param cols_ignore Character vector of column names to ignore when clustering.
#' @param n_clusters Integer; if provided, will run KMeans with this many clusters.
#'                   If `NULL`, will use HDBSCAN.
#' @param seed Integer; random seed for KMeans (or reproducibility in HDBSCAN).
#' @param min_cluster_size Integer; minimum cluster size for HDBSCAN.
#'                           If `NULL`, defaults to `nrow(data) %/% 25`.
#' @param cluster_selection_epsilon Numeric; epsilon parameter for HDBSCAN.
#'
#' @return A list with components:
#'   * `clusters`   — integer vector of cluster labels  
#'   * `silhouette` — numeric silhouette score, or `NA` if not computable  
#' @export
cluster_on_missing <- function(
  data,
  cols_ignore            = NULL,
  n_clusters             = NULL,
  seed                   = NULL,
  min_cluster_size       = NULL,
  cluster_selection_epsilon = 0.25
) {
  # load reticulate
  requireNamespace("reticulate", quietly = TRUE)

  # 1) import pandas and the helper function
  pd_mod      <- reticulate::import("pandas", convert = FALSE)
  helpers_mod <- reticulate::import("ciss_vae.utils.run_cissvae", convert = FALSE)
  cluster_fn  <- helpers_mod$cluster_on_missing

  # 2) convert R data → pandas DataFrame
  #    ensure colnames are kept (mask uses drop(columns=...))
  df_py <- pd_mod$DataFrame(reticulate::r_to_py(as.data.frame(data)))

  # 3) prepare arguments list
  args_py <- list(
    data                     = df_py,
    cluster_selection_epsilon = cluster_selection_epsilon
  )
  if (!is.null(cols_ignore)) {
    args_py$cols_ignore <- reticulate::r_to_py(as.character(cols_ignore))
  }
  if (!is.null(n_clusters))       args_py$n_clusters       <- as.integer(n_clusters)
  if (!is.null(seed))             args_py$seed             <- as.integer(seed)
  if (!is.null(min_cluster_size)) args_py$min_cluster_size <- as.integer(min_cluster_size)

  # 4) call Python
  out_py <- do.call(cluster_fn, args_py)

  # 5) unpack the 2‐tuple (0‐based indexing in Python)
  clusters_py   <- out_py[[0]]
  silhouette_py <- out_py[[1]]

  # 6) convert back to R
  clusters_r   <- as.integer(reticulate::py_to_r(clusters_py))
  sil_r_raw    <- reticulate::py_to_r(silhouette_py)
  silhouette_r <- if (is.null(sil_r_raw)) NA_real_ else as.numeric(sil_r_raw)

  # 7) return
  list(
    clusters   = clusters_r,
    silhouette = silhouette_r
  )
}


#' Create Missingness Proportion Matrix
#'
#' Creates a matrix where each entry represents the proportion of missing values
#' for each feature across samples. This matrix can be used with cluster_on_missing_prop
#' to identify features with similar missingness patterns.
#'
#' @param data Data frame or matrix containing the input data with potential missing values
#' @param index_col String name of index column to exclude from analysis (optional)
#' @param na_values Vector of values to treat as missing (default: c(NA, NaN, Inf, -Inf))
#' @param by_row Logical: if TRUE, compute missingness by row; if FALSE, by column (default: FALSE)
#'
#' @return Matrix where rows are samples and columns are features, with entries as missingness proportions [0,1]
#' @export
#'
#' @examples
#' # Create sample data with missing values
#' data <- data.frame(
#'   sample_id = 1:100,
#'   feature1 = c(rnorm(80), rep(NA, 20)),
#'   feature2 = c(rep(NA, 30), rnorm(70)),
#'   feature3 = rnorm(100)
#' )
#' 
#' # Create proportion matrix
#' prop_mat <- create_missingness_prop_matrix(data, index_col = "sample_id")
#' print(dim(prop_mat))  # Should be 100 x 3
#' print(head(prop_mat))
create_missingness_prop_matrix <- function(
  data,
  index_col = NULL,
  na_values = c(NA, NaN, Inf, -Inf),
  by_row = FALSE
) {
  requireNamespace("reticulate", quietly = TRUE)
  requireNamespace("dplyr", quietly = TRUE)
  # Handle index column
  if (!is.null(index_col)) {
    if (!index_col %in% colnames(data)) {
      stop("`index_col` '", index_col, "' not found in data.")
    }
    row_names <- data[[index_col]]
    data <- data[, setdiff(colnames(data), index_col), drop = FALSE]
  } else {
    row_names <- rownames(data)
  }
  
  # Convert to matrix
  mat <- if (is.data.frame(data)) as.matrix(data) else data
  
  # Create missingness indicator matrix
  is_missing <- matrix(FALSE, nrow = nrow(mat), ncol = ncol(mat))
  
  # Check for each type of missing value
  for (na_val in na_values) {
    if (is.na(na_val)) {
      is_missing <- is_missing | is.na(mat)
    } else if (is.infinite(na_val)) {
      if (na_val > 0) {
        is_missing <- is_missing | (mat == Inf)
      } else {
        is_missing <- is_missing | (mat == -Inf)
      }
    } else {
      is_missing <- is_missing | (mat == na_val)
    }
  }
  
  # Calculate proportions
  if (by_row) {
    # Each column represents proportion of missing values in that row
    prop_matrix <- apply(is_missing, 1, function(row) {
      colMeans(matrix(row, nrow = 1))
    })
    prop_matrix <- t(prop_matrix)  # Transpose to get samples as rows
  } else {
    # Each entry is the proportion of samples where that feature is missing
    # This creates a matrix where each row-column combination shows the
    # missingness proportion for that feature in that sample context
    n_samples <- nrow(mat)
    prop_matrix <- matrix(0, nrow = n_samples, ncol = ncol(mat))
    
    for (j in 1:ncol(mat)) {
      # For each feature, what proportion of samples have it missing
      feature_missing_prop <- sum(is_missing[, j]) / n_samples
      prop_matrix[, j] <- feature_missing_prop
    }
  }
  
  # Set row and column names
  rownames(prop_matrix) <- row_names
  colnames(prop_matrix) <- colnames(mat)
  
  return(prop_matrix)
}

#' Cluster Features Based on Missingness Proportions
#'
#' Groups features with similar patterns of missingness across samples using either
#' K-means clustering (when n_clusters is specified) or HDBSCAN (when n_clusters is NULL).
#' This helps identify features that tend to be missing together systematically.
#' 
#' Note: HDBSCAN may assign some features to "noise" (cluster -1). These features are 
#' kept and will be treated as individual clusters by CISS-VAE.
#'
#' @param prop_matrix Matrix or data frame where rows are samples, columns are features, 
#'   entries are missingness proportions [0,1]. Can be created with create_missingness_prop_matrix().
#' @param n_clusters Number of clusters for KMeans; if NULL, uses HDBSCAN (default: NULL)
#' @param seed Random seed for KMeans reproducibility (default: NULL)
#' @param min_cluster_size HDBSCAN minimum cluster size; if NULL, uses max(2, n_features/25) (default: NULL)
#' @param cluster_selection_epsilon HDBSCAN cluster selection threshold (default: 0.25)
#' @param metric Distance metric - "euclidean" or "cosine" (default: "euclidean")
#' @param scale_features Whether to standardize feature vectors before clustering (default: FALSE)
#' @param handle_noise How to handle HDBSCAN noise points: "keep" (default), "separate", or "merge"
#'
#' @return List containing:
#'   - labels: Integer vector of cluster assignments for each feature (includes -1 for noise if HDBSCAN)
#'   - labels_positive: Integer vector with noise points converted to positive cluster IDs
#'   - silhouette_score: Silhouette score for clustering quality (NULL if not calculable)
#'   - feature_names: Names of features corresponding to cluster labels
#'   - n_noise: Number of features assigned to noise (HDBSCAN only)
#' @export
#'
#' @examples
#' # Create sample data with systematic missingness patterns
#' set.seed(123)
#' data <- data.frame(
#'   sample_id = 1:100,
#'   # Group 1: High missingness in first 50 samples
#'   feat1 = c(rep(NA, 50), rnorm(50)),
#'   feat2 = c(rep(NA, 45), rnorm(55)),
#'   feat3 = c(rep(NA, 48), rnorm(52)),
#'   # Group 2: High missingness in last 50 samples  
#'   feat4 = c(rnorm(50), rep(NA, 50)),
#'   feat5 = c(rnorm(52), rep(NA, 48)),
#'   # Group 3: Low missingness throughout
#'   feat6 = c(rnorm(90), rep(NA, 10)),
#'   feat7 = c(rnorm(88), rep(NA, 12))
#' )
#' 
#' # Create proportion matrix
#' prop_mat <- create_missingness_prop_matrix(data, index_col = "sample_id")
#' 
#' # Cluster features by missingness pattern
#' clusters <- cluster_on_missing_prop(prop_mat, n_clusters = 3)
#' print(clusters$labels)
#' print(clusters$silhouette_score)
cluster_on_missing_prop <- function(
  prop_matrix,
  n_clusters = NULL,
  seed = NULL,
  min_cluster_size = NULL,
  cluster_selection_epsilon = 0.25,
  metric = "euclidean",
  scale_features = FALSE,
  handle_noise = "keep"
) {
  requireNamespace("reticulate", quietly = TRUE)
  requireNamespace("dplyr", quietly = TRUE)
  # Import Python clustering function
  cluster_mod <- import("ciss_vae.utils.helpers", convert = FALSE)
  cluster_func <- cluster_mod$cluster_on_missing_prop
  np <- import("numpy", convert = FALSE)
  
  # Prepare data
  if (is.data.frame(prop_matrix)) {
    feature_names <- colnames(prop_matrix)
    prop_py <- r_to_py(as.matrix(prop_matrix))
  } else {
    feature_names <- colnames(prop_matrix)
    if (is.null(feature_names)) {
      feature_names <- paste0("feature_", 1:ncol(prop_matrix))
    }
    prop_py <- r_to_py(prop_matrix)
  }
  
  # Validate inputs
  if (!is.null(n_clusters)) {
    n_clusters <- as.integer(n_clusters)
    if (n_clusters < 2) {
      stop("n_clusters must be >= 2")
    }
  }
  
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }
  
  if (!is.null(min_cluster_size)) {
    min_cluster_size <- as.integer(min_cluster_size)
    if (min_cluster_size < 2) {
      stop("min_cluster_size must be >= 2")
    }
  }
  
  if (!metric %in% c("euclidean", "cosine")) {
    stop("metric must be 'euclidean' or 'cosine'")
  }
  
  if (!handle_noise %in% c("keep", "separate", "merge")) {
    stop("handle_noise must be 'keep', 'separate', or 'merge'")
  }
  
  # Build arguments
  args_py <- list(
    prop_matrix = prop_py,
    n_clusters = n_clusters,
    seed = seed,
    min_cluster_size = min_cluster_size,
    cluster_selection_epsilon = cluster_selection_epsilon,
    metric = metric,
    scale_features = scale_features
  )
  
  # Remove NULL values
  args_py <- args_py[!sapply(args_py, is.null)]
  
  # Call Python function
  result_py <- do.call(cluster_func, args_py)
  
  # Convert results back to R
  result_list <- py_to_r(result_py)
  labels <- as.integer(result_list[[1]])
  silhouette_score <- result_list[[2]]
  
  # Handle noise points (HDBSCAN assigns -1 to noise)
  n_noise <- sum(labels == -1)
  
  # Create positive labels for CISS-VAE (which expects non-negative cluster IDs)
  labels_positive <- labels
  if (n_noise > 0) {
    max_cluster <- max(labels[labels >= 0])
    
    if (handle_noise == "keep") {
      # Give each noise point its own cluster ID
      noise_indices <- which(labels == -1)
      labels_positive[noise_indices] <- (max_cluster + 1):(max_cluster + n_noise)
      
    } else if (handle_noise == "separate") {
      # All noise points get the same new cluster ID
      labels_positive[labels == -1] <- max_cluster + 1
      
    } else if (handle_noise == "merge") {
      # Merge noise points with the largest existing cluster
      if (max_cluster >= 0) {
        cluster_counts <- table(labels[labels >= 0])
        largest_cluster <- as.integer(names(cluster_counts)[which.max(cluster_counts)])
        labels_positive[labels == -1] <- largest_cluster
      } else {
        # All points are noise - assign them all to cluster 0
        labels_positive[labels == -1] <- 0
      }
    }
  }
  
  # Calculate statistics
  unique_original <- unique(labels[labels >= 0])
  unique_positive <- unique(labels_positive)
  
  # Return structured result
  list(
    labels = labels,  # Original labels (may include -1 for noise)
    labels_positive = labels_positive,  # All non-negative labels for CISS-VAE
    silhouette_score = silhouette_score,
    feature_names = feature_names,
    n_features = length(feature_names),
    n_clusters_found = length(unique_original),  # Actual clusters found (excluding noise)
    n_clusters_final = length(unique_positive),  # Final clusters after noise handling
    n_noise = n_noise,
    handle_noise = handle_noise
  )
}