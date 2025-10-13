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
#' @param k_neighbors Integer; minimum cluster size for HDBSCAN.
#'                           If `NULL`, defaults to `nrow(data) %/% 25`.
#' @param leiden_resolution Numeric; epsilon parameter for HDBSCAN.
#'
#' @return A list with components:
#'   * `clusters`   — integer vector of cluster labels  
#'   * `silhouette` — numeric silhouette score, or `NA` if not computable  
#' @export
cluster_on_missing <- function(
  data,
  cols_ignore            = NULL,
  n_clusters             = NULL,
  seed                   = 42,
  k_neighbors       = NULL,
  leiden_resolution = 0.25,
  leiden_objective = "CPM",
  use_snn = TRUE
) {
  # load reticulate
  requireNamespace("reticulate", quietly = TRUE)

  # 1) import pandas and the helper function
  pd_mod      <- reticulate::import("pandas", convert = FALSE)
  helpers_mod <- reticulate::import("ciss_vae.utils.clustering", convert = FALSE)
  cluster_fn  <- helpers_mod$cluster_on_missing

  # 2) convert R data → pandas DataFrame
  #    ensure colnames are kept (mask uses drop(columns=...))
  df_py <- pd_mod$DataFrame(reticulate::r_to_py(as.data.frame(data)))

  # 3) prepare arguments list
  args_py <- list(
    data                     = df_py,
    leiden_resolution = reticulate::r_to_py(leiden_resolution),
    leiden_objective = reticulate::r_to_py(leiden_objective),
    use_snn = reticulate::r_to_py(use_snn),
    k_neighbors = reticulate::r_to_py(as.integer(k_neighbors))

  )
  if (!is.null(cols_ignore)) {
    args_py$cols_ignore <- reticulate::r_to_py(as.character(cols_ignore))
  }
  if (!is.null(n_clusters))       args_py$n_clusters       <- as.integer(n_clusters)
  if (!is.null(seed))             args_py$seed             <- as.integer(seed)
  # if (!is.null(k_neighbors)) args_py$k_neighbors <- as.integer(k_neighbors)

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

## ----------------------------------------------------






#' Cluster Samples Based on Missingness Proportions
#'
#' Groups **samples** with similar patterns of missingness across features using either
#' K-means clustering (when `n_clusters` is specified) or HDBSCAN (when `n_clusters` is `NULL`).
#' This is useful for detecting cohorts with shared missing-data behavior (e.g., site/batch effects).
#'
#' @param prop_matrix Matrix or data frame where **rows are samples** and **columns are features**,
#'   entries are missingness proportions in `[0,1]`. Can be created with `create_missingness_prop_matrix()`.
#' @param n_clusters Integer; number of clusters for KMeans. If `NULL`, uses HDBSCAN (default: `NULL`).
#' @param seed Integer; random seed for KMeans reproducibility (default: `NULL`).
#' @param k_neighbors Integer; HDBSCAN minimum cluster size. If `NULL`, Python default is used
#'   (typically a function of the number of samples) (default: `NULL`).
#' @param leiden_resolution Numeric; HDBSCAN cluster selection threshold (default: `0.25`).
#' @param metric Character; distance metric `"euclidean"` or `"cosine"` (default: `"euclidean"`).
#' @param scale_features Logical; whether to standardize **feature columns** before clustering samples (default: `FALSE`).
#' @param handle_noise Character; how to handle HDBSCAN noise points (`-1`):
#'   `"keep"` (each noise sample gets its own new cluster ID), `"separate"` (all noise samples share one new ID),
#'   or `"merge"` (noise samples assigned to largest existing cluster) (default: `"keep"`).
#'
#' @return A list with:
#' \itemize{
#'   \item \code{clusters}: Integer vector of cluster assignments per **sample** (may include -1 for HDBSCAN noise).
#'   \item \code{clusters_positive}: Integer vector with all labels non-negative after applying \code{handle_noise}.
#'   \item \code{silhouette_score}: Numeric silhouette score, or \code{NULL} if not computable.
#'   \item \code{sample_names}: Character vector of sample names corresponding to \code{clusters}.
#'   \item \code{n_samples}: Integer; number of samples (rows).
#'   \item \code{n_clusters_found}: Integer; number of clusters found (excluding noise).
#'   \item \code{n_clusters_final}: Integer; final number of clusters after noise handling.
#'   \item \code{n_noise}: Integer; number of samples assigned to noise (HDBSCAN only).
#'   \item \code{handle_noise}: The noise handling mode used.
#' }
#'
#' @examples
#' set.seed(123)
#' dat <- data.frame(
#'   sample_id = paste0("s", 1:12),
#'   # Two features measured at 3 timepoints each → proportions by feature per sample
#'   A_1 = c(NA, rnorm(11)), A_2 = c(NA, rnorm(11)), A_3 = rnorm(12),
#'   B_1 = rnorm(12),        B_2 = c(rnorm(10), NA, NA), B_3 = rnorm(12)
#' )
#' pm <- create_missingness_prop_matrix(dat, index_col = "sample_id",
#'                                      repeat_feature_names = c("A","B"))
#' res <- cluster_on_missing_prop(pm, n_clusters = 2, metric = "cosine", scale_features = TRUE)
#' table(res$clusters_positive)
#' res$silhouette_score
#'
#' @export
cluster_on_missing_prop <- function(
  prop_matrix,
  n_clusters = NULL,
  seed = NULL,
  k_neighbors = NULL,
  leiden_resolution = 0.25,
  use_snn = TRUE,
  leiden_objective = "CPM",
  metric = "euclidean",
  scale_features = FALSE,
  handle_noise = "keep"
) {
  # Dependencies
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it to use this function.")
  }

  # Locate Python function
  run_mod <- reticulate::import("ciss_vae.utils.clustering", convert = FALSE)
  cluster_func <- run_mod$cluster_on_missing_prop

  # Prepare data for Python

  if (is.data.frame(prop_matrix)) {
    sample_names <- rownames(prop_matrix)
    if (is.null(sample_names)) {
      sample_names <- paste0("sample_", seq_len(nrow(prop_matrix)))
    }
    prop_py <- reticulate::r_to_py(as.matrix(prop_matrix))
  } else if (is.matrix(prop_matrix)) {
    sample_names <- rownames(prop_matrix)
    if (is.null(sample_names)) {
      sample_names <- paste0("sample_", seq_len(nrow(prop_matrix)))
    }
    prop_py <- reticulate::r_to_py(prop_matrix)
  } else {
    stop("`prop_matrix` must be a data.frame or matrix with rows = samples and columns = features.")
  }

  # Validate inputs
  if (!is.null(n_clusters)) {
    n_clusters <- as.integer(n_clusters)
    if (n_clusters < 2) stop("n_clusters must be >= 2")
  }
  if (!is.null(seed)) seed <- as.integer(seed)
  if (!is.null(k_neighbors)) {
    k_neighbors <- as.integer(k_neighbors)
    if (k_neighbors < 2) stop("k_neighbors must be >= 2")
  }
  if (!metric %in% c("euclidean", "cosine")) {
    stop("metric must be 'euclidean' or 'cosine'")
  }
  if (!handle_noise %in% c("keep", "separate", "merge")) {
    stop("handle_noise must be 'keep', 'separate', or 'merge'")
  }

  # Build Python args, drop NULLs
  args_py <- list(
    prop_matrix = prop_py,
    n_clusters = n_clusters,
    seed = seed,
    k_neighbors = k_neighbors,
    leiden_resolution = leiden_resolution,
    metric = metric,
    scale_features = scale_features,
    leiden_objective = reticulate::r_to_py(leiden_objective),
    use_snn = reticulate::r_to_py(use_snn)
  )
  args_py <- args_py[!vapply(args_py, is.null, logical(1))]

  # Call Python and convert result
  result_py <- do.call(cluster_func, args_py)
  result_list <- reticulate::py_to_r(result_py)

  labels <- as.integer(result_list[[1]])          # per-sample labels
  silhouette_score <- result_list[[2]]

  # Noise handling (for HDBSCAN: -1 = noise)
  n_noise <- sum(labels == -1L, na.rm = TRUE)
  labels_positive <- labels

  if (n_noise > 0L) {
    max_cluster <- suppressWarnings(max(labels[labels >= 0L], na.rm = TRUE))
    if (!is.finite(max_cluster)) max_cluster <- -1L

    if (handle_noise == "keep") {
      noise_idx <- which(labels == -1L)
      labels_positive[noise_idx] <- seq.int(from = max_cluster + 1L,
                                            length.out = length(noise_idx))
    } else if (handle_noise == "separate") {
      labels_positive[labels == -1L] <- max_cluster + 1L
    } else if (handle_noise == "merge") {
      if (max_cluster >= 0L) {
        tab <- table(labels[labels >= 0L])
        largest <- as.integer(names(tab)[which.max(tab)])
        labels_positive[labels == -1L] <- largest
      } else {
        # all noise → assign cluster 0
        labels_positive[labels == -1L] <- 0L
      }
    }
  }

  # Stats
  n_samples <- length(sample_names)
  unique_original <- unique(labels[labels >= 0L])
  unique_positive <- unique(labels_positive)

  outs = list(
    clusters = labels,                          # may include -1 for noise
    clusters_positive = labels_positive,        # non-negative labels after handling
    silhouette_score = silhouette_score,
    sample_names = sample_names,
    n_samples = n_samples,
    n_clusters_found = length(unique_original),
    n_clusters_final = length(unique_positive),
    n_noise = n_noise,
    handle_noise = handle_noise
  )

  return(outs)
}
