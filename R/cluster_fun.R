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
