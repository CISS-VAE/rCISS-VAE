#' Cluster on Missingness Patterns
#'
#' Given an R data.frame or matrix with missing values, clusters on the pattern
#' of missingness and returns cluster labels plus silhouette score.
#'
#' @param data A data.frame or matrix (samples × features), may contain `NA`.
#' @param cols_ignore Character vector of column names to ignore when clustering.
#' @param n_clusters Integer; if provided, will run KMeans with this many clusters.
#'                   If `NULL`, will use Leiden.
#' @param seed Integer; random seed for KMeans (or reproducibility in Leiden).
#' @param k_neighbors Integer; minimum cluster size for Leiden.
#'                           If `NULL`, defaults to `nrow(data) %/% 25`.
#' @param leiden_resolution Resolution for Leiden Clustering.
#' @param leiden_objective objective
#' @param use_snn use snn
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
#' Groups **samples** with similar patterns of missingness across features using
#' either K-means clustering (when `n_clusters` is specified) or Leiden
#' (when `n_clusters` is `NULL`). This is useful for detecting cohorts with
#' shared missing-data behavior (e.g., site/batch effects).
#'
#' @param prop_matrix Matrix or data frame where **rows are samples** and
#'   **columns are features**, entries are missingness proportions in `[0,1]`.
#'   Can be created with `create_missingness_prop_matrix()`.
#' @param n_clusters Integer; number of clusters for KMeans. If `NULL`, uses
#'   Leiden (default: `NULL`).
#' @param seed Integer; random seed for KMeans reproducibility (default: `NULL`).
#' @param k_neighbors Integer; Leiden minimum cluster size. If `NULL`, Python
#'   default is used (default: `NULL`).
#' @param leiden_resolution Numeric; Leiden cluster selection threshold
#'   (default: `0.25`).
#' @param metric Character; distance metric. Options include:
#'   \code{
#'     "euclidean",
#'     "cosine"
#'   }
#'   (default: `"euclidean"`).
#' @param scale_features Logical; whether to standardize **feature columns**
#'   before clustering samples (default: `FALSE`).
#' @param leiden_objective Character; Leiden optimization objective (optional).
#' @param use_snn Logical; whether to use shared nearest neighbors (optional).
#'
#' @return A list with:
#' \itemize{
#'   \item \code{clusters}: Integer vector of cluster assignments per **sample**.
#'   \item \code{silhouette_score}: Numeric silhouette score, or \code{NULL}
#'     if not computable.
#' }
#'
#' @examples
#' set.seed(123)
#'
#' dat <- data.frame(
#'   sample_id = paste0("s", 1:12),
#'   # Two features measured at 3 timepoints each -> proportions by feature
#'   A_1 = c(NA, rnorm(11)),
#'   A_2 = c(NA, rnorm(11)),
#'   A_3 = rnorm(12),
#'   B_1 = rnorm(12),
#'   B_2 = c(rnorm(10), NA, NA),
#'   B_3 = rnorm(12)
#' )
#'
#' pm <- create_missingness_prop_matrix(
#'   dat,
#'   index_col = "sample_id",
#'   repeat_feature_names = c("A", "B")
#' )
#'
#' try({
#' res <- cluster_on_missing_prop(
#'   pm,
#'   n_clusters = 2,
#'   metric = "cosine",
#'   scale_features = TRUE
#' )
#'
#' table(res$clusters)
#' res$silhouette_score
#' })
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
  scale_features = FALSE
) {
  # Dependencies
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it to use this function.")
  }

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

  

  # Stats
  n_samples <- length(sample_names)
  unique_original <- unique(labels[labels >= 0L])


  outs = list(
    clusters = labels,                          # may include -1 for noise
         # non-negative labels after handling
    silhouette_score = silhouette_score,
    
    n_samples = n_samples,
    n_clusters_found = length(unique_original)
  )

  return(outs)
}
