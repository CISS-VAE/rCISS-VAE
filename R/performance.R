#' Compute per-cluster and per-group performance metrics (MSE, BCE)
#'
#' Calculates mean squared error (MSE) for continuous features and binary
#' cross-entropy (BCE) for features you explicitly mark as binary,
#' comparing model-imputed validation values against ground-truth validation data.
#'
#' @param res A list containing CISS-VAE run outputs. Must include:
#'   \itemize{
#'     \item \code{res$val_data}: validation data frame (with \code{NA} for non-validation cells)
#'     \item \code{res$val_imputed}: model-imputed validation predictions
#'     \item \code{res$clusters}: cluster labels for each row
#'   }
#' @param clusters Optional vector (same length as rows in \code{val_data}) of cluster labels.
#'   If \code{NULL}, will use \code{res$clusters}.
#' @param group_col Optional character, name of the column in \code{val_data} for grouping.
#' @param feature_cols Character vector specifying which feature columns to evaluate. Defaults to all numeric
#'   columns except \code{group_col} and those in \code{cols_ignore}.
#' @param binary_features Character vector naming those columns (subset of \code{feature_cols}) that
#'   should use BCE instead of MSE.
#' @param by_group Logical; if \code{TRUE} (default), summarize by \code{group_col}.
#' @param by_cluster Logical; if \code{TRUE} (default), summarize by cluster.
#' @param cols_ignore Character vector of column names to exclude from scoring (e.g., “id”).
#'
#' @return A named list containing:
#'   \itemize{
#'     \item \code{overall}: overall average metric (MSE for continuous, BCE for binary)  
#'     \item \code{per_cluster}: summaries by cluster  
#'     \item \code{per_group}: summaries by group  
#'     \item \code{group_by_cluster}: summaries by group and cluster  
#'     \item \code{per_feature_overall}: average per-feature metric  
#'   }
#'
#' @details
#' For features listed in \code{binary_features}, performance is binary cross-entropy (BCE):
#' \deqn{-[y\log(p) + (1-y)\log(1-p)]}.
#' For other numeric features, performance is mean squared error (MSE).
#'
#' @example inst/examples/performance_ex.R
#' 
#' @export
performance_by_cluster <- function(
  res,
  clusters         = NULL,
  group_col        = NULL,
  feature_cols     = NULL,
  binary_features  = character(0),
  by_group         = TRUE,
  by_cluster       = TRUE,
  cols_ignore      = NULL
) {
  val_data    <- res$val_data
  val_imputed <- res$val_imputed
  if (is.null(clusters)) clusters <- res$clusters

  if (!is.data.frame(val_data) || !is.data.frame(val_imputed))
    stop("`val_data` and `val_imputed` must both be data.frames.")
  if (nrow(val_data) != nrow(val_imputed))
    stop("Row counts differ between `val_data` and `val_imputed`.")
  if (length(clusters) != nrow(val_data))
    stop("`clusters` length must match number of rows in `val_data`.")

  has_group <- !is.null(group_col) && group_col %in% names(val_data)
  if (!has_group) by_group <- FALSE

  # Determine features
  if (is.null(feature_cols)) {
    num_cols <- names(val_data)[vapply(val_data, is.numeric, logical(1))]
    ignores  <- unique(c(cols_ignore, group_col))
    feature_cols <- setdiff(num_cols, ignores)
  }
  feature_cols <- Reduce(intersect, list(feature_cols, colnames(val_data), colnames(val_imputed)))
  if (length(feature_cols) == 0L)
    stop("No feature columns available to score.")

  # Ensure binary_features subset of feature_cols
  if (!all(binary_features %in% feature_cols))
    stop("`binary_features` must be a subset of `feature_cols`.")

  val_sub  <- val_data[, feature_cols, drop = FALSE]
  pred_sub <- val_imputed[, feature_cols, drop = FALSE]
  used_mask <- !is.na(val_sub)

  # Squared error matrix
  se_mat <- (as.matrix(pred_sub) - as.matrix(val_sub))^2
  se_mat[!used_mask] <- NA_real_

  # Binary cross‐entropy for binary features -- assumes that yhat is prbabilty
  bce_mat <- matrix(NA_real_, nrow = nrow(val_sub), ncol = ncol(val_sub))
  if (length(binary_features) > 0) {
    idx <- which(colnames(val_sub) %in% binary_features)
    # define BCE function
    bce_fun <- function(y_hat, y_true) {
      eps <- 1e-7
      y_hat <- pmin(pmax(y_hat, eps), 1 - eps)
      -(y_true * log(y_hat) + (1 - y_true) * log(1 - y_hat))
    }
    bce_mat[, idx] <- bce_fun(pred_sub[, idx, drop = FALSE], val_sub[, idx, drop = FALSE])
    bce_mat[!used_mask] <- NA_real_
  }

  # Long format
  df_long <- data.frame(
    row     = rep(seq_len(nrow(val_sub)), times = ncol(val_sub)),
    feature = rep(feature_cols, each = nrow(val_sub)),
    cluster = rep(clusters, times = ncol(val_sub)),
    type    = rep(ifelse(colnames(val_sub) %in% binary_features, "binary", "continuous"), each = nrow(val_sub)),
    se      = as.vector(se_mat),
    bce     = as.vector(bce_mat),
    used    = as.vector(used_mask),
    check.names = FALSE
  )
  if (has_group) df_long$group <- rep(val_data[[group_col]], times = ncol(val_sub))

  df_long <- df_long[df_long$used & !is.na(ifelse(df_long$type == "binary", df_long$bce, df_long$se)), ]
  df_long$metric_value <- ifelse(df_long$type == "binary", df_long$bce, df_long$se)
  df_long <- df_long[, c("row", "feature", "cluster", "type", if (has_group) "group", "metric_value")]

  .safe_aggs <- function(data, keys) {
    m <- stats::aggregate(metric_value ~ ., data = data[, c(keys, "metric_value")], FUN = mean)
    n <- stats::aggregate(metric_value ~ ., data = data[, c(keys, "metric_value")], FUN = length)
    names(m)[names(m) == "metric_value"] <- "mean_imputation_loss"
    names(n)[names(n) == "metric_value"] <- "n"
    merge(m, n, by = keys, sort = TRUE)
  }

  results <- list()
  results$overall <- data.frame(
    metric = mean(df_long$metric_value, na.rm = TRUE),
    n      = sum(!is.na(df_long$metric_value))
  )
  if (by_cluster)   results$per_cluster       <- .safe_aggs(df_long, c("cluster"))
  if (by_group && has_group)   results$per_group        <- .safe_aggs(df_long, c("group"))
  if (by_group && by_cluster && has_group) results$group_by_cluster <- .safe_aggs(df_long, c("group", "cluster"))
  results$per_feature_overall <- .safe_aggs(df_long, c("feature", "type"))

  results
}
