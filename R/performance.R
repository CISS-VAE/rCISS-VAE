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
#' @param eps Optional eps for calculating BCE. Default 1e-7
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
#' @examples
#' library(tidyverse)
#' library(reticulate)
#' library(rCISSVAE)
#' library(kableExtra)
#' library(gtsummary)
#' 
#' ## Make example results
#' data_complete = data.frame(
#' index = 1:10,
#' x1 = rnorm(10),
#' x2 = rnorm(10)*rnorm(10, mean = 50, sd=10)
#'  )
#' 
#' missing_mask = matrix(data = c(rep(FALSE, 10), 
#' sample(c(TRUE, FALSE), 
#' size = 20, replace = TRUE, 
#' prob = c(0.7, 0.3))), nrow = 10)
#' 
#' ## Example validation dataset
#' val_data = data_complete
#' val_data[missing_mask] <- NA
#' 
#' ## Example 'imputed' validation dataset
#' val_imputed = data.frame(index = 1:10, x1 = mean(data_complete$x1), x2 = mean(data_complete$x2))
#' val_imputed[missing_mask] <- NA
#' 
#' ## Example result list
#' result = list("val_data" = val_data, "val_imputed" = val_imputed)
#' clusters = sample(c(0, 1), size = 10, replace = TRUE)
#' 
#' ## Run the function
#' performance_by_cluster(res = result, 
#'   group_col = NULL, 
#'   clusters = clusters,
#'   feature_cols = NULL, 
#'   by_cluster = TRUE,
#'   cols_ignore = c("index") 
#' )
#' @export
performance_by_cluster <- function(
  res,
  clusters        = NULL,
  group_col  = NULL,
  feature_cols    = NULL,
  binary_features = character(0),
  by_group = FALSE, ## no longer needed
  by_cluster      = TRUE,
  cols_ignore     = NULL,
  eps             = 1e-7
) {

  ## ------------------------------------------------------------------
  ## Extract and validate inputs
  ## ------------------------------------------------------------------
  val_data    <- res$val_data
  val_imputed <- res$val_imputed

  if (!is.data.frame(val_data) || !is.data.frame(val_imputed))
    stop("`val_data` and `val_imputed` must be data.frames.")

  if (nrow(val_data) != nrow(val_imputed))
    stop("Row counts differ between `val_data` and `val_imputed`.")

  ## Clusters
  if (is.null(clusters)) {
    if (!is.null(res$clusters)) {
      clusters <- res$clusters
    } else {
      stop("Clusters must be provided via `clusters` or `res$clusters`.")
    }
  }
  if (length(clusters) != nrow(val_data))
    stop("`clusters` length must match number of rows.")

  ## ------------------------------------------------------------------
  ## Feature selection
  ## ------------------------------------------------------------------
  if (is.null(feature_cols)) {
    num_cols <- names(val_data)[vapply(val_data, is.numeric, logical(1))]
    feature_cols <- setdiff(num_cols, cols_ignore)
  }

  feature_cols <- Reduce(intersect, list(
    feature_cols,
    colnames(val_data),
    colnames(val_imputed)
  ))

  if (!all(binary_features %in% feature_cols))
    stop("`binary_features` must be a subset of `feature_cols`.")

  if (!is.null(group_col)) {
    if (!all(group_col %in% feature_cols))
      stop("All `group_col` must be contained in `feature_cols`.")
  }

  ## ------------------------------------------------------------------
  ## Subset data and build validation mask
  ## ------------------------------------------------------------------
  val_sub  <- val_data[, feature_cols, drop = FALSE]
  pred_sub <- val_imputed[, feature_cols, drop = FALSE]

  used_mask <- !is.na(val_sub)

  ## ------------------------------------------------------------------
  ## Build long-form cell-level loss table
  ## ------------------------------------------------------------------
  out <- vector("list", length(feature_cols))
  names(out) <- feature_cols

  for (j in seq_along(feature_cols)) {

    feat <- feature_cols[j]
    y    <- val_sub[[feat]]
    yhat <- pred_sub[[feat]]
    mask <- used_mask[, j]

    if (feat %in% binary_features) {
      y[is.na(y)] <- 0
      yhat <- pmin(pmax(yhat, eps), 1 - eps)
      loss <- -(y * log(yhat) + (1 - y) * log(1 - yhat))
      type <- "binary"
    } else {
      loss <- (yhat - y)^2
      type <- "continuous"
    }

    out[[j]] <- data.frame(
      cluster = clusters,
      feature = feat,
      type    = type,
      loss    = loss,
      used    = mask
    )
  }

  df <- do.call(rbind, out)
  df <- df[df$used & is.finite(df$loss), ]

  ## Restrict to selected features if requested
  if (!is.null(group_col)) {
    df <- df[df$feature %in% group_col, ]
  }

  ## ------------------------------------------------------------------
  ## Aggregation helper
  ## ------------------------------------------------------------------
  agg <- function(keys) {
    stats::aggregate(
      loss ~ ., data = df[, c(keys, "loss"), drop = FALSE],
      FUN = mean
    )
  }

  ## ------------------------------------------------------------------
  ## Results
  ## ------------------------------------------------------------------
  results <- list()

  ## Overall
  results$overall <- data.frame(
    mse = mean(df$loss[df$type == "continuous"], na.rm = TRUE),
    bce = mean(df$loss[df$type == "binary"],     na.rm = TRUE)
  )
  results$overall$imputation_error <-
    results$overall$mse + results$overall$bce

  ## By cluster
  if (by_cluster) {

    cdat <- agg(c("cluster", "type"))

    by_cluster <- reshape(
      cdat,
      idvar   = "cluster",
      timevar = "type",
      direction = "wide"
    )

    ## Rename columns
    names(by_cluster) <- sub("loss.continuous", "mse", names(by_cluster))
    names(by_cluster) <- sub("loss.binary",     "bce", names(by_cluster))

    by_cluster$imputation_error <-
      rowSums(by_cluster[, c("mse", "bce"), drop = FALSE], na.rm = TRUE)

    results$by_cluster <- by_cluster
  }

  ## By cluster × selected feature(s)
  if (isTRUE(by_cluster) && !is.null(group_col)) {

    cfdat <- agg(c("cluster", "feature", "type"))

    by_cluster_feature <- reshape(
      cfdat,
      idvar   = c("cluster", "feature"),
      timevar = "type",
      direction = "wide"
    )

    ## Rename columns
    names(by_cluster_feature) <- sub("loss.continuous", "mse", names(by_cluster_feature))
    names(by_cluster_feature) <- sub("loss.binary",     "bce", names(by_cluster_feature))

    by_cluster_feature$imputation_error <-
      rowSums(by_cluster_feature[, c("mse", "bce"), drop = FALSE], na.rm = TRUE)

    results$by_cluster_feature <- by_cluster_feature
  }

  results
}
