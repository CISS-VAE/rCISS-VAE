#' Compute per-cluster and per-group performance metrics (MSE, BCE)
#'
#' Calculates mean squared error (MSE) for continuous features and binary
#' cross-entropy (BCE) for features explicitly marked as binary, comparing
#' model-imputed validation values against ground-truth validation data.
#'
#' Validation loss is computed at the cell level and then aggregated to produce
#' overall, per-cluster, per-group, and group-by-cluster summaries.
#'
#' @param res A list containing CISS-VAE run outputs. Must include:
#'   \itemize{
#'     \item \code{res$val_data}: validation data frame (with \code{NA} for non-validation cells)
#'     \item \code{res$val_imputed}: model-imputed validation predictions
#'     \item \code{res$clusters}: cluster labels for each row
#'   }
#' @param clusters Optional vector (same length as rows in \code{val_data}) of cluster labels.
#'   If \code{NULL}, \code{res$clusters} will be used.
#' @param group_col Optional character string naming a column in \code{val_data}
#'   used for grouping (e.g., sex, treatment group, etc.). If supplied,
#'   summaries can be computed per group and group-by-cluster.
#' @param feature_cols Character vector specifying which feature columns to evaluate.
#'   Defaults to all numeric columns except \code{group_col} and those in
#'   \code{cols_ignore}.
#' @param binary_features Character vector naming those columns (subset of
#'   \code{feature_cols}) that should use BCE instead of MSE.
#' @param by_group Logical; if \code{TRUE}, compute summaries by \code{group_col}.
#'   Ignored if \code{group_col} is \code{NULL}.
#' @param by_cluster Logical; if \code{TRUE}, compute summaries by cluster.
#' @param cols_ignore Character vector of column names to exclude from scoring
#'   (e.g., IDs).
#' @param eps Numeric. Small constant used for clipping probabilities in BCE
#'   calculation. Default is \code{1e-7}.
#'
#' @return A named list containing:
#'   \itemize{
#'     \item \code{overall}: overall validation metrics (MSE, BCE, total)
#'     \item \code{per_cluster}: metrics summarized by cluster (if \code{by_cluster = TRUE})
#'     \item \code{per_group}: metrics summarized by group (if \code{by_group = TRUE})
#'     \item \code{group_by_cluster}: metrics summarized by group and cluster
#'           (if both \code{by_group} and \code{by_cluster} are \code{TRUE})
#'   }
#'
#' Each summary contains:
#' \itemize{
#'   \item \code{mse}: mean squared error across continuous validation cells
#'   \item \code{bce}: mean binary cross-entropy across binary validation cells
#'   \item \code{imputation_error}: \code{mse + bce}
#' }
#' 
#' @importFrom stats reshape
#'
#' @details
#' For features listed in \code{binary_features}, performance is binary
#' cross-entropy (BCE):
#' \deqn{-[y\log(p) + (1-y)\log(1-p)]}
#' where \eqn{p} is the predicted probability.
#'
#' For other numeric features, performance is mean squared error (MSE):
#' \deqn{(y - \hat{y})^2}.
#'
#' Losses are computed at the individual cell level using only validation
#' entries (non-NA in \code{val_data}), then aggregated.
#'
#' @examples
#' data_complete <- data.frame(
#'   id = 1:10,
#'   group = sample(c("A", "B"), 10, replace = TRUE),
#'   x1 = rnorm(10),
#'   x2 = rnorm(10)
#' )
#'
#' missing_mask <- matrix(
#'   sample(c(TRUE, FALSE), 20, replace = TRUE),
#'   nrow = 10
#' )
#'
#' val_data <- data_complete
#' val_data[which(missing_mask, arr.ind = TRUE)] <- NA
#'
#' val_imputed <- data.frame(
#'   id = data_complete$id,
#'   group = data_complete$group,
#'   x1 = mean(data_complete$x1),
#'   x2 = mean(data_complete$x2)
#' )
#'
#' val_imputed[which(missing_mask, arr.ind = TRUE)] <- NA
#'
#' result <- list(
#'   val_data = val_data,
#'   val_imputed = val_imputed,
#'   clusters = sample(c(0, 1), 10, replace = TRUE)
#' )
#'
#' performance_by_cluster(
#'   res = result,
#'   group_col = "group",
#'   binary_features = character(0),
#'   by_group = TRUE,
#'   by_cluster = TRUE,
#'   cols_ignore = "id"
#' )
#'
#' @export
performance_by_cluster <- function(
  res,
  clusters        = NULL,
  group_col       = NULL,
  feature_cols    = NULL,
  binary_features = character(0),
  by_group        = TRUE,
  by_cluster      = TRUE,
  cols_ignore     = NULL,
  eps             = 1e-7
) {

  ## ------------------------------------------------------------------
  ## Extract inputs
  ## ------------------------------------------------------------------
  val_data    <- res$val_data
  val_imputed <- res$val_imputed

  if (!is.data.frame(val_data) || !is.data.frame(val_imputed))
    stop("`val_data` and `val_imputed` must be data.frames.")

  if (nrow(val_data) != nrow(val_imputed))
    stop("Row counts differ between `val_data` and `val_imputed`.")

  ## ------------------------------------------------------------------
  ## Clusters
  ## ------------------------------------------------------------------
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
  ## Group column
  ## ------------------------------------------------------------------
  has_group <- FALSE
  if (!is.null(group_col)) {
    if (!(group_col %in% colnames(val_data)))
      stop("`group_col` must be a column in `val_data`.")
    group_vec <- val_data[[group_col]]
    has_group <- TRUE
  } else {
    by_group <- FALSE
  }

  ## ------------------------------------------------------------------
  ## Feature selection
  ## ------------------------------------------------------------------
  if (is.null(feature_cols)) {
    num_cols <- names(val_data)[vapply(val_data, is.numeric, logical(1))]
    feature_cols <- setdiff(num_cols, c(cols_ignore, group_col))
  }

  feature_cols <- Reduce(intersect, list(
    feature_cols,
    colnames(val_data),
    colnames(val_imputed)
  ))

  if (!all(binary_features %in% feature_cols))
    stop("`binary_features` must be a subset of `feature_cols`.")

  ## ------------------------------------------------------------------
  ## Subset data
  ## ------------------------------------------------------------------
  val_sub  <- val_data[, feature_cols, drop = FALSE]
  pred_sub <- val_imputed[, feature_cols, drop = FALSE]

  used_mask <- !is.na(val_sub)

  ## ------------------------------------------------------------------
  ## Build long-form cell-level loss table
  ## ------------------------------------------------------------------
  out <- vector("list", length(feature_cols))

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

    tmp <- data.frame(
      cluster = clusters,
      feature = feat,
      type    = type,
      loss    = loss,
      used    = mask
    )

    if (has_group) {
      tmp$group <- group_vec
    }

    out[[j]] <- tmp
  }

  df <- do.call(rbind, out)
  df <- df[df$used & is.finite(df$loss), ]

  ## ------------------------------------------------------------------
  ## Aggregation helper
  ## ------------------------------------------------------------------
  agg <- function(keys) {
    stats::aggregate(
      loss ~ ., data = df[, c(keys, "loss"), drop = FALSE],
      FUN = mean
    )
  }

  results <- list()

  ## ------------------------------------------------------------------
  ## Overall
  ## ------------------------------------------------------------------
  mse_vals <- df$loss[df$type == "continuous"]
  bce_vals <- df$loss[df$type == "binary"]

  mse_val <- if (length(mse_vals) > 0) {
    mean(mse_vals, na.rm = TRUE)
  } else {
    NA_real_
  }

  bce_val <- if (length(bce_vals) > 0) {
    mean(bce_vals, na.rm = TRUE)
  } else {
    NA_real_
  }

  results$overall <- data.frame(
    mse = mse_val,
    bce = bce_val
  )

results$overall$imputation_error <-
  rowSums(results$overall[, c("mse", "bce"), drop = FALSE], na.rm = TRUE)

  ## ------------------------------------------------------------------
  ## Per cluster
  ## ------------------------------------------------------------------
  if (isTRUE(by_cluster)) {

    cdat <- agg(c("cluster", "type"))

    by_cluster <- reshape(
      cdat,
      idvar   = "cluster",
      timevar = "type",
      direction = "wide"
    )

    names(by_cluster) <- sub("loss.continuous", "mse", names(by_cluster))
    names(by_cluster) <- sub("loss.binary",     "bce", names(by_cluster))

    # Ensure both columns exist
    if (!"mse" %in% names(by_cluster)) by_cluster$mse <- NA_real_
    if (!"bce" %in% names(by_cluster)) by_cluster$bce <- NA_real_

    by_cluster$imputation_error <-
      rowSums(by_cluster[, c("mse", "bce"), drop = FALSE], na.rm = TRUE)

    results$per_cluster <- by_cluster
  }

  ## ------------------------------------------------------------------
  ## Per group
  ## ------------------------------------------------------------------
  if (isTRUE(by_group) && has_group) {

    gdat <- agg(c("group", "type"))

    by_group_df <- reshape(
      gdat,
      idvar   = "group",
      timevar = "type",
      direction = "wide"
    )

    names(by_group_df) <- sub("loss.continuous", "mse", names(by_group_df))
    names(by_group_df) <- sub("loss.binary",     "bce", names(by_group_df))

    # Ensure both columns exist
    if (!"mse" %in% names(by_group_df)) by_group_df$mse <- NA_real_
    if (!"bce" %in% names(by_group_df)) by_group_df$bce <- NA_real_

    by_group_df$imputation_error <-
      rowSums(by_group_df[, c("mse", "bce"), drop = FALSE], na.rm = TRUE)

    results$per_group <- by_group_df
  }

  ## ------------------------------------------------------------------
  ## Group × cluster
  ## ------------------------------------------------------------------
  if (isTRUE(by_group) && isTRUE(by_cluster) && has_group) {

    gc_dat <- agg(c("group", "cluster", "type"))

    by_gc <- reshape(
      gc_dat,
      idvar   = c("group", "cluster"),
      timevar = "type",
      direction = "wide"
    )

    names(by_gc) <- sub("loss.continuous", "mse", names(by_gc))
    names(by_gc) <- sub("loss.binary",     "bce", names(by_gc))

    # Ensure both columns exist
    if (!"mse" %in% names(by_gc)) by_gc$mse <- NA_real_
    if (!"bce" %in% names(by_gc)) by_gc$bce <- NA_real_

    by_gc$imputation_error <-
      rowSums(by_gc[, c("mse", "bce"), drop = FALSE], na.rm = TRUE)

    results$group_by_cluster <- by_gc
  }

  results
}