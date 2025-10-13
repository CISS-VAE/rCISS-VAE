
#' Compute Per-Group and Per-Cluster Mean Squared Errors (MSE)
#'
#' Calculates mean squared error (MSE) metrics for a model's imputed validation data
#' against ground-truth validation values, with optional grouping by clusters and
#' any user-specified grouping variable (e.g., demographic or clinical categories).
#'
#' This function takes the output of `run_cissvae()` or `autotune_cissvae()`,
#' extracts validation and imputed datasets, and computes overall, per-cluster,
#' per-group, and group-by-cluster MSE summaries. It only evaluates error for
#' validation cells (non-NA values in `val_data`).
#'
#' @param res A list containing CISS-VAE run outputs. Must include:
#'   \itemize{
#'     \item \code{res$val_data}: Data frame of validation values (with \code{NA} for non-validation cells)
#'     \item \code{res$val_imputed}: Data frame of model-imputed values (same dimensions as \code{val_data})
#'     \item \code{res$clusters}: Vector of cluster labels for each row
#'     \item \code{res$raw_data}: (Optional) Original dataset, used only for consistency checks
#'   }
#' @param group_col Character scalar. Name of the column in \code{val_data} to use as the grouping variable.
#'   This can represent, for example, a patient characteristic or categorical feature.
#' @param feature_cols Optional character vector specifying which feature columns to include in MSE calculation.
#'   Defaults to all numeric columns in \code{val_data}, excluding \code{group_col} and any columns listed in \code{cols_ignore}.
#' @param by_group Logical; if \code{TRUE} (default), returns MSE summaries grouped by \code{group_col}.
#' @param by_cluster Logical; if \code{TRUE} (default), returns MSE summaries grouped by cluster labels.
#' @param cols_ignore Optional character vector of column names to exclude from scoring (e.g., IDs, indices).
#'
#' @return A named list containing data frames:
#'   \describe{
#'     \item{\code{overall}}{Overall MSE across all validation cells.}
#'     \item{\code{per_cluster}}{MSE summarized by cluster (if \code{by_cluster = TRUE}).}
#'     \item{\code{per_group}}{MSE summarized by group variable (if \code{by_group = TRUE}).}
#'     \item{\code{group_by_cluster}}{MSE summarized jointly by group and cluster (if both grouping options are TRUE).}
#'     \item{\code{per_feature_overall}}{Average MSE per feature across all rows.}
#'   }
#'
#' @details
#' The function first identifies the numeric feature columns to score and computes
#' squared errors only on validation cells (where \code{val_data} is non-NA).
#' It then aggregates those errors to calculate overall and grouped MSEs.
#'
#' Internally, it safely merges count and mean statistics using a helper
#' aggregation function (\code{.safe_aggs}) to avoid issues with list columns from
#' \code{aggregate()} output.
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' res <- run_cissvae(data, ...)
#' mse_results <- get_mse(
#'   res = res,
#'   group_col = "death_year",
#'   cols_ignore = "id",
#'   by_group = TRUE,
#'   by_cluster = TRUE
#' )
#'
#' # Access results
#' mse_results$overall
#' mse_results$per_group
#' mse_results$per_cluster
#' }
#'
#'
#'
#' @export
get_mse <- function(res,
  clusters = NULL, ## ifclusters not in result object, include them here. 
  group_col = NULL,
  feature_cols = NULL,  # default: all numeric columns excluding group_col & cols_ignore
  by_group = TRUE,
  by_cluster = TRUE,
  cols_ignore = NULL     # columns to not score
) {
  # ---- unpack ----
  raw_data   <- res$raw_data
  val_data   <- res$val_data
  if(is.null(clusters)){
    clusters   <- res$clusters
  }
  val_imputed<- res$val_imputed

  # ---- checks ----
  if (!is.data.frame(val_data) || !is.data.frame(val_imputed))
    stop("`val_data` and `val_imputed` must both be data.frames.")
  if (nrow(val_data) != nrow(val_imputed))
    stop("Row counts differ between `val_data` and `val_imputed`.")
  if (length(clusters) != nrow(val_data))
    stop("`clusters` length must match number of rows in `val_data`.")

  # group handling: allow NULL safely
  has_group <- !is.null(group_col) && group_col %in% colnames(val_data)
  if (!is.null(group_col) && !has_group) {
    stop(sprintf("group_col '%s' not found in val_data.", group_col))
  }
  # If group_col is NULL, ignore by_group requests
  if (!has_group) {
    by_group <- FALSE
  }

  # ---- feature columns to score ----
  if (is.null(feature_cols)) {
    num_cols <- names(val_data)[vapply(val_data, is.numeric, logical(1))]
    ignores <- cols_ignore
    if (has_group) {
      ignores <- c(ignores, group_col)
    }
    ignores <- unique(ignores)
    if (length(ignores)) {
      feature_cols <- setdiff(num_cols, ignores)
    } else {
      feature_cols <- num_cols
    }
  } else {
    # keep only those that exist in both frames
    feature_cols <- intersect(feature_cols, intersect(colnames(val_data), colnames(val_imputed)))
  }
  if (length(feature_cols) == 0L)
    stop("No feature columns to score after filtering. Provide `feature_cols` or ensure numeric features exist.")

  # ensure identical column order for scoring
  val_sub  <- val_data[, feature_cols, drop = FALSE]
  pred_sub <- val_imputed[, feature_cols, drop = FALSE]

  # mask: only score where we had validation targets
  used_mask <- !is.na(val_sub)

  # squared errors on used cells
  se_mat <- (as.matrix(pred_sub) - as.matrix(val_sub))^2
  se_mat[!as.matrix(used_mask)] <- NA_real_

  overall_mse <- mean(se_mat, na.rm = TRUE)
  message(sprintf("Overall MSE for validation data: %.4f", overall_mse))

  # ---- long data for aggregation ----
  df_long <- data.frame(
    row     = rep.int(seq_len(nrow(val_sub)), times = ncol(val_sub)),
    feature = rep(feature_cols, each = nrow(val_sub)),
    cluster = rep(clusters, times = ncol(val_sub)),
    se      = as.vector(se_mat),
    used    = as.vector(used_mask),
    check.names = FALSE
  )
  if (has_group) {
    df_long$group <- rep(val_data[[group_col]], times = ncol(val_sub))
  }

  # keep only scored cells
  keep_cols <- c("row", "feature", "cluster", if (has_group) "group", "se", "used")
  df_long <- df_long[ , keep_cols]
  df_long <- df_long[df_long$used & !is.na(df_long$se), setdiff(keep_cols, "used"), drop = FALSE]

  results <- list()

  # overall
  results$overall <- data.frame(
    mse = mean(df_long$se),
    n   = length(df_long$se)
  )

  # helper: safe aggregate of mean + count
  .safe_aggs <- function(data, keys) {
    # data must contain keys and 'se'
    m <- stats::aggregate(se ~ ., data = data[ , c(keys, "se"), drop = FALSE], FUN = mean)
    n <- stats::aggregate(se ~ ., data = data[ , c(keys, "se"), drop = FALSE], FUN = length)
    names(m)[names(m) == "se"] <- "mse"
    names(n)[names(n) == "se"] <- "n"
    merge(m, n, by = keys, sort = TRUE)
  }

  # by cluster
  if (by_cluster) {
    results$per_cluster <- .safe_aggs(df_long, c("cluster"))
  }

  # by group (only if group exists)
  if (by_group && has_group) {
    results$per_group <- .safe_aggs(df_long, c("group"))
  }

  # by group x cluster (only if both are requested and group exists)
  if (by_group && by_cluster && has_group) {
    results$group_by_cluster <- .safe_aggs(df_long, c("group", "cluster"))
  }


  results
}
