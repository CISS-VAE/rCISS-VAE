#' Cluster-wise summary table using a separate cluster vector (gtsummary + gt)
#'
#' @description
#' Produce a cluster-stratified summary table using **gtsummary**, where the
#' cluster assignments are supplied as a separate vector.
#' All additional arguments (`...`) are passed directly to
#' [gtsummary::tbl_summary()], so users can specify
#' `all_continuous()` / `all_categorical()` selectors and custom statistics.
#'
#'
#' @param data A data.frame or tibble of features to summarize.
#' @param clusters A vector (factor, character, or numeric) of cluster labels
#'   with length equal to `nrow(data)`.
#' @param add_options List of post-processing options:
#'   - `add_overall` (default `FALSE`): add overall column
#'   - `add_n`       (default `TRUE`) : add group Ns
#'   - `add_p`       (default `FALSE`): add p-values
#' @param return_as `"gtsummary"` (default) or `"gt"`. When `"gt"`, the function
#'   calls [gtsummary::as_gt()] for rendering.
#' @param include Optional character vector of variables to include.
#'   Defaults to all columns in `data`.
#' @param ... Passed to [gtsummary::tbl_summary()] (e.g., `statistic=`,
#'   `type=`, `digits=`, `missing=`, `label=`, etc.).
#'
#' @return A `gtsummary::tbl_summary` (default) or `gt::gt_tbl` if `return_as="gt"`.
#'
#' @examples
#' if(requireNamespace("gtsummary")){
#' df <- data.frame(
#'   age = rnorm(100, 60, 10),
#'   bmi = rnorm(100, 28, 5),
#'   sex = sample(c("F","M"), 100, TRUE)
#' )
#' cl <- sample(1:3, 100, TRUE)
#'
#' cluster_summary(
#'   data = df,
#'   clusters = cl,
#'   statistic = list(
#'     gtsummary::all_continuous()  ~ "{mean} ({sd})",
#'     gtsummary::all_categorical() ~ "{n} / {N} ({p}%)"
#'   ),
#'   missing = "always"
#' )
#' }
#'
#' @importFrom gtsummary tbl_summary add_overall add_n add_p as_gt modify_header all_stat_cols
#' @importFrom rlang quo_is_null enquo expr sym call2 list2 eval_tidy
#' @export
cluster_summary <- function(
  data,
  clusters,
  add_options = list(add_overall = FALSE, add_n = TRUE, add_p = FALSE),
  return_as = c("gtsummary", "gt"),
  include = NULL,
  ...
) {
  # ----------
  # Ensure that data is of type dataframe or tibble
  # Ensure that length of clusters = nrow(datA)
  # ----------
  if (!is.data.frame(data))
    stop("`data` must be a data.frame or tibble.", call. = FALSE)
  if (length(clusters) != nrow(data))
    stop("Length of `clusters` must equal nrow(data).", call. = FALSE)

  # --------------------------- 
  # Temporary cluster column
  # ----------------------
  data2 <- data
  tmp_by_col <- "..cluster.."
  while (tmp_by_col %in% names(data2)) tmp_by_col <- paste0(tmp_by_col, "_")
  data2[[tmp_by_col]] <- clusters

  # --------------------------- 
  # Capture arguments 
  # ---------------------------
  dots <- rlang::list2(...)

  # Handle include
  include_quo <- rlang::enquo(include)
  include_arg <- if (!rlang::quo_is_null(include_quo)) {
    include_quo
  } else {
    rlang::quo(setdiff(names(data2), !!tmp_by_col))
  }

  # --------------------------- 
  # Build tbl_summary 
  # ---------------------------
  call <- rlang::call2(
    gtsummary::tbl_summary,
    data = rlang::expr(data2),
    by   = rlang::sym(tmp_by_col),
    include = include_arg,
    !!!dots
  )
  tbl <- rlang::eval_tidy(call)

  # --------------------------- 
  # Add-ons 
  # ---------------------------
  if (isTRUE(is.list(add_options) && isTRUE(add_options$add_overall)))
    tbl <- gtsummary::add_overall(tbl)
  if (isTRUE(is.list(add_options) && isTRUE(add_options$add_n)))
    tbl <- gtsummary::add_n(tbl)
  if (isTRUE(is.list(add_options) && isTRUE(add_options$add_p)))
    tbl <- gtsummary::add_p(tbl)


  # --------------------------- 
  # Return type 
  # ---------------------------
  return_as <- match.arg(return_as)
  if (identical(return_as, "gt")) {
    return(gtsummary::as_gt(tbl))
  } else {
    return(tbl)
  }
}

#' Cluster-wise Heatmap of Missing Data Patterns
#'
#' @description
#' Visualize the pattern of missing values in a dataset, arranged by cluster. Each
#' column in the heatmap represents one observation and each row a feature. Tiles
#' indicate whether a value is missing (black) or present (white). Cluster labels
#' are shown as a column annotation bar above the heatmap. The package
#' \pkg{ComplexHeatmap} must be installed for this function to work.
#'
#' @param data A \code{data.frame} or tibble containing the dataset with possible
#'   missing values. Rows represent observations and columns represent features.
#' @param clusters A vector of cluster labels for each observation (row) in
#'   \code{data}. Must have the same length as \code{nrow(data)}.
#' @param cols_ignore Optional character vector of column names in \code{data} to
#'   exclude from the heatmap (e.g., identifiers or non-feature columns).
#' @param show_row_names Logical. If TRUE, displays feature names on plot
#' @param missing_color Display color of missing values. Default black.
#' @param observed_color Display color of observed values. Default white. 
#' @param title Optional plot title. Defaults to "Missingness Heatmap by Cluster"
#'
#' @details
#' This function constructs a binary missingness matrix where 1 indicates a
#' missing value and 0 a present value. Columns (observations) are ordered by
#' their cluster labels, and the function displays a heatmap of missingness
#' patterns using \pkg{ComplexHeatmap}. Cluster membership is displayed as an
#' annotation above the heatmap.
#'
#' @return A list of class \code{"ComplexHeatmap"} containing the heatmap
#'   object. This can be used for further inspection or manual redraw.
#'
#' @examples
#' if(requireNamespace("ComplexHeatmap")){
#' # Simple example with small dataset
#' df <- data.frame(
#'   x1 = c(1, NA, 3),
#'   x2 = c(NA, 2, 3),
#'   x3 = c(1, 2, NA)
#' )
#' cl <- c("A", "B", "A")
#' cluster_heatmap(df, cl)
#'
#' # Example excluding a column prior to plotting
#' cluster_heatmap(df, cl, cols_ignore = "x2")
#' }
#' 
#' @importFrom ComplexHeatmap HeatmapAnnotation anno_block Heatmap
#' @export
cluster_heatmap <- function(data, clusters, cols_ignore = NULL, show_row_names = TRUE,
missing_color = "black", observed_color = "white", title = "Missingness Heatmap by Cluster") {

  if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) {
    stop(
      "The package 'ComplexHeatmap' is required but not installed. ",
      "Please install it with BiocManager::install('ComplexHeatmap').",
      call. = FALSE
    )
  }

  if (!is.data.frame(data)) {
    stop("`data` must be a data.frame or tibble.", call. = FALSE)
  }
  if (length(clusters) != nrow(data)) {
    stop("Length of `clusters` must equal nrow(data).", call. = FALSE)
  }

  if (!is.null(cols_ignore) && length(cols_ignore) > 0) {
    data <- dplyr::select(data, -dplyr::any_of(cols_ignore))
  }

  # Missingness matrix: features x observations
  missing_mat <- is.na(data)
  missing_mat <- 1L * missing_mat
  missing_mat <- t(as.matrix(missing_mat))

  rownames(missing_mat) <- colnames(data)
  colnames(missing_mat) <- paste0("Obs_", seq_len(ncol(missing_mat)))

  # Order by cluster
  ord <- order(clusters)
  missing_mat <- missing_mat[, ord, drop = FALSE]
  clusters_ord <- clusters[ord]

  # Define cluster slices
  cluster_factor <- factor(clusters_ord, levels = unique(clusters_ord))
  slice_lengths <- table(cluster_factor)

  # Top annotation with centered cluster labels
  top_anno <- ComplexHeatmap::HeatmapAnnotation(
    Cluster = ComplexHeatmap::anno_block(
      gp = grid::gpar(fill = NA, col = NA),
      labels = names(slice_lengths),
      labels_gp = grid::gpar(fontsize = 12, fontface = "bold")
    ),
    show_annotation_name = TRUE,
    annotation_name_side = "left"
  )

  hm <- ComplexHeatmap::Heatmap(
    missing_mat,
    name = "Missing",
    col = c("0" = observed_color, "1" = missing_color),
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    show_column_names = FALSE,
    show_row_names = show_row_names,
    top_annotation = top_anno,
    column_split = cluster_factor,
    column_title = title,
    heatmap_legend_param = list(
      at = c(0, 1),
      labels = c("Observed", "Missing")
    )
  )

  return(hm)
}
