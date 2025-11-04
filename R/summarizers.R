#' Cluster-wise summary table using a separate cluster vector (gtsummary + gt)
#'
#' @description
#' Produce a cluster-stratified summary table using **gtsummary**, where the
#' cluster assignments are supplied as a separate vector (not a column in `data`).
#' All additional arguments (`...`) are passed directly to
#' [gtsummary::tbl_summary()], so users can specify
#' `all_continuous()` / `all_categorical()` selectors and custom statistics.
#'
#' The resulting table automatically labels each cluster column with the actual
#' cluster levels (e.g., "1", "2", "A", "B") instead of the generic "Cluster¹".
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
#' @param include Optional tidyselect or character vector of variables to include.
#'   Defaults to all columns in `data`.
#' @param ... Passed to [gtsummary::tbl_summary()] (e.g., `statistic=`,
#'   `type=`, `digits=`, `missing=`, `label=`, etc.).
#'
#' @return A `gtsummary::tbl_summary` (default) or `gt::gt_tbl` if `return_as="gt"`.
#'
#' @examples
#' \dontrun{
#' df <- tibble::tibble(
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
  # --------------------------- Validation ------------------------------------
  if (!is.data.frame(data))
    stop("`data` must be a data.frame or tibble.", call. = FALSE)
  if (length(clusters) != nrow(data))
    stop("Length of `clusters` must equal nrow(data).", call. = FALSE)

  # --------------------------- Temporary cluster column ----------------------
  data2 <- data
  tmp_by_col <- "..cluster.."
  while (tmp_by_col %in% names(data2)) tmp_by_col <- paste0(tmp_by_col, "_")
  data2[[tmp_by_col]] <- clusters

  # --------------------------- Capture arguments -----------------------------
  dots <- rlang::list2(...)

  # Handle include: allow tidyselect or default to all columns
  include_quo <- rlang::enquo(include)
  include_arg <- if (!rlang::quo_is_null(include_quo)) {
    include_quo
  } else {
    rlang::quo(setdiff(names(data2), !!tmp_by_col))
  }

  # --------------------------- Build tbl_summary -----------------------------
  call <- rlang::call2(
    gtsummary::tbl_summary,
    data = rlang::expr(data2),
    by   = rlang::sym(tmp_by_col),
    include = include_arg,
    !!!dots
  )
  tbl <- rlang::eval_tidy(call)

  # --------------------------- Add-ons ---------------------------------------
  if (isTRUE(is.list(add_options) && isTRUE(add_options$add_overall)))
    tbl <- gtsummary::add_overall(tbl)
  if (isTRUE(is.list(add_options) && isTRUE(add_options$add_n)))
    tbl <- gtsummary::add_n(tbl)
  if (isTRUE(is.list(add_options) && isTRUE(add_options$add_p)))
    tbl <- gtsummary::add_p(tbl)

  # --------------------------- Replace cluster headers -----------------------
  # determine the label order
  # if (is.factor(clusters)) {
  #   cl_levels <- levels(stats::droplevels(clusters))
  # } else {
  #   cl_levels <- unique(as.character(clusters))
  # }
  # cl_levels <- as.character(sort(unique(as.character(clusters))))

  # # Build modify_header() call directly with dynamic dots
  # tbl <- gtsummary::modify_header(
  #   tbl,
  #   gtsummary::all_stat_cols() ~ paste0("**", cl_levels, "**")
  # )



  # --------------------------- Return type -----------------------------------
  return_as <- match.arg(return_as)
  if (identical(return_as, "gt")) {
    return(gtsummary::as_gt(tbl))
  } else {
    return(tbl)
  }
}
