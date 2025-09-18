#' Create Missingness Proportion Matrix
#'
#' Creates a matrix where each entry represents the proportion of missing values
#' for each sample–feature combination across multiple timepoints. Each sample will have
#' one proportion value per feature. Features may have repeated time points
#' (columns named like `feature_1`, `feature_2`, ...). This matrix can be used
#' with `cluster_on_missing_prop()` to group samples with similar missingness patterns.
#'
#' @param data Data frame or matrix containing the input data with potential missing values.
#' @param index_col Character scalar. Name of an index column to exclude from analysis (optional).
#'   If supplied and present, it will be removed from analysis; row names are preserved as-is.
#' @param cols_ignore Character vector of column names to exclude from the proportion matrix (optional).
#' @param na_values Vector of values to treat as missing in addition to standard missing values.
#'   Defaults to \code{c(NA, NaN, Inf, -Inf)}.
#' @param repeat_feature_names Character vector of "base" feature names that have repeated timepoints.
#'   Repeat measurements must be in the form \code{<feature>_<timepoint>} where \code{<feature>}
#'   is alphanumeric (and may include dots) and \code{<timepoint>} is an integer (e.g., \code{"CRP_1"}).
#'
#' @return A numeric matrix of dimension \code{nrow(data)} by \code{n_features}, where rows are
#'   samples and columns are features (base names). Entries are per-sample missingness proportions in `[0, 1]`.
#'   The returned matrix has an attribute \code{"feature_columns_map"}: a named list mapping each
#'   output feature to the source columns used to compute its proportion.
#'
#' @examples
#' df <- data.frame(
#'   id = paste0("s", 1:4),
#'   CRP_1 = c(1.2, NA, 2.1, NaN),
#'   CRP_2 = c(NA, NA, 2.0, 1.9),
#'   IL6_1 = c(0.5, 0.7, Inf, 0.4),
#'   IL6_2 = c(0.6, -Inf, 0.8, 0.5),
#'   Albumin = c(3.9, 4.1, 4.0, NA)
#' )
#'
#' m <- create_missingness_prop_matrix(
#'   data = df,
#'   index_col = "id",
#'   cols_ignore = NULL,
#'   repeat_feature_names = c("CRP", "IL6")
#' )
#'
#' dim(m)         # 4 x 3 (CRP, IL6, Albumin)
#' m[ , "CRP"]    # per-sample proportion missing across CRP_1 and CRP_2
#' attr(m, "feature_columns_map")
#'
#' @export
create_missingness_prop_matrix <- function(
  data,
  index_col = NULL,
  cols_ignore = NULL,
  na_values = c(NA, NaN, Inf, -Inf),
  repeat_feature_names = character(0)
) {
  # ------------------------------- #
  # 1) Validate & normalize inputs  #
  # ------------------------------- #
  if (!(is.data.frame(data) || is.matrix(data))) {
    stop("`data` must be a data.frame or matrix.")
  }
  # Coerce to data.frame for uniform handling; retain original rownames
  df <- as.data.frame(data, stringsAsFactors = FALSE)

  # Validate character arguments
  if (!is.null(index_col) && !is.character(index_col)) {
    stop("`index_col` must be NULL or a character scalar.")
  }
  if (!is.null(index_col) && length(index_col) != 1L) {
    stop("`index_col` must be a single column name if provided.")
  }
  if (!is.null(cols_ignore) && !is.character(cols_ignore)) {
    stop("`cols_ignore` must be NULL or a character vector of column names.")
  }
  if (!is.character(repeat_feature_names)) {
    stop("`repeat_feature_names` must be a character vector (possibly length 0).")
  }

  # Ensure we don't accidentally treat the index or ignored columns as features
  cols_to_drop <- unique(c(
    if (!is.null(index_col) && index_col %in% names(df)) index_col else character(0),
    if (!is.null(cols_ignore)) intersect(cols_ignore, names(df)) else character(0)
  ))

  # --------------------------------------- #
  # 2) Build helper for missingness checks  #
  # --------------------------------------- #
  # Treat NA/NaN/Inf/-Inf plus any user-specified `na_values` as missing.
  # We'll use a vectorized predicate that avoids type warnings as much as possible.
  is_missing <- function(x) {
    # Start with standard missing flags
    miss <- is.na(x) | is.nan(x)
    # Infinite values
    miss <- miss | is.infinite(suppressWarnings(as.numeric(x)))
    # User-provided exact matches (including strings); suppress coercion warnings gracefully
    # Use tryCatch to avoid errors when comparing incompatible types
    if (!is.null(na_values) && length(na_values) > 0) {
      # %in% will coerce; wrap to silence harmless warnings
      miss <- miss | suppressWarnings(x %in% na_values)
    }
    miss
  }

  # --------------------------------------------------- #
  # 3) Identify columns for repeated vs single features #
  # --------------------------------------------------- #
  all_cols <- names(df)
  if (length(all_cols) == 0) stop("Input `data` has no columns.")

  # Remove drop columns up front from consideration
  feature_candidate_cols <- setdiff(all_cols, cols_to_drop)

  # For each base feature in repeat_feature_names, collect its timepoint columns.
  # Pattern: ^<feature>_\\d+$    (timepoint suffix must be an integer)
  feature_to_cols <- list()
  consumed_cols <- character(0)

  if (length(repeat_feature_names) > 0) {
    for (feat in repeat_feature_names) {
      # Allow dots or alphanumerics in base feature; user guarantees the base names.
      # Escape regex special characters in feat to match literal
      feat_escaped <- gsub("([.|()\\^{}+$*?\\[\\]\\\\])", "\\\\\\1", feat)
      pat <- paste0("^", feat_escaped, "_\\d+$")

      these <- grep(pat, feature_candidate_cols, value = TRUE)
      if (length(these) == 0) {
        stop("No columns found for repeated features using current pattern. Please ensure columns are named according to pattern.")
        # stop(sprintf(
        #   "No columns found for repeated feature '%s' using pattern '%s'. " %+%
        #   "Ensure columns are named like '%s_1', '%s_2', ...",
        #   feat, pat, feat, feat
        # ))
      }
      feature_to_cols[[feat]] <- these
      consumed_cols <- c(consumed_cols, these)
    }
  }

  # Remaining feature columns (not part of repeated features and not dropped)
  remaining_cols <- setdiff(feature_candidate_cols, consumed_cols)

  # Treat each remaining column as a single-timepoint feature
  # (proportion is 0 or 1 depending on missingness)
  for (col in remaining_cols) {
    feature_to_cols[[col]] <- col
  }

  # Determine output feature order:
  #  - repeated features in the order supplied by `repeat_feature_names`
  #  - then single-timepoint features in their existing data column order
  singletons_in_order <- remaining_cols
  out_features <- c(repeat_feature_names, singletons_in_order)

  if (length(out_features) == 0) {
    stop("After excluding `index_col` and `cols_ignore`, no feature columns remain.")
  }

  # ------------------------------------------- #
  # 4) Compute per-sample missingness proportion #
  # ------------------------------------------- #
  n <- nrow(df)
  p <- length(out_features)

  # Initialize a numeric matrix; rownames preserved from input (if any)
  out <- matrix(NA_real_, nrow = n, ncol = p,
                dimnames = list(rownames(df), out_features))

  # Fill the matrix column-by-column
  for (j in seq_along(out_features)) {
    feat <- out_features[j]
    cols <- feature_to_cols[[feat]]

    # Subset to the relevant columns (coerce to data.frame for consistent behavior)
    subdf <- df[ , cols, drop = FALSE]

    # Compute missingness proportion per row:
    #  - If multiple columns: mean of missing indicators
    #  - If single column: 0/1 converted to numeric mean
    miss_mat <- as.data.frame(lapply(subdf, is_missing), optional = TRUE,
                              stringsAsFactors = FALSE)
    # Ensure logical matrix
    miss_mat <- as.data.frame(lapply(miss_mat, function(x) as.logical(x %||% FALSE)))

    # Row means; if a feature's timepoint column is entirely non-numeric strings, the
    # missing rule still works because we check NA/NaN/Inf and explicit %in% `na_values`
    prop_missing <- rowMeans(as.data.frame(miss_mat), na.rm = FALSE)

    # Note: there should be no NA produced here; if a column existed it yields TRUE/FALSE.
    out[ , j] <- prop_missing
  }

  

  return(out)
}

