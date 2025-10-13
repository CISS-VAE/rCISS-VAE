#' Cluster Summary Table (wide)
#'
#' Summarize continuous and categorical variables overall and by cluster.
#' For continuous variables, choose "mean_sd", "median_iqr", or "both".
#'
#' @param data data.frame; rows are samples
#' @param clusters vector; cluster labels (length = nrow(data))
#' @param index_col character or NULL; column to drop from `data` (e.g., ID)
#' @param categorical_vars character or NULL; vars treated as categorical
#' @param continuous_vars character or NULL; vars treated as continuous
#' @param digits integer; rounding for numeric summaries
#' @param show_overall logical; include an "Overall" column
#' @param na_label character; label for missing factor levels in categorical summaries
#' @param cont_style one of c("mean_sd","median_iqr","both")
#'
#' @return tibble with one row per variable (and each level for categoricals)
#'         and one column per Overall/Cluster.
#' @export
cluster_summary <- function(
  data,
  clusters,
  index_col = NULL,
  categorical_vars = NULL,
  continuous_vars = NULL,
  digits = 2,
  show_overall = TRUE,
  na_label = "Missing",
  cont_style = c("mean_sd", "median_iqr", "both")
) {
  cont_style <- match.arg(cont_style)

  # ---- input checks ----
  if (!is.data.frame(data)) stop("`data` must be a data.frame")
  if (nrow(data) != length(clusters)) {
    stop("Length of `clusters` must equal number of rows in `data`.")
  }

  # optional index col removal
  if (!is.null(index_col)) {
    if (!index_col %in% colnames(data)) {
      stop(sprintf("index_col '%s' not found in data.", index_col))
    }
    data <- dplyr::select(data, -dplyr::all_of(index_col))
  }

  # add cluster column (as character)
  data$cluster <- paste0("Cluster ", as.character(clusters))
  cluster_levels <- sort(unique(data$cluster))

  # ---- variable specification / auto-detect ----
  available_vars <- setdiff(colnames(data), "cluster")

  if (is.null(categorical_vars) && is.null(continuous_vars)) {
    categorical_vars <- data |>
      dplyr::select(-"cluster") |>
      dplyr::select(dplyr::where(\(x) is.factor(x) || is.character(x) || is.logical(x))) |>
      colnames()
    continuous_vars <- data |>
      dplyr::select(-"cluster", -dplyr::all_of(categorical_vars)) |>
      dplyr::select(dplyr::where(is.numeric)) |>
      colnames()
  } else if (is.null(categorical_vars)) {
    # user specified continuous → infer categorical from the rest that are non-numeric
    categorical_vars <- setdiff(available_vars, continuous_vars)
    categorical_vars <- intersect(
      categorical_vars,
      data |>
        dplyr::select(dplyr::where(\(x) is.factor(x) || is.character(x) || is.logical(x))) |>
        colnames()
    )
  } else if (is.null(continuous_vars)) {
    # user specified categorical → infer continuous from remaining numeric
    continuous_vars <- setdiff(available_vars, categorical_vars)
    continuous_vars <- intersect(
      continuous_vars,
      data |>
        dplyr::select(dplyr::where(is.numeric)) |>
        colnames()
    )
  }

  # sanity checks on specified names
  if (!is.null(categorical_vars)) {
    missing_cat <- setdiff(categorical_vars, colnames(data))
    if (length(missing_cat) > 0) {
      stop("Categorical variables not found in data: ", paste(missing_cat, collapse = ", "))
    }
  }
  if (!is.null(continuous_vars)) {
    missing_cont <- setdiff(continuous_vars, colnames(data))
    if (length(missing_cont) > 0) {
      stop("Continuous variables not found in data: ", paste(missing_cont, collapse = ", "))
    }
  }

  # ensure continuous are numeric; demote to categorical otherwise
  for (v in continuous_vars) {
    if (!is.numeric(data[[v]])) {
      warning(sprintf("Variable '%s' is not numeric. Moving to categorical.", v))
      categorical_vars <- unique(c(categorical_vars, v))
    }
  }
  continuous_vars <- intersect(continuous_vars, names(data)[vapply(data, is.numeric, TRUE)])

  # ---- helpers ----
  fmt_mean_sd <- function(x, digits = 2) {
    m <- mean(x, na.rm = TRUE); s <- stats::sd(x, na.rm = TRUE)
    paste0(round(m, digits), " (", round(s, digits), ")")
  }
  fmt_median_iqr <- function(x, digits = 2) {
    med <- stats::median(x, na.rm = TRUE)
    q1  <- stats::quantile(x, 0.25, na.rm = TRUE, type = 7)
    q3  <- stats::quantile(x, 0.75, na.rm = TRUE, type = 7)
    paste0(round(med, digits), " [", round(q1, digits), ", ", round(q3, digits), "]")
  }
  format_cont <- function(x, digits = 2, style = "mean_sd") {
    if (!is.numeric(x)) return("Invalid")
    n_total    <- length(x)
    n_missing  <- sum(is.na(x))
    n_complete <- n_total - n_missing
    if (n_complete == 0) {
      return(paste0("—, ", n_missing, " (", round(100 * n_missing / n_total, 1), "%) missing"))
    }
    pieces <- switch(
      style,
      mean_sd    = fmt_mean_sd(x, digits),
      median_iqr = fmt_median_iqr(x, digits),
      both       = paste0(fmt_mean_sd(x, digits), "; ", fmt_median_iqr(x, digits))
    )
    if (n_missing > 0) {
      pieces <- paste0(pieces, ", ", n_missing, " (", round(100 * n_missing / n_total, 1), "%) missing")
    }
    pieces
  }

  format_cat <- function(x, var_name) {
    # factor w/ explicit NA level
    x_chr   <- as.character(x)
    is_na   <- is.na(x_chr)
    x_chr[is_na] <- na_label
    x_fac   <- factor(x_chr, levels = unique(c(var_name, sort(unique(x_chr)))))
    counts  <- table(x_fac, useNA = "no")
    total   <- length(x_fac)

    # header row (N=)
    header <- tibble::tibble(Variable = var_name, Statistic = paste0("N = ", total))

    # level rows
    levs <- setdiff(names(counts), var_name)
    rows <- purrr::map_dfr(levs, function(lv) {
      n <- counts[[lv]]
      pct <- round(100 * n / total, 1)
      tibble::tibble(Variable = paste0("  ", lv), Statistic = paste0(n, " (", pct, "%)"))
    })

    tibble::tibble(header) |>
      dplyr::bind_rows(rows)
  }

  # ---- build summaries ----
  all_summaries <- list()

  # continuous block
  if (length(continuous_vars) > 0) {
    cont_tbl <- purrr::map_dfr(continuous_vars, function(v) {
      row <- tibble::tibble(Variable = v)
      if (show_overall) {
        row$Overall <- format_cont(data[[v]], digits = digits, style = cont_style)
      }
      for (cl in cluster_levels) {
        x <- data[data$cluster == cl, v, drop = TRUE]
        row[[cl]] <- format_cont(x, digits = digits, style = cont_style)
      }
      row
    })
    all_summaries <- append(all_summaries, list(cont_tbl))
  }

  # categorical block
  if (length(categorical_vars) > 0) {
    cat_tbl <- purrr::map_dfr(categorical_vars, function(v) {
      # base structure: header + level rows (for alignment)
      all_levels <- unique(as.character(data[[v]]))
      all_levels <- all_levels[!is.na(all_levels)]
      if (any(is.na(data[[v]]))) all_levels <- c(all_levels, na_label)

      var_header <- tibble::tibble(Variable = v)
      level_rows <- tibble::tibble(Variable = paste0("  ", all_levels))
      var_structure <- dplyr::bind_rows(var_header, level_rows)

      # Overall
      if (show_overall) {
        ovr <- format_cat(data[[v]], v) |>
          dplyr::rename(Overall = Statistic)
        var_structure <- var_structure |>
          dplyr::left_join(ovr, by = "Variable") |>
          dplyr::mutate(
            Overall = dplyr::if_else(
              is.na(.data$Overall) & .data$Variable == v,
              paste0("N = ", nrow(data)),
              .data$Overall
            )
          )
      }

      # per cluster
      for (cl in cluster_levels) {
        sub <- data[data$cluster == cl, v, drop = TRUE]
        ct  <- format_cat(sub, v) |>
          dplyr::rename(!!cl := Statistic)
        var_structure <- var_structure |>
          dplyr::left_join(ct, by = "Variable") |>
          dplyr::mutate(
            !!cl := dplyr::if_else(
              is.na(.data[[cl]]) & .data$Variable == v,
              paste0("N = ", sum(data$cluster == cl)),
              .data[[cl]]
            )
          )
      }

      var_structure
    })
    all_summaries <- append(all_summaries, list(cat_tbl))
  }

  # combine
  final_summary <- if (length(all_summaries)) dplyr::bind_rows(all_summaries) else {
    out <- tibble::tibble(Variable = character(0))
    if (show_overall) out$Overall <- character(0)
    for (cl in cluster_levels) out[[cl]] <- character(0)
    out
  }

  # sample size header row
  cluster_sizes <- table(data$cluster)
  size_row <- tibble::tibble(Variable = "N")
  if (show_overall) size_row$Overall <- as.character(nrow(data))
  for (cl in cluster_levels) {
    size_row[[cl]] <- as.character(cluster_sizes[[cl]])
  }

  final_summary <- dplyr::bind_rows(size_row, final_summary) |>
    dplyr::mutate(dplyr::across(dplyr::everything(), \(x) ifelse(is.na(x), "", as.character(x))))

  final_summary
}




#--------------------------------

# Helper function to create a formatted gt table from cluster_summary output
# Simplified version with basic functionality
format_cluster_summary_gt <- function(summary_df, title = "Cluster Characteristics") {
summary_df %>%
gt::gt() %>%
gt::tab_header(title = title) %>%
gt::tab_style(
style = gt::cell_text(weight = "bold"),
locations = gt::cells_body(columns = "Variable", rows = !grepl("^  ", Variable))
) %>%
gt::tab_style(
style = gt::cell_text(indent = gt::px(20)),
locations = gt::cells_body(columns = "Variable", rows = grepl("^  ", Variable))
) %>%
gt::cols_align(align = "left", columns = c("Variable")) %>%
gt::cols_align(align = "center", columns = everything()) %>%
gt::cols_align(align = "left", columns = c("Variable"))  # Override Variable back to left
}

format_performance_gt <- function(performance_df, title = "Model Performance by Cluster") {
performance_df %>%
gt::gt() %>%
gt::tab_header(title = title) %>%
{if("mse" %in% names(performance_df)) gt::fmt_number(., columns = c("mse"), decimals = 4) else .} %>%
{if("mae" %in% names(performance_df)) gt::fmt_number(., columns = c("mae"), decimals = 4) else .} %>%
{if("rmse" %in% names(performance_df)) gt::fmt_number(., columns = c("rmse"), decimals = 4) else .} %>%
{if("correlation" %in% names(performance_df)) gt::fmt_number(., columns = c("correlation"), decimals = 3) else .} %>%
{if("feature" %in% names(performance_df)) 
gt::tab_style(., style = gt::cell_text(weight = "bold"), locations = gt::cells_body(rows = feature == "Overall")) 
else .} %>%
gt::cols_align(align = "left", columns = c("feature", "cluster")[c("feature", "cluster") %in% names(performance_df)]) %>%
gt::cols_align(align = "center", columns = everything()) %>%
gt::cols_align(align = "left", columns = c("feature", "cluster")[c("feature", "cluster") %in% names(performance_df)])
}