cluster_summary <- function(data, 
  clusters, 
  index_col = NULL,
  categorical_vars = NULL,
  continuous_vars = NULL,  # Add explicit continuous vars
  digits = 2,
  show_overall = TRUE,
  na_label = "Missing") {

# Input validation
if (nrow(data) != length(clusters)) {
stop("Length of clusters must equal number of rows in data")
}

# Remove index column if specified
if (!is.null(index_col)) {
if (!index_col %in% colnames(data)) {
stop("index_col '", index_col, "' not found in data")
}
data <- data %>% select(-all_of(index_col))
}

# Add cluster column with consistent naming
data$cluster <- paste0("Cluster ", as.character(clusters))
cluster_levels <- sort(unique(data$cluster))

# Validate variable specifications
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

# Auto-detect variable types if not specified
available_vars <- setdiff(colnames(data), "cluster")

if (is.null(categorical_vars) && is.null(continuous_vars)) {
# Auto-detect both
categorical_vars <- data %>%
select(-cluster) %>%
select_if(function(x) is.factor(x) || is.character(x) || is.logical(x)) %>%
colnames()

continuous_vars <- data %>%
select(-cluster, -all_of(categorical_vars)) %>%
select_if(is.numeric) %>%
colnames()
} else if (is.null(categorical_vars)) {
# Continuous specified, auto-detect categorical from remaining
categorical_vars <- setdiff(available_vars, continuous_vars)
categorical_vars <- intersect(categorical_vars, 
       data %>% 
         select_if(function(x) is.factor(x) || is.character(x) || is.logical(x)) %>%
         colnames())
} else if (is.null(continuous_vars)) {
# Categorical specified, auto-detect continuous from remaining
continuous_vars <- setdiff(available_vars, categorical_vars)
continuous_vars <- intersect(continuous_vars,
      data %>%
        select_if(is.numeric) %>%
        colnames())
}

# Final validation - check for numeric data in continuous vars
for (var in continuous_vars) {
if (!is.numeric(data[[var]])) {
warning("Variable '", var, "' is not numeric. Moving to categorical variables.")
categorical_vars <- c(categorical_vars, var)
continuous_vars <- setdiff(continuous_vars, var)
}
}

# Print what we're processing (for debugging)
cat("Processing continuous variables:", paste(continuous_vars, collapse = ", "), "\n")
cat("Processing categorical variables:", paste(categorical_vars, collapse = ", "), "\n")

# Helper function to format continuous variables
format_continuous <- function(x, digits = 2) {
# Additional safety check
if (!is.numeric(x)) {
warning("Non-numeric data passed to format_continuous")
return("Invalid data type")
}

n_total <- length(x)
n_missing <- sum(is.na(x))
n_complete <- n_total - n_missing

if (n_complete == 0) {
return(paste0("—, ", n_missing, " (", round(100 * n_missing / n_total, 1), "%) missing"))
}

mean_val <- mean(x, na.rm = TRUE)
sd_val <- sd(x, na.rm = TRUE)
median_val <- median(x, na.rm = TRUE)
q1_val <- quantile(x, 0.25, na.rm = TRUE)
q3_val <- quantile(x, 0.75, na.rm = TRUE)

mean_sd <- paste0(round(mean_val, digits), " (", round(sd_val, digits), ")")
median_iqr <- paste0(round(median_val, digits), " [", round(q1_val, digits), ", ", round(q3_val, digits), "]")

if (n_missing > 0) {
missing_info <- paste0(", ", n_missing, " (", round(100 * n_missing / n_total, 1), "%) missing")
} else {
missing_info <- ""
}

return(paste0(mean_sd, "; ", median_iqr, missing_info))
}

# Helper function to format categorical variables
format_categorical <- function(x, var_name) {
# Convert to character first, then factor to handle various input types
x_char <- as.character(x)
x_factor <- factor(x_char, exclude = NULL)
if (any(is.na(x_char))) {
levels(x_factor)[is.na(levels(x_factor))] <- na_label
}

counts <- table(x_factor, useNA = "ifany")
total <- length(x)

# Create summary for each level
result <- purrr::map_dfr(names(counts), function(level) {
count <- counts[[level]]
percent <- round(100 * count / total, 1)
tibble(
Variable = paste0("  ", level),
Statistic = paste0(count, " (", percent, "%)")
)
})

# Add variable header
header <- tibble(
Variable = var_name,
Statistic = paste0("N = ", total)
)

bind_rows(header, result)
}

# Initialize results list
all_summaries <- list()

# Process continuous variables
if (length(continuous_vars) > 0) {
continuous_summary <- purrr::map_dfr(continuous_vars, function(var) {
# Create base tibble with variable name
result <- tibble(Variable = var)

# Add overall column if requested
if (show_overall) {
result$Overall <- format_continuous(data[[var]], digits)
}

# Add cluster columns
for (cluster_name in cluster_levels) {
cluster_data <- data[data$cluster == cluster_name, var, drop = TRUE]  # drop = TRUE ensures vector output
result[[cluster_name]] <- format_continuous(cluster_data, digits)
}

result
})
all_summaries <- append(all_summaries, list(continuous_summary))
}

# Process categorical variables
if (length(categorical_vars) > 0) {
categorical_summary <- purrr::map_dfr(categorical_vars, function(var) {
# Get all unique levels across all clusters to ensure consistent structure
all_levels <- unique(as.character(data[[var]]))
all_levels <- all_levels[!is.na(all_levels)]
if (any(is.na(data[[var]]))) {
all_levels <- c(all_levels, na_label)
}

# Create base structure
var_header <- tibble(Variable = var)
level_rows <- tibble(Variable = paste0("  ", all_levels))

# Combine header and levels
var_structure <- bind_rows(var_header, level_rows)

# Add overall column if requested
if (show_overall) {
overall_summary <- format_categorical(data[[var]], var)
# Match structure
var_structure <- var_structure %>%
left_join(overall_summary %>% rename(Overall = Statistic), by = "Variable") %>%
mutate(Overall = ifelse(is.na(Overall) & Variable == var, 
       paste0("N = ", nrow(data)), Overall))
}

# Add cluster columns
for (cluster_name in cluster_levels) {
cluster_data <- data[data$cluster == cluster_name, var, drop = TRUE]
cluster_summary <- format_categorical(cluster_data, var)

# Match structure
var_structure <- var_structure %>%
left_join(cluster_summary %>% rename(!!cluster_name := Statistic), by = "Variable") %>%
mutate(!!cluster_name := ifelse(is.na(.data[[cluster_name]]) & Variable == var,
               paste0("N = ", sum(data$cluster == cluster_name)), 
               .data[[cluster_name]]))
}

var_structure
})
all_summaries <- append(all_summaries, list(categorical_summary))
}

# Combine all summaries
if (length(all_summaries) > 0) {
final_summary <- bind_rows(all_summaries)
} else {
# Create empty structure if no variables
final_summary <- tibble(Variable = character(0))
if (show_overall) final_summary$Overall <- character(0)
for (cluster_name in cluster_levels) {
final_summary[[cluster_name]] <- character(0)
}
}

# Add sample size row at the top
cluster_sizes <- table(data$cluster)
size_row <- tibble(Variable = "N")

if (show_overall) {
size_row$Overall <- as.character(nrow(data))
}

for (cluster_name in cluster_levels) {
size_row[[cluster_name]] <- as.character(cluster_sizes[[cluster_name]])
}

# Combine size row with summary
final_summary <- bind_rows(size_row, final_summary)

# Replace NA values with empty strings for cleaner output
final_summary <- final_summary %>%
mutate(across(everything(), ~ifelse(is.na(.), "", as.character(.))))

return(final_summary)
}

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#' Calculate performance metrics by cluster and subgroups
#'
#' @param original_data Original data frame with true values (for grouping variables)
#' @param model Trained Python CISS-VAE model object
#' @param dataset Python ClusterDataset object used for training (contains validation data)
#' @param clusters Vector of cluster assignments
#' @param index_col Optional name of index column to align data
#' @param grouping_vars Optional vector of column names to group performance by (e.g., c("race", "sex"))
#' @param device Device for model inference ("cpu" or "cuda")
#' @param metrics Vector of metrics to compute (default: c("mse", "mae", "correlation"))
#' @param only_validation Logical; if TRUE, only compute metrics on validation (masked) entries
#' @param verbose Logical; if TRUE, prints debug information
#'
#' @return Data frame with performance metrics by cluster and optional subgroups
#' @export
performance_by_cluster <- function(original_data,
  model,
  dataset,
  clusters,
  index_col = NULL,
  grouping_vars = NULL,
  device = "cpu",
  metrics = c("mse", "mae", "correlation"),
  only_validation = TRUE,
  verbose = FALSE) {  # Added verbose parameter

# Import Python function for getting model predictions
helpers_mod <- reticulate::import("ciss_vae.utils.helpers", convert = FALSE)

# Check if we have the get_val_comp_df function and use it
if (reticulate::py_has_attr(helpers_mod, "get_val_comp_df")) {
tryCatch({
# Get predictions using the helper function with ClusterDataset
predictions_py <- helpers_mod$get_val_comp_df(model, dataset, device = device)
predictions_df <- reticulate::py_to_r(predictions_py)
}, error = function(e) {
stop("Failed to get model predictions from ClusterDataset. Error: ", e$message)
})
} else {
stop("get_val_comp_df function not found in ciss_vae.utils.helpers module")
}

# Extract validation data from ClusterDataset object
tryCatch({
# Extract val_data tensor from ClusterDataset
val_data_py <- dataset$val_data

# Convert tensor to R data.frame
if (reticulate::py_has_attr(val_data_py, "cpu")) {
# PyTorch tensor on GPU
val_data_array <- reticulate::py_to_r(val_data_py$cpu()$numpy())
} else if (reticulate::py_has_attr(val_data_py, "numpy")) {
# PyTorch tensor on CPU
val_data_array <- reticulate::py_to_r(val_data_py$numpy())
} else {
# Already numpy array or other format
val_data_array <- reticulate::py_to_r(val_data_py)
}

val_data_df <- as.data.frame(val_data_array, stringsAsFactors = FALSE)

}, error = function(e) {
stop("Failed to extract validation data from ClusterDataset. Error: ", e$message)
})

# Get feature names from ClusterDataset
tryCatch({
if (reticulate::py_has_attr(dataset, "feature_names")) {
feature_names <- reticulate::py_to_r(dataset$feature_names)

# Set column names for both dataframes
if (length(feature_names) == ncol(val_data_df)) {
colnames(val_data_df) <- feature_names
}
if (length(feature_names) == ncol(predictions_df)) {
colnames(predictions_df) <- feature_names
}
} else {
warning("ClusterDataset does not have feature_names attribute")
# Generate generic column names
feature_names <- paste0("feature_", seq_len(ncol(val_data_df)))
colnames(val_data_df) <- feature_names
colnames(predictions_df) <- feature_names
}
}, error = function(e) {
warning("Failed to get feature names from ClusterDataset: ", e$message)
# Generate generic column names
feature_names <- paste0("feature_", seq_len(ncol(val_data_df)))
colnames(val_data_df) <- feature_names
colnames(predictions_df) <- feature_names
})

# Since there's no val_mask, we'll compute metrics on all data when only_validation = TRUE
if (only_validation) {
warning("ClusterDataset does not contain validation mask. Computing metrics on all data.")
warning("To get true validation-only metrics, you would need access to the original validation mask.")
only_validation <- FALSE
}

# Prepare original data for grouping variables
if (!is.null(index_col)) {
if (!index_col %in% colnames(original_data)) {
stop("index_col '", index_col, "' not found in original_data")
}
original_clean <- original_data %>% dplyr::select(-dplyr::all_of(index_col))
} else {
original_clean <- original_data
}

# Validate dimensions
if (length(clusters) != nrow(val_data_df)) {
stop("Length of clusters (", length(clusters), ") must match number of rows in validation data (", nrow(val_data_df), ")")
}

if (nrow(original_data) != nrow(val_data_df)) {
stop("Number of rows in original_data (", nrow(original_data), ") must match validation data (", nrow(val_data_df), ")")
}

# Add cluster and grouping variables to datasets
predictions_df$cluster <- clusters
val_data_df$cluster <- clusters

if (!is.null(grouping_vars)) {
for (var in grouping_vars) {
if (!var %in% colnames(original_data)) {
stop("Grouping variable '", var, "' not found in original_data")
}
predictions_df[[var]] <- original_data[[var]]
val_data_df[[var]] <- original_data[[var]]
}
}

# Get feature columns (exclude cluster and grouping vars)
exclude_cols <- c("cluster", grouping_vars)
feature_cols <- setdiff(colnames(predictions_df), exclude_cols)

# Ensure feature columns match between prediction and validation data
feature_cols <- intersect(feature_cols, colnames(val_data_df))

if (length(feature_cols) == 0) {
if (verbose) {
cat("Prediction columns:", colnames(predictions_df), "\n")
cat("Validation columns:", colnames(val_data_df), "\n")
}
stop("No matching feature columns found between predictions and validation data")
}

if (verbose) {
cat("Found", length(feature_cols), "matching feature columns\n")
cat("Feature columns:", head(feature_cols, 10), "\n")
}

# Helper function to compute metrics
compute_metrics <- function(true_vals, pred_vals, requested_metrics) {
# Remove rows where either value is missing
complete_cases <- complete.cases(true_vals, pred_vals)
true_clean <- true_vals[complete_cases]
pred_clean <- pred_vals[complete_cases]

if (length(true_clean) == 0) {
result <- list(n_obs = 0)
for (metric in requested_metrics) {
result[[metric]] <- NA_real_
}
return(result)
}

result <- list(n_obs = length(true_clean))

if ("mse" %in% requested_metrics) {
result$mse <- mean((true_clean - pred_clean)^2)
}

if ("mae" %in% requested_metrics) {
result$mae <- mean(abs(true_clean - pred_clean))
}

if ("rmse" %in% requested_metrics) {
result$rmse <- sqrt(mean((true_clean - pred_clean)^2))
}

if ("correlation" %in% requested_metrics) {
if (length(unique(true_clean)) > 1 && length(unique(pred_clean)) > 1) {
result$correlation <- cor(true_clean, pred_clean, use = "complete.obs")
} else {
result$correlation <- NA_real_
}
}

return(result)
}

# Prepare data for metric calculation - compute on all data since no validation mask
metric_data <- purrr::map_dfr(feature_cols, function(feature) {
result_df <- tibble::tibble(
feature = feature,
cluster = clusters,
true_val = val_data_df[[feature]],  # True values from val_data
pred_val = predictions_df[[feature]] # Predictions from model
)

# Add grouping variables if specified
if (!is.null(grouping_vars)) {
for (var in grouping_vars) {
result_df[[var]] <- original_data[[var]]
}
}

return(result_df)
})

# Remove rows with missing values
metric_data <- metric_data %>%
dplyr::filter(!is.na(true_val), !is.na(pred_val))

if (nrow(metric_data) == 0) {
warning("No valid data points found for performance calculation")
return(tibble::tibble())
}

if (verbose) {
cat("Final metric data has", nrow(metric_data), "observations\n")
}

# Set up grouping variables
group_vars <- c("cluster")
if (!is.null(grouping_vars)) {
group_vars <- c(group_vars, grouping_vars)
}

# Compute overall metrics (across all features)
overall_metrics <- metric_data %>%
dplyr::group_by(dplyr::across(dplyr::all_of(group_vars))) %>%
dplyr::summarise(
feature = "Overall",
!!!compute_metrics(true_val, pred_val, metrics),
.groups = "drop"
)

# Compute feature-specific metrics
feature_metrics <- metric_data %>%
dplyr::group_by(feature, dplyr::across(dplyr::all_of(group_vars))) %>%
dplyr::summarise(
!!!compute_metrics(true_val, pred_val, metrics),
.groups = "drop"
)

# Combine results
all_metrics <- dplyr::bind_rows(overall_metrics, feature_metrics)

# Format for gt table
all_metrics <- all_metrics %>%
dplyr::mutate(
cluster = paste("Cluster", cluster),
dplyr::across(dplyr::all_of(metrics), ~ round(.x, 4))
) %>%
dplyr::arrange(cluster, feature)

return(all_metrics)
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