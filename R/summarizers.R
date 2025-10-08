#'@export
cluster_summary <- function(data, 
  clusters, 
  index_col = NULL,
  categorical_vars = NULL,
  continuous_vars = NULL,  # Add explicit continuous vars
  digits = 2,
  show_overall = TRUE,
  na_label = "Missing") {
  
  ## for cluster_summary maybe just import gt_summary's tbl_summary and pretty it up for the task at hand? 

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