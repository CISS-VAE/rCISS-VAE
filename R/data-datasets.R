
#' Sample dataset with missing values
#'
#' A tibble of simulated biomarker measurements with missing entries.  
#' Each row corresponds to one observation (indexed by `index`), and the remaining
#' columns are the measured biomarker values, some of which are set to NA to
#' demonstrate imputation workflows.
#'
#' @docType data
#' @format A tibble with *8,000* rows and *30* variables:
#' \describe{
#'   \item{index}{Integer. Row identifier imported from `data_raw/df_missing.csv`.}
#'   \item{Age, Salary, ZipCode10001-ZipCode30003}{Demographic columns. Omit from selection of validation set. No missingness}
#'   \item{Y11, ..., Y55}{Simulated Biomarker columns, have missingness}
#' }
#' @source Imported from `data_raw/df_missing.csv`, then renamed `...1` → `index`.  
#' @examples
#' data(df_missing)
#' str(df_missing)
#' summary(df_missing)
"df_missing"

#' Complete version of the sample dataset
#'
#' The same simulated biomarker measurements as in `df_missing`, but with **no** missing values—
#' useful as a ground truth for evaluating imputation methods.
#'
#' @docType data
#' @format A tibble with *8,000* rows and *30* variables containing full simulated data:
#' \describe{
#'   \item{index}{Integer. Row identifier imported from `data_raw/df_complete.csv`.}
#'   \item{Age, Salary, ZipCode10001-ZipCode30003}{Demographic columns.}
#'   \item{Y11, ..., Y55}{Simulated Biomarker columns}
#' }
#' @source Imported from `data_raw/df_complete.csv`, then renamed `...1` → `index`.  
#' @examples
#' data(df_complete)
#' head(df_complete)
"df_complete"

#' Cluster assignments based on missingness patterns
#'
#' A tibble assigning each observation in `df_missing`/`df_complete` to a cluster
#' determined by its missingness pattern.  Useful for stratifying downstream analyses.
#'
#' @docType data
#' @format A tibble with *8000* rows and 2 variables:
#' \describe{
#'   \item{index}{Integer. Row identifier imported from `data_raw/clusters.csv`.}
#'   \item{cluster}{Factor (or integer) giving the missingness‐based cluster for each row.}
#' }
#' @source Imported from `data_raw/clusters.csv`, then renamed `...1` → `index`.  
#' @examples
#' data(clusters)
#' table(clusters$cluster)
"clusters"


#' Example dni matrix for demo of imputable_matrix
#'
#' A sample imputable_matrix (dataframe)
#'
#' @docType data
#' @format A dataframe:
#' \describe{
#'   \item{imputable_matrix}{A mock imputable_matrix dataframe}
#' }
#' @source Imported from `data_raw/dni.csv`
#' @examples
#' data(dni)
"dni"

#' Example survival data for demo of imputable_matrix
#'
#' A sample survival dataset
#'
#' @docType data
#' @format A dataframe:
#' \describe{
#'   \item{mock_surv}{A mock survival dataset}
#' }
#' @source Imported from `data_raw/mock_survival.csv`
#' @examples
#' data(mock_surv)
"mock_surv"