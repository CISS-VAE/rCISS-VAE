#' Evaluate imputation accuracy (R wrapper)
#'
#' Compare imputed values to ground truth at originally missing positions,
#' by calling the Python function `evaluate_imputation()` defined in the
#' ciss_vae module.
#'
#' @param imputed_df A data.frame or tibble of imputed values (same dim as df_complete)
#' @param df_complete A data.frame or tibble of the complete (ground-truth) values
#' @param df_missing A data.frame or tibble with NAs indicating the original missing entries
#' @return A list with components:
#'   \itemize{
#'     \item{\code{mse}}{Mean squared error at the originally missing positions}
#'     \item{\code{comparison}}{A data.frame with columns \code{row}, \code{col},
#'       \code{true}, \code{imputed}, and \code{squared_error}}
#'   }
#' @export
evaluate_imputation <- function(imputed_df, df_complete, df_missing) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for evaluate_imputation()")
  }

  # Import pandas and your Python module
  pd     <- reticulate::import("pandas", convert = FALSE)
  cvae   <- reticulate::import("ciss_vae", convert = FALSE)

  # Convert R data.frames/tibbles to pandas DataFrames
  imp_py      <- pd$DataFrame(imputed_df)
  complete_py <- pd$DataFrame(df_complete)
  missing_py  <- pd$DataFrame(df_missing)

  # Call the Python function directly
  result <- cvae$evaluate_imputation(imp_py, complete_py, missing_py)

  # Unpack and convert back to R
  res      <- reticulate::py_to_r(result)

  mse = res[1]
  
}

## Make helper that 