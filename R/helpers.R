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

#' Check PyTorch device availability
#'
#' This function prints the available devices (cpu, cuda, mps) detected by PyTorch. If your mps/cuda device is not shown, check your PyTorch installation. 
#'
#' @param env_path Path to virtual environment containing PyTorch and ciss-vae. Defaults to NULL.
#' @return Vector of strings for available devices. 
#' @export
check_devices <- function(env_path = NULL){
  
  if(!is.null(env_path)){
    reticulate::use_virtualenv(env_path, required = TRUE)
  }

  torch <- reticulate::import("torch")

  get_available_torch_devices <- function() {
    devices <- c()
    pretty <- c()
    
    # --- MPS (Apple Silicon) ---
    if (torch$backends$mps$is_available()) {
      devices <- c(devices, "mps")
      pretty  <- c(pretty, "mps  (Apple Metal Performance Shaders (GPU))")
    }
    
    # --- CUDA devices ---
    if (torch$cuda$is_available()) {
      cuda_count <- torch$cuda$device_count()
      if (cuda_count > 0) {
        for (i in 0:(cuda_count - 1)) {
          dev_string <- sprintf("cuda:%d", i)
          gpu_name <- torch$cuda$get_device_name(i)
          
          devices <- c(devices, dev_string)
          pretty  <- c(pretty, sprintf("%s  (%s)", dev_string, gpu_name))
        }
      }
    }
    
    # --- CPU ---
    devices <- c(devices, "cpu")
    pretty  <- c(pretty, "cpu  (Main system processor)")
    
    list(
      usable = devices,    # what you pass to torch$device()
      pretty = pretty      # human-readable names
    )
  }

  # Get devices
  devs <- get_available_torch_devices()

  cat("Available Devices:\n")
  cat(paste0("  • ", devs$pretty, collapse = "\n"), "\n\n")
  
  return(devs$usable)
}

