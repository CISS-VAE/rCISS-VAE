#' Check PyTorch device availability
#'
#' This function prints the available devices (cpu, cuda, mps) detected by PyTorch. If your mps/cuda device is not shown, check your PyTorch installation. 
#'
#' @param env_path Path to virtual environment containing PyTorch and ciss-vae. Defaults to NULL.
#' @return Vector of strings for available devices. 
#' @examples
#' \donttest{
#' try(
#' check_devices()
#' )}
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
  cat(paste0("  * ", devs$pretty, collapse = "\n"), "\n\n")
  
  return(devs$usable)
}

