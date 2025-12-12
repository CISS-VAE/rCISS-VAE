#' Create or reuse a CISSVAE Python virtual environment
#'
#' This function will either find an existing virtualenv by name (in
#' the default location) or at a custom filesystem path, or create it
#' (and install CISSVAE into it).
#'
#' @param envname Name of the virtual environment (when using the default env location).
#' @param path Character; optional path to the directory in which to create/use the virtualenv.
#' @param install_python Logical; if TRUE, install Python if none of at least the requested
#'   version is found on the system.
#' @param python_version Python version string (major.minor), used when installing Python.
#' 
#' @examples
#' \dontrun{
#' create_cissvae_env(
#' envname = "cissvae_environment",
#' install_python = FALSE,
#' python_version = "3.10")
#' }
#' @export
create_cissvae_env <- function(
  envname        = "cissvae_environment",
  path           = NULL,
  install_python = FALSE,
  python_version = "3.10"
) {
  # decide what “env_spec” we pass to reticulate:
  #  - if the user gave a path, use that (full directory + envname)
  #  - otherwise use the envname (i.e. default location + name)
  env_spec <- if (!is.null(path)) file.path(path, envname) else envname

  # 1. Check for a suitable Python starter (>= requested version)
  starter <- reticulate::virtualenv_starter(python_version)
  if (is.null(starter)) {
    if (install_python) {
      message("No suitable Python found; installing Python ", python_version)
      reticulate::install_python(version = python_version)
      starter <- reticulate::virtualenv_starter(python_version)
      if (is.null(starter)) {
        stop("Failed to install Python ", python_version)
      }
    } else {
      stop(
        "No Python >= ", python_version,
        " found. Please install Python or set install_python = TRUE."
      )
    }
  }

  # 2. Create the virtual environment (or skip if it already exists)
  env_exists <- if (is.null(path)) {
    # in default location, list known envs by name
    envname %in% reticulate::virtualenv_list()
  } else {
    # if path provided, just check the directory
    dir.exists(file.path(path, envname))
  }

  if (!env_exists) {
    message(
      "Creating virtualenv '", env_spec, 
      "' with Python: ", starter
    )
    reticulate::virtualenv_create(
      envname = env_spec,
      python  = starter,
      packages = c("numpy", "pandas", "torch", "rich", 
      "matplotlib", "scikit-learn", "optuna", "typing", "ciss-vae")
    )
  } else {
    message(
      "Virtualenv '", env_spec, 
      "' already exists; skipping creation."
    )
  }

  invisible(NULL)
}
