#' Create or re-create the Python virtual environment for YourVAEWrapper
#'
#' This function checks for the existence of a specified Python virtual environment and creates it if
#' missing, then installs (or updates) the required Python packages.
#'
#' @param envpath Character string. Path to the virtual environment directory. Defaults to the value of
#'   the R option `rCISSVAE.python_env`, then the environment variable `VAE_R_ENV`,
#'   and finally `"~/.virtualenvs/"` if neither is set.
#' @param python Character string. The Python interpreter to use when creating the environment.
#'   Defaults to `"python3"` (system Python 3).
#' @param packages Character vector. Python packages to install in the environment.
#'   Defaults to `c("torch", "numpy")`. You can add other dependencies as needed.
#'
#' @return Invisibly returns the path to the Python virtual environment (`envpath`).
#' @export
#'
setup_python_env <- function(
  envpath = getOption(
    "rCISSVAE.python_env",
    Sys.getenv("VAE_R_ENV", "~/.virtualenvs")
  ),
  python = "python3",
  packages = c("torch", "numpy")
) {
  # Ensure reticulate is installed; stop if missing
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop(
      "The 'reticulate' package is required but not installed. ",
      "Please install it via install.packages('reticulate')."
    )
  }

  # Check if the virtual environment already exists
  if (!reticulate::virtualenv_exists(envpath)) {
    message("Creating Python virtual environment at: ", envpath)
    # Create a new virtual environment using the specified Python interpreter
    reticulate::virtualenv_create(envpath, python = python)
  } else {
    message("Virtual environment already exists at: ", envpath)
  }

  # Install (or update) the required Python packages within the virtual environment
  message("Installing Python packages: ", paste(packages, collapse = ", "))
  reticulate::virtualenv_install(
    envpath,
    packages = packages,
    ignore_installed = FALSE # set TRUE to force reinstallation of packages
  )

  # Return the environment path invisibly for programmatic use
  invisible(envpath)
}
