#' Create or reuse a CISSVAE Python virtual environment
#'
#' @param envname Name of the virtual environment to create/use.
#' @param install_python Logical; if TRUE, install Python if none found.
#' @param python_version Python version string (major.minor), if you need to install.
#' @export
create_cissvae_env <- function(
  envname = "cissvae_environment",
  install_python = FALSE,
  python_version = "3.10"
) {
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
  if (!envname %in% reticulate::virtualenv_list()) {
    message("Creating virtualenv '", envname, "' with Python: ", starter)
    reticulate::virtualenv_create(
      envname = envname,
      python  = starter,
      packages = c("numpy", "pandas", "torch")
    )
  } else {
    message("Virtualenv '", envname, "' already exists; skipping creation.")
  }

  # 3. Activate and install CISSVAE
  reticulate::use_virtualenv(envname, required = TRUE)
  message("Installing 'cissvae' into '", envname, "' from test.pypi.org")
  reticulate::py_install(
    packages        = "cissvae",
    envname         = envname,
    extra_index_url = "https://test.pypi.org/simple/"
  )

  invisible(NULL)
}

create_cissvae_env()
