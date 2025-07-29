#' Create or re-create the Python virtual environment
#'
#' This helper lives inside your package and can be called by users after
#' `library(yourpkg)`. It will:
#'  - Look for a Python interpreter in several ways
#'  - Optionally install Miniconda if none is found
#'  - Create a venv at `envpath` (if missing)
#'  - Install the requested PyPI packages
#'
#' @param envpath Path to create/use the virtualenv.
#' @param python Optional path to the Python binary. If `NULL`, the function tries:
#'   1. `Sys.which("python3")` and `Sys.which("python")`  
#'   2. `system2("which", "python", stdout = TRUE, stderr = FALSE)`  
#'   3. `reticulate::py_discover_config()`  
#' @param packages Character vector of PyPI packages to install. Defaults to `c("torch","numpy")`.  
#' @param install_miniconda Logical; if `TRUE` and no other Python is found, installs Miniconda via reticulate. Defaults to `FALSE`.  
#' @return Invisibly returns `envpath`.  
#' @export
setup_python_env <- function(
  envpath,
  python = NULL,
  packages = c("torch", "numpy"),
  install_miniconda = FALSE
) {
  # 1. If the user passed a python path, check it right away
  if (!is.null(python) && nzchar(Sys.which(python))) {
    python_bin <- python
  } else {
    python_bin <- NULL
    
    # 2. Try Sys.which for python3 and python
    for (exe in c("python3", "python")) {
      path_exe <- Sys.which(exe)
      if (nzchar(path_exe)) {
        python_bin <- exe
        break
      }
    }
    
    # 3. Try system2("which", "python") for shell PATH
    if (is.null(python_bin)) {
      which_out <- tryCatch(
        system2("which", "python", stdout = TRUE, stderr = FALSE),
        error = function(e) character(0)
      )
      if (length(which_out) && nzchar(which_out)) {
        python_bin <- which_out[1]
      }
    }
    
    # 4. Try reticulate discovery
    if (is.null(python_bin)) {
      cfg <- try(reticulate::py_discover_config(TRUE), silent = TRUE)
      if (!inherits(cfg, "try-error") && nzchar(cfg$python)) {
        python_bin <- cfg$python
      }
    }
  }
  
  # 5. Optionally install Miniconda if still no python
  if (is.null(python_bin) && install_miniconda) {
    message("No Python found—installing Miniconda via reticulate…")
    reticulate::install_miniconda()
    python_bin <- reticulate::miniconda_python()
  }
  
  # 6. Final check
  if (is.null(python_bin) || !nzchar(Sys.which(basename(python_bin)))) {
    stop(
      "Could not locate a Python interpreter.\n",
      "Please install Python, or call:\n",
      "  setup_python_env(envpath, install_miniconda = TRUE)\n",
      "Or explicitly pass your python path:\n",
      "  setup_python_env(envpath, python = '/usr/local/bin/python3')"
    )
  }
  
  # 7. Create virtualenv if missing
  if (!reticulate::virtualenv_exists(envpath)) {
    message("Creating Python virtual environment at: ", envpath)
    reticulate::virtualenv_create(envpath, python = python_bin)
  } else {
    message("Virtual environment already exists at: ", envpath)
  }
  
  # 8. Install requested packages
  message("Installing Python packages: ", paste(packages, collapse = ", "))
  reticulate::virtualenv_install(envpath, packages = packages)
  
  invisible(envpath)
}
