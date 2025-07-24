.onLoad <- function(libname, pkgname) {
  envname <- "~/.virtualenvs/cissvae_env"
  
  # If the virtualenv doesn't exist, create it and install torch
  if (!reticulate::virtualenv_exists(envname)) {
    message("Creating Python virtualenv for ", pkgname, " at ", envname)
    reticulate::virtualenv_create(envname, python = "python3")
    # Install your Python dependencies
    reticulate::virtualenv_install(
      envname,
      packages = c("torch", "numpy")  # add other deps here
    )
  }
  
  # Always point reticulate at it (but don't force-throw if something else is OK)
  reticulate::use_virtualenv(envname, required = FALSE)
  
  invisible()
}
