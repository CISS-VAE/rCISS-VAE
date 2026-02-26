library(testthat)
library(reticulate)
library(rCISSVAE)

# Utility to skip tests if Python or modules are unavailable
skip_if_no_python <- function(env = "cissvae_environment") {
  if (!reticulate::py_available(initialize = FALSE) ||
      !reticulate::py_module_available("torch")) {
    skip("Python or required Python modules not available")
  }
  # Try activating the environment
  tryCatch(
    reticulate::use_virtualenv(env, required = TRUE),
    error = function(e) skip(paste0("Cannot activate Python env: ", env))
  )
}

test_that("save_cissvae_model saves and creates a file", {
  skip_if_no_python()

  # Train a very small CISSVAE model (or use a saved dummy)
  # Here we create a minimal run_cissvae call for testing
  # You need test data obj df_missing & clusters defined in tests/testthat/
  df_missing <- data.frame(
    x = c(NA_real_, 1.0, 2.0),
    y = c(1.0, NA_real_, 3.0)
  )
  clusters <- c(1L, 1L, 2L)

  # Train with minimal settings
  res <- run_cissvae(
    data = df_missing,
    index_col = NULL,
    clusters = clusters,
    epochs = 1,
    return_model = TRUE,
    verbose = FALSE,
    device = "cpu"
  )

  # Save
  tmp_file <- tempfile(fileext = ".pt")
  expect_silent(save_cissvae_model(res$model, tmp_file))
  
  # File should exist
  expect_true(file.exists(tmp_file))
})

test_that("load_cissvae_model loads a saved model", {
  skip_if_no_python()

  tmp_file <- tempfile(fileext = ".pt")

  # Minimal model train + save to get a file
  df_missing <- data.frame(
    x = c(NA_real_, 1.0, 2.0),
    y = c(1.0, NA_real_, 3.0)
  )
  clusters <- c(1L, 1L, 2L)
  res <- run_cissvae(
    data = df_missing,
    index_col = NULL,
    clusters = clusters,
    epochs = 1,
    return_model = TRUE,
    verbose = FALSE,
    device = "cpu"
  )
  save_cissvae_model(res$model, tmp_file)

  # Loading should return a Python object
  loaded <- load_cissvae_model(tmp_file, python_env = "cissvae_environment")
  expect_true(inherits(loaded, "python.builtin.object"))
})

test_that("impute_with_cissvae produces expected output structure", {
  skip_if_no_python()

  tmp_file <- tempfile(fileext = ".pt")

  # Train a minimal model
  df_missing <- data.frame(
    x = c(NA_real_, 1.0, 2.0),
    y = c(1.0, NA_real_, 3.0)
  )
  clusters <- c(1L, 1L, 2L)
  
  res <- run_cissvae(
    data = df_missing,
    index_col = NULL,
    clusters = clusters,
    epochs = 1,
    return_model = TRUE,
    verbose = FALSE,
    device = "cpu"
  )
  save_cissvae_model(res$model, tmp_file)

  # Load it
  model <- load_cissvae_model(tmp_file, python_env = "cissvae_environment")

  # Run imputation
  out <- impute_with_cissvae(
    model_py = model,
    data = df_missing,
    index_col = NULL,
    columns_ignore = NULL,
    clusters = clusters,
    imputable_matrix = NULL,
    binary_feature_mask = NULL,
    val_proportion = 0.1,
    replacement_value = 0,
    batch_size = 2L,
    seed = 123
  )

  expect_true(is.data.frame(out))
  expect_equal(nrow(out), nrow(df_missing))
  expect_equal(ncol(out), ncol(df_missing))
})