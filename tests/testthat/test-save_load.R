# =============================================================================
# tests/testthat/test-save_load_model.R
#
# Tests for:
#   - save_cissvae_model()
#   - load_cissvae_model()
#   - impute_with_cissvae()
# =============================================================================

library(testthat)
library(reticulate)
library(rCISSVAE)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

skip_if_no_python <- function() {
  if (!reticulate::py_module_available("ciss_vae")) {
    skip("Python module ciss_vae not available")
  }
}


# Minimal data sufficient to run a single training epoch
.make_test_data <- function() {
  list(
    df = data.frame(
      x = c(NA_real_, 1.0, 2.0, 1.5, NA_real_),
      y = c(1.0, NA_real_, 3.0, NA_real_, 2.0),
      z = c(0.5, 1.5, NA_real_, 2.5, 1.0)
    ),
    clusters = c(0L, 0L, 1L, 1L, 0L)
  )
}

.make_test_data_with_categorical <- function() {
  df <- data.frame(
    x = c(NA_real_, 1.0, 2.0, 1.5, NA_real_),
    y = c(1.0, NA_real_, 3.0, NA_real_, 2.0),

    # categorical dummy group 1
    C1_b1 = c(1, 0, 1, 0, 1),
    C1_b2 = c(0, 1, 0, 1, 0),

    # categorical dummy group 2
    C2_b1 = c(1, 1, 0, 0, 1),
    C2_b2 = c(0, 0, 1, 1, 0)
  )

  list(
    df = df,
    clusters = c(0L, 0L, 1L, 1L, 0L),

    categorical_column_map = list(
      C1 = c("C1_b1", "C1_b2"),
      C2 = c("C2_b1", "C2_b2")
    ),

    # ✅ binary_feature_mask aligned with df columns
    # x, y → continuous → FALSE
    # all dummy columns → TRUE
    binary_feature_mask = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)
  )
}

# Train the smallest possible model and return the result object
.train_minimal <- function(td) {
  run_cissvae(
    data = td$df,
    clusters = td$clusters,
    categorical_column_map = td$categorical_column_map,
    binary_feature_mask = td$binary_feature_mask,
    epochs = 1L,
    return_model = TRUE,
    verbose = FALSE,
    device = "cpu"
  )
}

# =============================================================================
# save_cissvae_model
# =============================================================================

test_that("save_cissvae_model (state_dict) writes a .pt file", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file <- tempfile(fileext = ".pt")
  on.exit(unlink(c(pt_file, paste0(pt_file, ".config.rds"))))

  expect_no_error(
    save_cissvae_model(res$model, file = pt_file)
  )
  expect_true(file.exists(pt_file))
})

test_that("save_cissvae_model (state_dict) writes a paired .config.rds", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)
  expect_true(file.exists(config_file))
})

test_that("save_cissvae_model config contains all required architecture fields", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)
  cfg <- readRDS(config_file)

  expect_named(cfg, c(
    "input_dim", "hidden_dims",
    "layer_order_enc", "layer_order_dec",
    "latent_shared", "latent_dim",
    "output_shared", "num_clusters",
    "activation_groups"
  ), ignore.order = TRUE)
})

test_that("save_cissvae_model (full) writes only a single .pt file", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file, method = "full")

  expect_true(file.exists(pt_file))
  expect_false(file.exists(config_file))
})

test_that("save_cissvae_model errors when files exist and overwrite = FALSE", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)

  expect_error(
    save_cissvae_model(res$model, file = pt_file, overwrite = FALSE),
    regexp = "already exist"
  )
})

test_that("save_cissvae_model overwrites silently when overwrite = TRUE", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)

  expect_no_error(
    save_cissvae_model(res$model, file = pt_file, overwrite = TRUE)
  )
})

test_that("save_cissvae_model errors on invalid method", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  expect_error(
    save_cissvae_model(res$model, file = tempfile(), method = "bad_method"),
    regexp = "arg"   # match.arg error
  )
})

# =============================================================================
# load_cissvae_model
# =============================================================================

test_that("load_cissvae_model (state_dict) returns a Python object", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)
  loaded <- load_cissvae_model(file = pt_file)

  expect_true(inherits(loaded, "python.builtin.object"))
})

test_that("load_cissvae_model (state_dict) restores architecture correctly", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)
  loaded <- load_cissvae_model(file = pt_file)

  expect_equal(
    reticulate::py_to_r(loaded$input_dim),
    reticulate::py_to_r(res$model$input_dim)
  )
  expect_equal(
    reticulate::py_to_r(loaded$latent_dim),
    reticulate::py_to_r(res$model$latent_dim)
  )
  expect_equal(
    reticulate::py_to_r(loaded$num_clusters),
    reticulate::py_to_r(res$model$num_clusters)
  )
})

test_that("load_cissvae_model (state_dict) loads model in eval mode", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)
  loaded <- load_cissvae_model(file = pt_file)

  # training attribute should be False after eval()
  expect_false(reticulate::py_to_r(loaded$training))
})

test_that("load_cissvae_model (full) returns a Python object", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file <- tempfile(fileext = ".pt")
  on.exit(unlink(pt_file))

  save_cissvae_model(res$model, file = pt_file, method = "full")
  loaded <- load_cissvae_model(file = pt_file, method = "full")

  expect_true(inherits(loaded, "python.builtin.object"))
})

test_that("load_cissvae_model errors when .pt file not found", {
  skip_if_no_python()

  expect_error(
    load_cissvae_model(file = "/no/such/file.pt"),
    regexp = "not found"
  )
})

test_that("load_cissvae_model (state_dict) errors when config file is missing", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)

  # delete the config to simulate a missing file
  unlink(config_file)

  expect_error(
    load_cissvae_model(file = pt_file, method = "state_dict"),
    regexp = "Config file not found"
  )
})

test_that("load_cissvae_model errors on invalid method", {
  skip_if_no_python()

  expect_error(
    load_cissvae_model(file = tempfile(), method = "bad_method"),
    regexp = "arg"
  )
})

# =============================================================================
# impute_with_cissvae
# =============================================================================

test_that("impute_with_cissvae produces a data.frame with correct dimensions", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)
  model <- load_cissvae_model(file = pt_file)

  out <- impute_with_cissvae(
    model                = model,
    data                 = td$df,
    index_col            = NULL,
    cols_ignore          = NULL,
    clusters             = td$clusters,
    imputable_matrix     = NULL,
    binary_feature_mask  = NULL,
    replacement_value    = 0,
    batch_size           = 4000L,
    seed                 = 123L
  )

  expect_true(is.data.frame(out))
  expect_equal(nrow(out), nrow(td$df))
  expect_equal(ncol(out), ncol(td$df))
})

test_that("impute_with_cissvae output contains no NA in imputable cells", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)
  model <- load_cissvae_model(file = pt_file)

  out <- impute_with_cissvae(
    model                = model,
    data                 = td$df,
    index_col            = NULL,
    cols_ignore          = NULL,
    clusters             = td$clusters,
    imputable_matrix     = NULL,
    binary_feature_mask  = NULL,
    replacement_value    = 0,
    batch_size           = 4000L,
    seed                 = 123L
  )

  expect_false(anyNA(out))
})

test_that("impute_with_cissvae preserves column names", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)
  model <- load_cissvae_model(file = pt_file)

  out <- impute_with_cissvae(
    model                = model,
    data                 = td$df,
    index_col            = NULL,
    cols_ignore          = NULL,
    clusters             = td$clusters,
    imputable_matrix     = NULL,
    binary_feature_mask  = NULL,
    replacement_value    = 0,
    batch_size           = 4000L,
    seed                 = 123L
  )

  expect_equal(colnames(out), colnames(td$df))
})

## ====================
## Add test for activation groups
## ====================

test_that("load_cissvae_model errors if activation_groups missing", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)

  cfg <- readRDS(config_file)
  cfg$activation_groups <- NULL
  saveRDS(cfg, config_file)

  expect_error(
    load_cissvae_model(file = pt_file),
    regexp = "activation_groups"
  )
})

test_that("activation_groups is preserved after save/load", {
  skip_if_no_python()
  td  <- .make_test_data()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)
  loaded <- load_cissvae_model(file = pt_file)

  expect_equal(
    reticulate::py_to_r(loaded$activation_groups),
    reticulate::py_to_r(res$model$activation_groups)
  )
})

test_that("categorical_column_map correctly populates activation_groups", {
  skip_if_no_python()
  td  <- .make_test_data_with_categorical()
  res <- .train_minimal(td)

  ag <- reticulate::py_to_r(res$model$activation_groups)

  expect_true("C1" %in% names(ag))
  expect_true("C2" %in% names(ag))

  expect_length(ag$C1, 2)
  expect_length(ag$C2, 2)
})

test_that("categorical activation_groups preserved after save/load", {
  skip_if_no_python()
  td  <- .make_test_data_with_categorical()
  res <- .train_minimal(td)

  pt_file     <- tempfile(fileext = ".pt")
  config_file <- paste0(pt_file, ".config.rds")
  on.exit(unlink(c(pt_file, config_file)))

  save_cissvae_model(res$model, file = pt_file)
  loaded <- load_cissvae_model(file = pt_file)

  expect_equal(
    reticulate::py_to_r(loaded$activation_groups),
    reticulate::py_to_r(res$model$activation_groups)
  )
})

test_that("impute_with_cissvae works with categorical_column_map", {
  skip_if_no_python()
  td  <- .make_test_data_with_categorical()
  res <- .train_minimal(td)

  pt_file <- tempfile(fileext = ".pt")
  on.exit(unlink(c(pt_file, paste0(pt_file, ".config.rds"))))

  save_cissvae_model(res$model, file = pt_file)
  model <- load_cissvae_model(file = pt_file)

  out <- impute_with_cissvae(
    model                   = model,
    data                    = td$df,
    clusters                = td$clusters,
    categorical_column_map  = td$categorical_column_map,
    binary_feature_mask     = td$binary_feature_mask
  )

  expect_true(is.data.frame(out))
  expect_equal(ncol(out), ncol(td$df))
})