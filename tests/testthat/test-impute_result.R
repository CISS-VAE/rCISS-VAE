# =============================================================================
# tests/testthat/test-impute_result.R
#
# Tests for:
#   - new_impute_result()
#   - print.impute_result()
#   - save_impute_result()
#   - load_impute_result()
#   - .cissvae_config()  (internal)
# =============================================================================

# -----------------------------------------------------------------------------
# Helpers: build minimal impute_result objects without needing Python
# -----------------------------------------------------------------------------

.make_minimal_result <- function() {
  new_impute_result(
    imputed_dataset = data.frame(x1 = 1:5, x2 = rnorm(5)),
    raw_data        = data.frame(x1 = 1:5, x2 = rnorm(5))
  )
}

.make_full_result <- function() {
  new_impute_result(
    imputed_dataset  = data.frame(x1 = 1:10, x2 = rnorm(10)),
    raw_data         = data.frame(x1 = 1:10, x2 = rnorm(10)),
    clusters         = c(0L, 0L, 1L, 1L, 2L, 2L, 0L, 1L, 2L, 0L),
    silhouette_width = runif(10),
    training_history = data.frame(epoch = 1:5, loss = runif(5)),
    val_data         = data.frame(x1 = c(1, NA, 3, NA, 5,  6,  7, NA,  9, 10),
                                  x2 = c(NA, 2, NA,  4, NA, NA, 7,  8, NA, 10)),
    val_imputed      = data.frame(x1 = rnorm(10), x2 = rnorm(10))
  )
}

# =============================================================================
# 1. new_impute_result
# =============================================================================

test_that("new_impute_result returns correct class", {
  res <- .make_minimal_result()
  expect_s3_class(res, "impute_result")
})

test_that("new_impute_result is also a list", {
  res <- .make_minimal_result()
  expect_true(is.list(res))
})

test_that("new_impute_result preserves all supplied fields", {
  res <- .make_minimal_result()
  expect_named(res, c("imputed_dataset", "raw_data"), ignore.order = TRUE)
})

test_that("new_impute_result fields are accessible with $", {
  res <- .make_minimal_result()
  expect_s3_class(res$imputed_dataset, "data.frame")
  expect_s3_class(res$raw_data, "data.frame")
})

test_that("new_impute_result fields are accessible with [[", {
  res <- .make_minimal_result()
  expect_s3_class(res[["imputed_dataset"]], "data.frame")
})

test_that("new_impute_result with no fields is still an impute_result", {
  res <- new_impute_result()
  expect_s3_class(res, "impute_result")
  expect_length(res, 0)
})

# =============================================================================
# 2. print.impute_result
# =============================================================================

test_that("print.impute_result returns x invisibly", {
  res <- .make_minimal_result()
  out <- withVisible(print(res))
  expect_false(out$visible)
  expect_identical(out$value, res)
})

test_that("print.impute_result produces output", {
  res <- .make_minimal_result()
  expect_output(print(res), regexp = "<impute_result>")
})

test_that("print.impute_result shows dataset dimensions", {
  res <- .make_minimal_result()
  expect_output(print(res), regexp = "5 rows x 2 columns")
})

test_that("print.impute_result shows cluster summary when clusters present", {
  res <- .make_full_result()
  expect_output(print(res), regexp = "cluster(s)", fixed = TRUE)
})

test_that("print.impute_result reports absent clusters gracefully", {
  res <- .make_minimal_result()
  expect_output(print(res), regexp = "<not present>")
})

test_that("print.impute_result respects n argument", {
  res <- .make_full_result()
  # n = 2 should not error and should produce less output than n = 10
  out_2  <- capture.output(print(res, n = 2))
  out_10 <- capture.output(print(res, n = 10))
  expect_true(length(out_10) >= length(out_2))
})

test_that("print.impute_result n = 0 does not error", {
  res <- .make_minimal_result()
  expect_no_error(print(res, n = 0))
})

test_that("print.impute_result marks present optional fields with [x]", {
  res <- .make_full_result()
  out <- capture.output(print(res))
  # silhouette_width is present in full result
  sil_line <- out[grepl("silhouette_width", out)]
  expect_true(any(grepl("[x]", sil_line, fixed = TRUE)))
})

test_that("print.impute_result marks absent optional fields with [ ]", {
  res <- .make_minimal_result()
  out <- capture.output(print(res))
  # model is absent in minimal result
  model_line <- out[grepl("model", out)]
  expect_true(any(grepl("[ ]", model_line, fixed = TRUE)))
})

test_that("print.impute_result shows training history dimensions", {
  res <- .make_full_result()
  expect_output(print(res), regexp = "5 epochs x 2 columns")
})

# =============================================================================
# 3. save_impute_result  (no Python / no model)
# =============================================================================

test_that("save_impute_result errors on non-impute_result input", {
  expect_error(
    save_impute_result(list(a = 1), dir = tempdir()),
    regexp = "impute_result"
  )
})

test_that("save_impute_result writes an RDS file", {
  res <- .make_minimal_result()
  dir <- file.path(tempdir(), "test_save_minimal")
  on.exit(unlink(dir, recursive = TRUE))

  save_impute_result(res, dir = dir, name = "test")
  expect_true(file.exists(file.path(dir, "test.rds")))
})

test_that("save_impute_result does NOT write a .pt file when no model present", {
  res <- .make_minimal_result()
  dir <- file.path(tempdir(), "test_save_no_model")
  on.exit(unlink(dir, recursive = TRUE))

  save_impute_result(res, dir = dir, name = "test")
  expect_false(file.exists(file.path(dir, "test_state_dict.pt")))
})

test_that("save_impute_result creates the output directory if needed", {
  res <- .make_minimal_result()
  dir <- file.path(tempdir(), "new_dir_xyz_123")
  on.exit(unlink(dir, recursive = TRUE))

  expect_false(dir.exists(dir))
  save_impute_result(res, dir = dir, name = "test")
  expect_true(dir.exists(dir))
})

test_that("save_impute_result errors when files exist and overwrite = FALSE", {
  res <- .make_minimal_result()
  dir <- file.path(tempdir(), "test_overwrite")
  on.exit(unlink(dir, recursive = TRUE))

  save_impute_result(res, dir = dir, name = "test")
  expect_error(
    save_impute_result(res, dir = dir, name = "test", overwrite = FALSE),
    regexp = "already exist"
  )
})

test_that("save_impute_result overwrites silently when overwrite = TRUE", {
  res <- .make_minimal_result()
  dir <- file.path(tempdir(), "test_overwrite_true")
  on.exit(unlink(dir, recursive = TRUE))

  save_impute_result(res, dir = dir, name = "test")
  expect_no_error(
    save_impute_result(res, dir = dir, name = "test", overwrite = TRUE)
  )
})

test_that("save_impute_result returns invisible list with correct names", {
  res <- .make_minimal_result()
  dir <- file.path(tempdir(), "test_return_val")
  on.exit(unlink(dir, recursive = TRUE))

  out <- withVisible(save_impute_result(res, dir = dir, name = "test"))
  expect_false(out$visible)
  expect_named(out$value, c("meta", "model"), ignore.order = TRUE)
})

# =============================================================================
# 4. load_impute_result  (no Python / no model)
# =============================================================================

test_that("load_impute_result round-trips a minimal result correctly", {
  res <- .make_minimal_result()
  dir <- file.path(tempdir(), "test_roundtrip_minimal")
  on.exit(unlink(dir, recursive = TRUE))

  save_impute_result(res, dir = dir, name = "test")
  res2 <- load_impute_result(dir = dir, name = "test")

  expect_s3_class(res2, "impute_result")
  expect_equal(res2$imputed_dataset, res$imputed_dataset)
  expect_equal(res2$raw_data,        res$raw_data)
})

test_that("load_impute_result round-trips a full result correctly", {
  res <- .make_full_result()
  dir <- file.path(tempdir(), "test_roundtrip_full")
  on.exit(unlink(dir, recursive = TRUE))

  save_impute_result(res, dir = dir, name = "full")
  res2 <- load_impute_result(dir = dir, name = "full")

  expect_s3_class(res2, "impute_result")
  expect_equal(res2$clusters,         res$clusters)
  expect_equal(res2$silhouette_width,  res$silhouette_width)
  expect_equal(res2$training_history,  res$training_history)
  expect_equal(res2$val_data,          res$val_data)
  expect_equal(res2$val_imputed,       res$val_imputed)
})

test_that("load_impute_result errors when RDS file not found", {
  expect_error(
    load_impute_result(dir = tempdir(), name = "does_not_exist_xyz"),
    regexp = "No file found"
  )
})

test_that("load_impute_result restores class even after plain saveRDS", {
  # If someone saved with base saveRDS, class should still be restored
  res <- .make_minimal_result()
  dir <- file.path(tempdir(), "test_plain_rds")
  on.exit(unlink(dir, recursive = TRUE))
  dir.create(dir)

  # strip class and save manually
  plain <- unclass(res)
  saveRDS(plain, file.path(dir, "test.rds"))

  res2 <- load_impute_result(dir = dir, name = "test")
  expect_s3_class(res2, "impute_result")
})

# =============================================================================
# 5. Python-dependent tests  (skipped if reticulate / ciss-vae unavailable)
# =============================================================================

skip_if_no_python <- function() {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_available(), "Python not available")
  skip_if_not(
    tryCatch({
      reticulate::py_run_string(
        "import importlib; importlib.import_module('ciss-vae.classes')"
      )
      TRUE
    }, error = function(e) FALSE),
    "ciss-vae Python package not available"
  )
}

test_that(".cissvae_config extracts all required fields", {
  skip_if_no_python()

  mod <- reticulate::py_run_string(
    "import importlib; _mod = importlib.import_module('ciss-vae.classes')"
  )$`_mod`

  model <- mod$CISSVAE(
    input_dim       = 10L,
    hidden_dims     = list(8L, 4L),
    layer_order_enc = list("shared", "unshared"),
    layer_order_dec = list("unshared", "shared"),
    latent_shared   = TRUE,
    latent_dim      = 2L,
    output_shared   = TRUE,
    num_clusters    = 3L,
    debug           = FALSE
  )

  cfg <- .cissvae_config(model)

  expect_type(cfg, "list")
  expect_named(cfg, c(
    "input_dim", "hidden_dims", "layer_order_enc", "layer_order_dec",
    "latent_shared", "latent_dim", "output_shared",
    "num_clusters", "binary_feature_mask"
  ), ignore.order = TRUE)

  expect_equal(cfg$input_dim,    10L)
  expect_equal(cfg$latent_dim,    2L)
  expect_equal(cfg$num_clusters,  3L)
  expect_true(cfg$latent_shared)
  expect_true(cfg$output_shared)
  expect_length(cfg$binary_feature_mask, 10L)
  expect_type(cfg$binary_feature_mask, "logical")
})

test_that("save_impute_result writes a .pt file when model is present", {
  skip_if_no_python()

  mod <- reticulate::py_run_string(
    "import importlib; _mod = importlib.import_module('ciss-vae.classes')"
  )$`_mod`

  model <- mod$CISSVAE(
    input_dim       = 10L,
    hidden_dims     = list(8L),
    layer_order_enc = list("shared"),
    layer_order_dec = list("shared"),
    latent_shared   = TRUE,
    latent_dim      = 2L,
    output_shared   = TRUE,
    num_clusters    = 2L,
    debug           = FALSE
  )

  res <- new_impute_result(
    imputed_dataset = data.frame(x = 1:5),
    model           = model
  )
  dir <- file.path(tempdir(), "test_save_with_model")
  on.exit(unlink(dir, recursive = TRUE))

  save_impute_result(res, dir = dir, name = "m")
  expect_true(file.exists(file.path(dir, "m_state_dict.pt")))
})

test_that("load_impute_result rebuilds a CISSVAE model correctly", {
  skip_if_no_python()

  mod <- reticulate::py_run_string(
    "import importlib; _mod = importlib.import_module('ciss-vae.classes')"
  )$`_mod`

  model <- mod$CISSVAE(
    input_dim       = 10L,
    hidden_dims     = list(8L),
    layer_order_enc = list("shared"),
    layer_order_dec = list("shared"),
    latent_shared   = TRUE,
    latent_dim      = 2L,
    output_shared   = TRUE,
    num_clusters    = 2L,
    debug           = FALSE
  )

  res <- new_impute_result(
    imputed_dataset = data.frame(x = 1:5),
    model           = model
  )
  dir <- file.path(tempdir(), "test_load_model")
  on.exit(unlink(dir, recursive = TRUE))

  save_impute_result(res, dir = dir, name = "m")
  res2 <- load_impute_result(dir = dir, name = "m", device = "cpu")

  expect_s3_class(res2, "impute_result")
  expect_false(is.null(res2$model))
  expect_true(reticulate::py_has_attr(res2$model, "state_dict"))
  expect_equal(
    reticulate::py_to_r(res2$model$input_dim),
    10L
  )
})

test_that("load_impute_result errors when model_config is missing", {
  skip_if_no_python()

  res <- .make_minimal_result()
  # inject a fake model_path without a model_config
  res$model_path <- "fake_state_dict.pt"

  dir <- file.path(tempdir(), "test_missing_config")
  on.exit(unlink(dir, recursive = TRUE))
  dir.create(dir)
  saveRDS(res, file.path(dir, "test.rds"))

  expect_error(
    load_impute_result(dir = dir, name = "test"),
    regexp = "model_config"
  )
})