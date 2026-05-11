# =============================================================================
# tests/testthat/test-impute_result.R
# =============================================================================

# -----------------------------------------------------------------------------
# Helpers
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
    val_data         = data.frame(x1 = c(1, NA, 3, NA, 5, 6, 7, NA, 9, 10),
                                 x2 = c(NA, 2, NA, 4, NA, NA, 7, 8, NA, 10)),
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

test_that("new_impute_result fields accessible", {
  res <- .make_minimal_result()
  expect_s3_class(res$imputed_dataset, "data.frame")
})

test_that("new_impute_result empty still valid", {
  res <- new_impute_result()
  expect_s3_class(res, "impute_result")
})

# =============================================================================
# 2. print.impute_result
# =============================================================================

test_that("print.impute_result works", {
  res <- .make_minimal_result()
  expect_output(print(res), "<impute_result>")
})

# =============================================================================
# 3. save/load (no Python)
# =============================================================================

test_that("save/load roundtrip minimal", {
  res <- .make_minimal_result()
  dir <- tempfile()
  on.exit(unlink(dir, recursive = TRUE))

  save_impute_result(res, dir = dir)
  res2 <- load_impute_result(dir = dir)

  expect_equal(res2$imputed_dataset, res$imputed_dataset)
})

# =============================================================================
# 4. Python-dependent tests
# =============================================================================

skip_if_no_python <- function() {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_available())
}

# -----------------------------------------------------------------------------
# 🔴 UPDATED: activation_groups instead of binary_feature_mask
# -----------------------------------------------------------------------------

test_that(".cissvae_config extracts activation_groups", {
  skip_if_no_python()

  mod <- reticulate::import("ciss_vae.classes")

  model <- mod$CISSVAE(
    input_dim       = 10L,
    hidden_dims     = list(8L),
    layer_order_enc = list("shared"),
    layer_order_dec = list("shared"),
    latent_shared   = TRUE,
    latent_dim      = 2L,
    output_shared   = TRUE,
    num_clusters    = 2L
  )

  cfg <- .cissvae_config(model)

  expect_named(cfg, c(
    "input_dim", "hidden_dims", "layer_order_enc", "layer_order_dec",
    "latent_shared", "latent_dim", "output_shared",
    "num_clusters", "activation_groups"
  ), ignore.order = TRUE)

  expect_true(is.list(cfg$activation_groups))
  expect_true(length(cfg$activation_groups) > 0)
})

# -----------------------------------------------------------------------------
# NEW: activation_groups structure validation
# -----------------------------------------------------------------------------

test_that("activation_groups structure is valid", {
  skip_if_no_python()

  mod <- reticulate::import("ciss_vae.classes")

  model <- mod$CISSVAE(
    input_dim = 6L,
    hidden_dims = list(4L),
    layer_order_enc = list("shared"),
    layer_order_dec = list("shared"),
    latent_shared = TRUE,
    latent_dim = 2L,
    output_shared = TRUE,
    num_clusters = 2L
  )

  cfg <- .cissvae_config(model)

  ag <- cfg$activation_groups

  expect_true(is.list(ag))
  expect_true(all(unlist(ag) >= 0))
})

# -----------------------------------------------------------------------------
# 🔴 UPDATED: save/load preserves activation_groups
# -----------------------------------------------------------------------------
test_that("save/load preserves activation_groups", {
  library(tidyverse)
  library(fastDummies)
  skip_if_no_python()

  data = data.frame(
    x1 = rnorm(100, 50),
    x2 = rnorm(100, 20),
    c1 = sample(c("a", "b", "c"), 100, replace = TRUE),
    b1 = rbinom(100, 1, 0.5),
    c2 = sample(c("d", "e"), 100, replace = TRUE),
    x3 = rnorm(100, -2),
    b2 = rbinom(100, 1, prob = 0.3)
  )

  data_missing = data  %>%
    dplyr::mutate(across(everything(), ~ ifelse(runif(n()) < 0.3, NA, .)))

  data_dummies = data_missing %>% 
      dummy_cols(select_columns = c("c1", "c2"),
      ignore_na = TRUE,
      remove_first_dummy = FALSE,
      remove_selected_columns = TRUE) 

  ccm = list("c1" = c("c1_a", "c1_b", "c1_c"), "c2" = c("c2_d", "c2_e"))

  bfm = c(FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)

  names(bfm) = colnames(data_dummies)

  res = run_cissvae(
    data = data_dummies, 
    n_clusters = 3, 
    epochs = 2,  
    # hidden_dims = c(64),
    # layer_order_enc = c("u"),
    # layer_order_dec = c("s"),
    categorical_column_map = ccm, 
    binary_feature_mask = bfm, 
    val_proportion = 0.3, 
    return_validation_dataset = TRUE)

  dir <- tempfile()
  on.exit(unlink(dir, recursive = TRUE))

  save_impute_result(res, dir = dir)
  res2 <- load_impute_result(dir = dir)

  expect_true(reticulate::py_has_attr(res2$model, "activation_groups"))

  ag <- reticulate::py_to_r(res2$model$activation_groups)

  expect_true(is.list(ag))
  expect_true(length(ag) > 0)
})

# -----------------------------------------------------------------------------
# NEW: failure case
# -----------------------------------------------------------------------------

test_that("load_impute_result errors when activation_groups missing", {
  skip_if_no_python()

  res <- .make_minimal_result()
  res$model_path <- "fake.pt"
  res$model_config <- list(input_dim = 10L)

  dir <- tempfile()
  on.exit(unlink(dir, recursive = TRUE))
  dir.create(dir)

  saveRDS(res, file.path(dir, "result.rds"))

  expect_error(
    load_impute_result(dir = dir),
    "activation_groups"
  )
})