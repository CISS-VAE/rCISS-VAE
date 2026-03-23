# tests/testthat/helper-fixtures.R
# -------------------------------------------------------------------
# Shared fixtures & utilities for tests (R equivalent of pytest conftest.py)
# -------------------------------------------------------------------

# CRAN-friendly skips ---------------------------------------------------------
skip_if_no_python <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    testthat::skip("reticulate not installed")
  }
}

skip_if_no_cissvae_py <- function() {
  skip_if_no_python()
  ok <- tryCatch(
    reticulate::py_module_available("ciss_vae"),
    error = function(e) FALSE
  )
  if (!isTRUE(ok)) {
    testthat::skip("Python module 'ciss_vae' not available")
  }
}

# -------------------------------------------------------------------
# Multivariate normal generator WITHOUT MASS::mvrnorm()
# -------------------------------------------------------------------
rmvnorm2 <- function(n, mu, Sigma) {
  L <- chol(Sigma)
  Z <- matrix(rnorm(n * length(mu)), nrow = n)
  sweep(Z %*% L, 2, mu, "+")
}

# -------------------------------------------------------------------
# Data generators
# -------------------------------------------------------------------

#' Sample data with 2 clusters + noise and ~5% missing
#' @return data.frame (100 x 20)
make_sample_data <- function() {
  set.seed(42)

  Sigma <- diag(2) * 0.3
  cl1 <- rmvnorm2(50, mu = c(0, 0), Sigma = Sigma)
  cl2 <- rmvnorm2(50, mu = c(3, 3), Sigma = Sigma)

  # 18 noise features
  noise <- matrix(rnorm(100 * 18, sd = 0.5), nrow = 100)
  X <- cbind(rbind(cl1, cl2), noise)

  df <- as.data.frame(X)
  names(df) <- sprintf("feature_%d", seq_len(20) - 1)

  # ~5% missing
  set.seed(43)
  miss <- matrix(runif(nrow(df) * ncol(df)) < 0.05, nrow(df), ncol(df))
  df[miss] <- NA
  df
}

#' Longitudinal wide biomarkers: y1_*, y2_*, y3_* with ~5% missing
make_longitudinal_data <- function(n_samples = 100, n_times = 5) {
  set.seed(123)

  tp <- seq_len(n_times)
  y1_mu <- 0.5 * tp
  y2_mu <- sin(tp / 2)
  y3_mu <- log1p(tp)

  gen_block <- function(mu) sapply(mu, function(m) rnorm(n_samples, m, 0.3))

  df <- as.data.frame(cbind(
    gen_block(y1_mu),
    gen_block(y2_mu),
    gen_block(y3_mu)
  ))

  names(df) <- c(
    sprintf("y1_%d", tp),
    sprintf("y2_%d", tp),
    sprintf("y3_%d", tp)
  )

  # ~5% missing
  set.seed(321)
  miss <- matrix(runif(nrow(df) * ncol(df)) < 0.05, nrow(df), ncol(df))
  df[miss] <- NA
  df
}

#' Larger random dataset (1000 x 50) with ~5% missing
make_large_sample_data <- function() {
  set.seed(42)
  df <- as.data.frame(matrix(rnorm(1000 * 50), nrow = 1000))
  names(df) <- sprintf("feature_%d", seq_len(50) - 1)

  set.seed(43)
  miss <- matrix(runif(nrow(df) * ncol(df)) < 0.05, nrow(df), ncol(df))
  df[miss] <- NA
  df
}

## Categorical validation data

make_categorical_validation_data <- function() {
  data.frame(
    index = 1:6,
    X1    = c(1, 2, 3, 4, 5, 6),
    X2    = c(10, 20, 30, 40, 50, 60),
    B1    = c(0, 1, 0, 1, 0, 1),
    B2    = c(1, 0, 1, 0, 1, 0),
    C1b1  = c(1, 0, 1, 0, 1, 0),
    C1b2  = c(0, 1, 0, 1, 0, 1),
    C2b1  = c(1, 1, 0, 0, 1, 0),
    C2b2  = c(0, 0, 1, 1, 0, 1)
  )
}

# -------------------------------------------------------------------
# Minimal parameter sets (updated for modern run_cissvae and autotune API)
# -------------------------------------------------------------------

minimal_params_run <- function() {
  list(
    hidden_dims       = c(32L, 16L),
    latent_dim        = 8L,
    epochs            = 2L,
    batch_size        = 32L,
    max_loops         = 2L,
    patience          = 1L,
    epochs_per_loop   = 1L,
    verbose           = FALSE,
    n_clusters        = 2L,   # explicitly KMeans path
    layer_order_enc   = c("unshared", "shared"),
    layer_order_dec   = c("shared", "unshared"),
    return_model      = FALSE,
    return_dataset    = FALSE,
    return_silhouettes = FALSE
  )
}

minimal_params_autotune <- function() {
  list(
    n_trials             = 2L,
    study_name           = "vae_autotune_test",
    device_preference    = "cpu",
    show_progress        = FALSE,
    load_if_exists       = FALSE,
    seed                 = 42L,
    verbose              = FALSE,
    constant_layer_size  = FALSE,
    evaluate_all_orders  = FALSE,
    max_exhaustive_orders = 10L,

    num_hidden_layers    = c(1L, 2L),
    hidden_dims          = c(32L, 64L),
    latent_dim           = c(4L, 8L),
    latent_shared        = c(TRUE, FALSE),
    output_shared        = c(TRUE, FALSE),
    lr                   = c(1e-3, 1e-2),
    decay_factor         = c(0.9, 0.999),
    beta                 = 0.01,
    num_epochs           = 2L,
    batch_size           = 64L,
    num_shared_encode    = c(0L, 1L),
    num_shared_decode    = c(0L, 1L),
    encoder_shared_placement = c("at_end", "at_start"),
    decoder_shared_placement = c("at_end", "at_start"),
    refit_patience       = 1L,
    refit_loops          = 1L,
    epochs_per_loop      = 1L,
    reset_lr_refit       = c(TRUE, FALSE)
  )
}

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

local_tempdir <- function() withr::local_tempdir()

make_clusters_for <- function(df, k = 3L) {
  set.seed(7)

  complete <- stats::complete.cases(df)

  if (sum(complete) >= k) {
    km <- stats::kmeans(df[complete, , drop = FALSE], centers = k, nstart = 5)
    out <- integer(nrow(df))
    out[complete] <- km$cluster - 1L
    out[!complete] <- sample.int(k, sum(!complete), replace = TRUE) - 1L
    return(out)
  }

  # fallback random
  sample.int(k, nrow(df), replace = TRUE) - 1L
}


make_sample_data_binary = function() {
  set.seed(42)

  library(dplyr)
  library(MASS)

  # Generate clustered data
  Sigma <- diag(2) * 0.3
  cl1 <- MASS::mvrnorm(50, mu = c(0, 0), Sigma)
  cl2 <- MASS::mvrnorm(50, mu = c(3, 3), Sigma)

  # Add noise features
  noise <- matrix(rnorm(100 * 18, sd = 0.5), 100, 18)

  # Combine into feature matrix
  X <- cbind(rbind(cl1, cl2), noise)
  colnames(X) <- paste0("feature_", seq_len(ncol(X)) - 1)

  # Convert to data frame
  df <- as.data.frame(X)

  # Binary column 1: random Bernoulli
  df$binary_1 <- rbinom(n = nrow(df), size = 1, prob = 0.5)

  # Binary column 2: cluster-structure dependent
  df$binary_2 <- ifelse(
    df$feature_0^2 + df$feature_1^2 > 4,
    1,
    0
  )

  # Add index column (kept non-missing)
  df$index <- seq_len(nrow(df))
  # Introduce missing values (5%) to all columns EXCEPT index
  mask <- matrix(
    runif(nrow(df) * (ncol(df) - 1)) < 0.05,
    nrow = nrow(df),
    ncol = ncol(df) - 1
  )

  df[, -which(names(df) == "index")][mask] <- NA

  # Convert binaries to factors if desired
  df$binary_1 <- factor(df$binary_1, levels = c(0, 1))
  df$binary_2 <- factor(df$binary_2, levels = c(0, 1))

  return(df)
}

# -------------------------------
# Implement Categorical column map
# -------------------------------

test_that("categorical_column_map creates validation units using original category names", {
  skip_if_no_cissvae_py()

  df <- make_categorical_validation_data()

  binary_feature_mask <- c(
    index = FALSE,
    X1 = FALSE,
    X2 = FALSE,
    B1 = TRUE,
    B2 = TRUE,
    C1b1 = TRUE,
    C1b2 = TRUE,
    C2b1 = TRUE,
    C2b2 = TRUE
  )

  categorical_column_map <- list(
    C1 = c("C1b1", "C1b2"),
    C2 = c("C2b1", "C2b2")
  )

  res <- run_cissvae(
    data = df,
    index_col = "index",
    binary_feature_mask = binary_feature_mask,
    categorical_column_map = categorical_column_map,
    clusters = c(0, 0, 0, 1, 1, 1),
    val_proportion = 0.5,
    seed = 42,
    hidden_dims = c(4, 2),
    latent_dim = 2,
    layer_order_enc = c("unshared", "shared"),
    layer_order_dec = c("unshared", "shared"),
    epochs = 1,
    max_loops = 1,
    epochs_per_loop = 1,
    return_dataset = TRUE,
    verbose = FALSE
  )

  validation_units <- reticulate::py_to_r(res$cluster_dataset$validation_units)

  expect_true(all(c("X1", "X2", "B1", "B2", "C1", "C2") %in% names(validation_units)))
  expect_false(any(c("C1b1", "C1b2", "C2b1", "C2b2") %in% names(validation_units)))

  expect_equal(validation_units$C1$kind, "categorical")
  expect_equal(validation_units$C2$kind, "categorical")

  feature_names <- reticulate::py_to_r(res$cluster_dataset$feature_names)

  c1_cols <- unlist(validation_units$C1$cols) + 1L
  c2_cols <- unlist(validation_units$C2$cols) + 1L

  expect_equal(feature_names[c1_cols], c("C1b1", "C1b2"))
  expect_equal(feature_names[c2_cols], c("C2b1", "C2b2"))
})


test_that("categorical_column_map masks all dummy variables in a category together", {
  skip_if_no_cissvae_py()

  df <- make_categorical_validation_data()

  binary_feature_mask <- c(
    index = FALSE,
    X1 = FALSE,
    X2 = FALSE,
    B1 = TRUE,
    B2 = TRUE,
    C1b1 = TRUE,
    C1b2 = TRUE,
    C2b1 = TRUE,
    C2b2 = TRUE
  )

  categorical_column_map <- list(
    C1 = c("C1b1", "C1b2"),
    C2 = c("C2b1", "C2b2")
  )

  res <- run_cissvae(
    data = df,
    index_col = "index",
    binary_feature_mask = binary_feature_mask,
    categorical_column_map = categorical_column_map,
    clusters = c(0, 0, 0, 1, 1, 1),
    val_proportion = 0.5,
    seed = 42,
    hidden_dims = c(4, 2),
    latent_dim = 2,
    layer_order_enc = c("unshared", "shared"),
    layer_order_dec = c("unshared", "shared"),
    epochs = 1,
    max_loops = 1,
    epochs_per_loop = 1,
    return_dataset = TRUE,
    verbose = FALSE
  )

  feature_names <- reticulate::py_to_r(res$cluster_dataset$feature_names)
  val_mask <- reticulate::py_to_r(res$cluster_dataset$val_mask$cpu()$numpy())

  c1b1_idx <- match("C1b1", feature_names)
  c1b2_idx <- match("C1b2", feature_names)
  c2b1_idx <- match("C2b1", feature_names)
  c2b2_idx <- match("C2b2", feature_names)

  expect_equal(val_mask[, c1b1_idx], val_mask[, c1b2_idx])
  expect_equal(val_mask[, c2b1_idx], val_mask[, c2b2_idx])
})


test_that("categorical_column_map respects partially missing dummy groups", {
  skip_if_no_cissvae_py()

  df <- make_categorical_validation_data()

  # Make first row partially missing for C1:
  # this row should not be eligible for grouped validation masking for C1
  df$C1b1[1] <- NA
  df$C1b2[1] <- 0

  binary_feature_mask <- c(
    index = FALSE,
    X1 = FALSE,
    X2 = FALSE,
    B1 = TRUE,
    B2 = TRUE,
    C1b1 = TRUE,
    C1b2 = TRUE,
    C2b1 = TRUE,
    C2b2 = TRUE
  )

  categorical_column_map <- list(
    C1 = c("C1b1", "C1b2"),
    C2 = c("C2b1", "C2b2")
  )

  res <- run_cissvae(
    data = df,
    index_col = "index",
    binary_feature_mask = binary_feature_mask,
    categorical_column_map = categorical_column_map,
    clusters = c(0, 0, 0, 1, 1, 1),
    val_proportion = 0.5,
    seed = 42,
    hidden_dims = c(4, 2),
    latent_dim = 2,
    layer_order_enc = c("unshared", "shared"),
    layer_order_dec = c("unshared", "shared"),
    epochs = 1,
    max_loops = 1,
    epochs_per_loop = 1,
    return_dataset = TRUE,
    verbose = FALSE
  )

  feature_names <- reticulate::py_to_r(res$cluster_dataset$feature_names)
  val_mask <- reticulate::py_to_r(res$cluster_dataset$val_mask$cpu()$numpy())

  c1b1_idx <- match("C1b1", feature_names)
  c1b2_idx <- match("C1b2", feature_names)

  expect_false(isTRUE(val_mask[1, c1b1_idx]))
  expect_false(isTRUE(val_mask[1, c1b2_idx]))
})


test_that("categorical_column_map is stored correctly on returned cluster_dataset", {
  skip_if_no_cissvae_py()

  df <- make_categorical_validation_data()

  binary_feature_mask <- c(
    index = FALSE,
    X1 = FALSE,
    X2 = FALSE,
    B1 = TRUE,
    B2 = TRUE,
    C1b1 = TRUE,
    C1b2 = TRUE,
    C2b1 = TRUE,
    C2b2 = TRUE
  )

  categorical_column_map <- list(
    C1 = c("C1b1", "C1b2"),
    C2 = c("C2b1", "C2b2")
  )

  res <- run_cissvae(
    data = df,
    index_col = "index",
    binary_feature_mask = binary_feature_mask,
    categorical_column_map = categorical_column_map,
    clusters = c(0, 0, 0, 1, 1, 1),
    val_proportion = 0.5,
    seed = 42,
    hidden_dims = c(4, 2),
    latent_dim = 2,
    layer_order_enc = c("unshared", "shared"),
    layer_order_dec = c("unshared", "shared"),
    epochs = 1,
    max_loops = 1,
    epochs_per_loop = 1,
    return_dataset = TRUE,
    verbose = FALSE
  )

  returned_map <- reticulate::py_to_r(res$cluster_dataset$categorical_column_map)

  expect_equal(returned_map$C1, c("C1b1", "C1b2"))
  expect_equal(returned_map$C2, c("C2b1", "C2b2"))
})


test_that("categorical_column_map requires dummy columns to be marked binary", {
  skip_if_no_cissvae_py()

  df <- make_categorical_validation_data()

  # Deliberately mark one dummy as non-binary to trigger the validation error
  bad_binary_feature_mask <- c(
    index = FALSE,
    X1 = FALSE,
    X2 = FALSE,
    B1 = TRUE,
    B2 = TRUE,
    C1b1 = FALSE,
    C1b2 = TRUE,
    C2b1 = TRUE,
    C2b2 = TRUE
  )

  categorical_column_map <- list(
    C1 = c("C1b1", "C1b2"),
    C2 = c("C2b1", "C2b2")
  )

  expect_error(
    run_cissvae(
      data = df,
      index_col = "index",
      binary_feature_mask = bad_binary_feature_mask,
      categorical_column_map = categorical_column_map,
      clusters = c(0, 0, 0, 1, 1, 1),
      val_proportion = 0.5,
      seed = 42,
      hidden_dims = c(4, 2),
      latent_dim = 2,
      layer_order_enc = c("unshared", "shared"),
      layer_order_dec = c("unshared", "shared"),
      epochs = 1,
      max_loops = 1,
      epochs_per_loop = 1,
      return_dataset = TRUE,
      verbose = FALSE
    ),
    regexp = "not marked True in binary_feature_mask"
  )
})