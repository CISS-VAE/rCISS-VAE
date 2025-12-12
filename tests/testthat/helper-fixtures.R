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
