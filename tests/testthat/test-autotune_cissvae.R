# tests/testthat/test-autotune_cissvae.R
# -------------------------------------------------------------------
# Correct tests for autotune_cissvae(), aligned with current implementation
# -------------------------------------------------------------------
testthat::skip_if_not(
  reticulate::py_module_available("ciss_vae"),
  message = "Skipping Python tests because 'ciss_vae' is not available"
)

# ----------- Fixtures ------------------

make_sample_data_binary = function() {
  set.seed(42)

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

# ------------ TESTS ------------------

test_that("Autotune handles binary feature matrix", {
  skip_if_no_cissvae_py()

  data = make_sample_data_binary()

  ## make binary feature mask 
  bfm = c(rep(FALSE, 20), TRUE, TRUE, FALSE)
  names(bfm) = colnames(data)

  clusters = make_clusters_for(data, k = 3L)
  res_fixed <- autotune_cissvae(
    data = data,
    clusters = clusters,
    n_trials = 1L,
    device_preference = "cpu",
    load_if_exists = FALSE,

    num_hidden_layers = 2L,
    hidden_dims       = 64L,
    latent_dim        = 16L,
    latent_shared     = TRUE,
    output_shared     = FALSE,
    lr                = 1e-3,
    decay_factor      = 0.95,
    weight_decay      = 0.001,
    beta              = 0.01,
    num_epochs        = 1L,
    batch_size        = 32L,
    num_shared_encode = 1L,
    num_shared_decode = 1L,
    encoder_shared_placement = c("at_end", "at_start"),
    decoder_shared_placement = c("at_end", "at_start"),
    refit_patience    = 1L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,
    reset_lr_refit    = TRUE,

    show_progress = FALSE,
    verbose       = FALSE
  )

  expect_true(all(c("imputed_dataset","model","study","results") %in% names(res_fixed)))

  expect_true(all(reticulate::py_to_r(res_fixed$cluster_dataset$binary_feature_mask) == c(rep(FALSE, 20), TRUE, TRUE)))
})


test_that("Autotune accepts both fixed and ranged parameter styles", {
  skip_if_no_cissvae_py()

  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  # --- Fixed hyperparameters (degenerate ranges allowed) ---------------------
  res_fixed <- autotune_cissvae(
    data = df,
    clusters = clusters,
    n_trials = 1L,
    device_preference = "cpu",
    load_if_exists = FALSE,

    num_hidden_layers = 2L,
    hidden_dims       = 64L,
    latent_dim        = 16L,
    latent_shared     = TRUE,
    output_shared     = FALSE,
    lr                = 1e-3,
    decay_factor      = 0.95,
    weight_decay      = 0.001,
    beta              = 0.01,
    num_epochs        = 1L,
    batch_size        = 32L,
    num_shared_encode = 1L,
    num_shared_decode = 1L,
    encoder_shared_placement = c("at_end", "at_start"),
    decoder_shared_placement = c("at_end", "at_start"),
    refit_patience    = 1L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,
    reset_lr_refit    = TRUE,

    show_progress = FALSE,
    verbose       = FALSE
  )

  expect_true(all(c("imputed_dataset","model","study","results") %in% names(res_fixed)))
  expect_s3_class(res_fixed$imputed_dataset, "data.frame")
  expect_s3_class(res_fixed$results, "data.frame")

  # --- Ranged / tunable hyperparameters -------------------------------------
  res_tune <- autotune_cissvae(
    data = df,
    clusters = clusters,
    n_trials = 2L,
    device_preference = "cpu",
    load_if_exists = FALSE,

    num_hidden_layers = c(1L, 3L),
    hidden_dims       = c(32L, 64L),
    latent_dim        = c(8L, 16L),
    latent_shared     = c(TRUE, FALSE),
    output_shared     = c(TRUE, FALSE),
    lr                = c(1e-4, 1e-3),
    decay_factor      = c(0.9, 0.999),
    weight_decay      = 0.001,
    beta              = 0.01,
    num_epochs        = 1L,
    batch_size        = 32L,
    num_shared_encode = c(0L, 1L),
    num_shared_decode = c(0L, 1L),
    encoder_shared_placement = c("at_end", "at_start"),
    decoder_shared_placement = c("at_end", "at_start"),
    refit_patience    = 1L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,
    reset_lr_refit    = c(TRUE, FALSE),

    show_progress = FALSE,
    verbose       = FALSE
  )

  expect_true(all(c("imputed_dataset","model","study","results") %in% names(res_tune)))
  expect_equal(nrow(res_tune$results), 2L)
})

# ---------------------------------------------------------------------------

test_that("n_trials controls number of results rows", {
  skip_if_no_cissvae_py()

  df <- make_sample_data()
  clusters <- make_clusters_for(df)

  for (n in c(1L, 2L, 3L)) {
    res <- autotune_cissvae(
      data = df,
      clusters = clusters,
      n_trials = n,
      device_preference = "cpu",
      load_if_exists = FALSE,

      num_hidden_layers = c(1L, 2L),
      hidden_dims       = c(32L, 64L),
      latent_dim        = c(8L, 16L),
      lr                = c(1e-4, 1e-3),
      decay_factor      = c(0.9, 0.99),
      weight_decay      = 0.001,
      num_epochs        = 1L,
      batch_size        = 32L,
      num_shared_encode = c(0L, 1L),
      num_shared_decode = c(0L, 1L),
      refit_patience    = 1L,
      refit_loops       = 1L,
      epochs_per_loop   = 1L,

      show_progress = FALSE,
      verbose       = FALSE
    )

    expect_equal(nrow(res$results), n)
  }
})

# ---------------------------------------------------------------------------

test_that("evaluate_all_orders = TRUE respects max_exhaustive_orders", {
  skip_if_no_cissvae_py()

  df <- make_sample_data()
  clusters <- make_clusters_for(df)

  res <- autotune_cissvae(
    data = df,
    clusters = clusters,
    n_trials = 1L,
    evaluate_all_orders = TRUE,
    max_exhaustive_orders = 5L,

    num_hidden_layers = 4L,
    hidden_dims       = 32L,
    latent_dim        = 16L,
    num_shared_encode = c(0L, 1L, 2L),
    num_shared_decode = c(0L, 1L, 2L),

    device_preference = "cpu",
    load_if_exists = FALSE,
    num_epochs      = 1L,
    batch_size      = 32L,
    refit_loops     = 1L,
    epochs_per_loop = 1L,
    show_progress   = FALSE,
    verbose         = FALSE
  )

  expect_true(all(c("imputed_dataset","model","study","results") %in% names(res)))
})

# ---------------------------------------------------------------------------

test_that("invalid placement strategies are rejected", {
  skip_if_no_cissvae_py()

  df <- make_sample_data()
  clusters <- make_clusters_for(df)

  expect_error(
    autotune_cissvae(
      data = df,
      clusters = clusters,
      n_trials = 1L,
      encoder_shared_placement = "NOPE"
    ),
    "Invalid encoder_shared_placement"
  )

  expect_error(
    autotune_cissvae(
      data = df,
      clusters = clusters,
      n_trials = 1L,
      decoder_shared_placement = "BAD"
    ),
    "Invalid decoder_shared_placement"
  )
})

# ---------------------------------------------------------------------------

test_that("seed reproducibility preserves the number of rows", {
  skip_if_no_cissvae_py()

  df <- make_sample_data()
  clusters <- make_clusters_for(df)
  p <- minimal_params_autotune()

  run_one <- function() {
    autotune_cissvae(
      data = df,
      clusters = clusters,
      n_trials = 2L,
      seed = 42L,
      device_preference = "cpu",
      load_if_exists = FALSE,

      # from minimal_params_autotune()
      num_hidden_layers = p$num_hidden_layers,
      hidden_dims       = p$hidden_dims,
      latent_dim        = p$latent_dim,
      lr                = c(1e-4, 1e-3),
      decay_factor      = c(0.9, 0.99),
      weight_decay      = 0.001,
      num_epochs        = p$num_epochs,
      batch_size        = p$batch_size,
      num_shared_encode = p$num_shared_encode,
      num_shared_decode = p$num_shared_decode,
      encoder_shared_placement = p$encoder_shared_placement,
      decoder_shared_placement = p$decoder_shared_placement,
      refit_patience    = p$refit_patience,
      refit_loops       = p$refit_loops,
      epochs_per_loop   = p$epochs_per_loop,
      reset_lr_refit    = p$reset_lr_refit,

      show_progress = FALSE,
      verbose       = FALSE
    )
  }

  res1 <- run_one()
  res2 <- run_one()

  expect_equal(nrow(res1$results), nrow(res2$results))
})

# ---------------------------------------------------------------------------

test_that("tiny optimization path runs end-to-end", {
  skip_if_no_cissvae_py()

  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  res <- autotune_cissvae(
    data = df,
    clusters = clusters,
    n_trials = 2L,
    device_preference = "cpu",
    load_if_exists = FALSE,

    num_hidden_layers = 1L,
    hidden_dims       = 16L,
    latent_dim        = c(4L, 8L),
    lr                = 0.01,
    decay_factor      = 0.95,
    weight_decay      = 0.001,
    num_epochs        = 1L,
    batch_size        = 32L,
    num_shared_encode = 0L,
    num_shared_decode = 1L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,

    show_progress = FALSE,
    verbose       = FALSE
  )

  expect_true(all(c("imputed_dataset","model","study","results") %in% names(res)))
  expect_equal(nrow(res$results), 2L)
})
