# tests/testthat/test-autotune.R

test_that("Search-space style arguments accept fixed and ranged values", {
  skip_if_no_cissvae_py()

  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  # Fixed-only style
  res_fixed <- autotune_cissvae(
    data = df,
    clusters = clusters,
    n_trials = 2L,
    device_preference = "cpu",
    load_if_exists = FALSE,
    # "SearchSpace" fixed style:
    num_hidden_layers = 2L,
    hidden_dims       = 64L,
    latent_dim        = 16L,
    latent_shared     = TRUE,
    output_shared     = FALSE,
    lr                = c(1e-3, 1e-3),   # degenerate range = fixed
    decay_factor      = c(0.95, 0.95),   # degenerate range = fixed
    beta              = 0.01,
    num_epochs        = 2L,
    batch_size        = 64L,
    num_shared_encode = 1L,
    num_shared_decode = 1L,
    encoder_shared_placement = "at_end",
    decoder_shared_placement = "at_start",
    refit_patience    = 1L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,
    reset_lr_refit    = TRUE,
    show_progress     = FALSE,
    verbose           = FALSE
  )

  expect_true(all(c("imputed","model","study","results") %in% names(res_fixed)))
  expect_s3_class(res_fixed$imputed, "data.frame")
  expect_s3_class(res_fixed$results, "data.frame")

  # Tunable/ranged style
  res_tune <- autotune_cissvae(
    data = df,
    clusters = clusters,
    n_trials = 3L,
    device_preference = "cpu",
    load_if_exists = FALSE,
    num_hidden_layers = c(1L, 3L),
    hidden_dims       = c(32L, 64L),
    latent_dim        = c(8L, 16L),
    latent_shared     = c(TRUE, FALSE),
    output_shared     = c(TRUE, FALSE),
    lr                = c(1e-4, 1e-3),
    decay_factor      = c(0.9, 0.999),
    beta              = 0.01,
    num_epochs        = 2L,
    batch_size        = 32L,
    num_shared_encode = c(0L, 1L),
    num_shared_decode = c(0L, 1L),
    encoder_shared_placement = c("at_end", "at_start"),
    decoder_shared_placement = c("at_end", "at_start"),
    refit_patience    = 1L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,
    reset_lr_refit    = c(TRUE, FALSE),
    show_progress     = FALSE,
    verbose           = FALSE
  )

  expect_true(all(c("imputed","model","study","results") %in% names(res_tune)))
  expect_equal(nrow(res_tune$results), 3L)
})

test_that("n_trials controls number of results rows", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  for (n in c(1L, 3L, 5L)) {
    res <- autotune_cissvae(
      data = df, clusters = clusters,
      n_trials = n, device_preference = "cpu", load_if_exists = FALSE,
      num_hidden_layers = c(1L, 2L),
      hidden_dims       = c(32L, 64L),
      latent_dim        = c(8L, 16L),
      num_epochs        = 2L,
      batch_size        = 32L,
      num_shared_encode = c(0L, 1L),
      num_shared_decode = c(0L, 1L),
      refit_patience    = 1L,
      refit_loops       = 1L,
      epochs_per_loop   = 1L,
      show_progress     = FALSE,
      verbose           = FALSE
    )
    expect_equal(nrow(res$results), n)
  }
})

test_that("evaluate_all_orders=TRUE runs with a cap on combinations", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  res <- autotune_cissvae(
    data = df, clusters = clusters,
    n_trials = 2L,
    device_preference = "cpu",
    load_if_exists = FALSE,
    evaluate_all_orders = TRUE,
    max_exhaustive_orders = 5L,
    num_hidden_layers = 4L,  # many combos
    hidden_dims       = 64L,
    latent_dim        = 16L,
    num_shared_encode = c(0L, 1L, 2L, 3L),
    num_shared_decode = c(0L, 1L, 2L, 3L),
    num_epochs        = 1L,
    batch_size        = 64L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,
    show_progress     = FALSE,
    verbose           = FALSE
  )
  expect_true(all(c("imputed","model","study","results") %in% names(res)))
})

test_that("evaluate_all_orders=FALSE runs", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  res <- autotune_cissvae(
    data = df, clusters = clusters,
    n_trials = 2L, device_preference = "cpu", load_if_exists = FALSE,
    evaluate_all_orders = FALSE,
    num_hidden_layers = c(1L, 3L),
    hidden_dims       = c(32L, 64L),
    latent_dim        = c(8L, 16L),
    num_epochs        = 2L,
    batch_size        = 32L,
    num_shared_encode = c(0L, 1L),
    num_shared_decode = c(0L, 1L),
    refit_patience    = 1L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,
    show_progress     = FALSE,
    verbose           = FALSE
  )
  expect_true(all(c("imputed","model","study","results") %in% names(res)))
})

test_that("return format is consistent", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  res <- autotune_cissvae(
    data = df, clusters = clusters,
    n_trials = 2L, device_preference = "cpu", load_if_exists = FALSE,
    num_hidden_layers = c(1L, 2L),
    hidden_dims       = c(32L, 64L),
    latent_dim        = c(8L, 16L),
    num_epochs        = 2L,
    batch_size        = 32L,
    num_shared_encode = c(0L, 1L),
    num_shared_decode = c(0L, 1L),
    refit_patience    = 1L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,
    show_progress     = FALSE,
    verbose           = FALSE
  )

  expect_named(res, c("imputed","model","study","results"), ignore.order = TRUE)
  expect_s3_class(res$imputed, "data.frame")
  expect_s3_class(res$results, "data.frame")
})

test_that("constant_layer_size=TRUE runs", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  res <- autotune_cissvae(
    data = df, clusters = clusters,
    n_trials = 2L, device_preference = "cpu", load_if_exists = FALSE,
    constant_layer_size = TRUE,
    num_hidden_layers = 3L,
    hidden_dims       = c(64L, 128L),  # should be treated as constant
    latent_dim        = 16L,
    num_epochs        = 2L,
    batch_size        = 32L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,
    show_progress     = FALSE,
    verbose           = FALSE
  )
  expect_true(all(c("imputed","model","study","results") %in% names(res)))
})

test_that("invalid parameter values are validated", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  expect_error(
    autotune_cissvae(
      data = df, clusters = clusters,
      n_trials = 1L, device_preference = "cpu", load_if_exists = FALSE,
      encoder_shared_placement = "not_a_valid_option"  # invalid
    ),
    "Invalid encoder_shared_placement"
  )

  expect_error(
    autotune_cissvae(
      data = df, clusters = clusters,
      n_trials = 1L, device_preference = "cpu", load_if_exists = FALSE,
      decoder_shared_placement = "nope"  # invalid
    ),
    "Invalid decoder_shared_placement"
  )
})

test_that("seed reproducibility: same nrow(results) for same seed", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  p <- minimal_params_autotune()

  res1 <- autotune_cissvae(
    data = df, clusters = clusters,
    n_trials = 2L, device_preference = "cpu", load_if_exists = FALSE,
    seed = 42L,
    num_hidden_layers = p$num_hidden_layers,
    hidden_dims       = p$hidden_dims,
    latent_dim        = p$latent_dim,
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
    show_progress     = FALSE, verbose = FALSE
  )

  res2 <- autotune_cissvae(
    data = df, clusters = clusters,
    n_trials = 2L, device_preference = "cpu", load_if_exists = FALSE,
    seed = 42L,
    num_hidden_layers = p$num_hidden_layers,
    hidden_dims       = p$hidden_dims,
    latent_dim        = p$latent_dim,
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
    show_progress     = FALSE, verbose = FALSE
  )

  expect_equal(nrow(res1$results), nrow(res2$results))
})

test_that("actual tiny optimization runs end-to-end", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  clusters <- make_clusters_for(df, k = 3L)

  res <- autotune_cissvae(
    data = df,
    clusters = clusters,
    n_trials = 2L,
    study_name = "vae_autotune_rtest",
    device_preference = "cpu",
    show_progress = FALSE,
    load_if_exists = FALSE,
    # minimal space
    num_hidden_layers = 1L,
    hidden_dims       = 16L,
    latent_dim        = c(4L, 8L),
    lr                = c(0.01, 0.01),
    num_epochs        = 1L,
    batch_size        = 64L,
    refit_loops       = 1L,
    epochs_per_loop   = 1L,
    num_shared_encode = 0L,
    num_shared_decode = 1L,
    verbose           = FALSE
  )

  expect_true(all(c("imputed","model","study","results") %in% names(res)))
  expect_equal(nrow(res$results), 2L)
})
