# tests/testthat/test-core.R
# -------------------------------------------------------------------
# Tests aligned with CURRENT run_cissvae() implementation
# -------------------------------------------------------------------
testthat::skip_if_not(
  reticulate::py_module_available("ciss_vae"),
  message = "Skipping Python tests because 'ciss_vae' is not available"
)


skip_if_no_cissvae_py <- function() {
  if (!reticulate::py_module_available("ciss_vae")) {
    skip("Python module ciss_vae not available")
  }
}

# ---------- Fixtures ---------------------------------------------------------

make_sample_data <- function() {
  set.seed(42)
  Sigma <- diag(2) * 0.3
  cl1 <- MASS::mvrnorm(50, mu = c(0,0), Sigma)
  cl2 <- MASS::mvrnorm(50, mu = c(3,3), Sigma)
  noise <- matrix(rnorm(100 * 18, sd = 0.5), 100, 18)
  X <- cbind(rbind(cl1, cl2), noise)
  colnames(X) <- paste0("feature_", seq_len(ncol(X)) - 1)
  df <- as.data.frame(X)
  mask <- matrix(runif(length(df)) < 0.05, nrow(df), ncol(df))
  df[mask] <- NA
  df
}

make_longitudinal_data <- function() {
  set.seed(123)
  n <- 100; t <- 5; tp <- 1:t
  y1 <- 0.5 * tp
  y2 <- sin(tp/2)
  y3 <- log1p(tp)
  block <- function(mu) sapply(mu, \(m) rnorm(n, m, 0.3))
  df <- as.data.frame(cbind(block(y1), block(y2), block(y3)))
  colnames(df) <- c(paste0("y1_",1:t), paste0("y2_",1:t), paste0("y3_",1:t))
  mask <- matrix(runif(length(df)) < 0.05, nrow(df), ncol(df))
  df[mask] <- NA
  df
}

make_large_sample_data <- function() {
  set.seed(42)
  df <- as.data.frame(matrix(rnorm(1000*50), 1000, 50))
  colnames(df) <- paste0("feature_", seq_len(50)-1)
  mask <- matrix(runif(1000*50) < 0.05, 1000, 50)
  df[mask] <- NA
  df
}

minimal_params <- function() {
  list(
    hidden_dims      = c(32L,16L),
    latent_dim       = 8L,
    epochs           = 1L,
    batch_size       = 32L,
    max_loops        = 1L,
    patience         = 1L,
    epochs_per_loop  = 1L,
    verbose          = FALSE,
    n_clusters       = 2L,
    layer_order_enc  = c("unshared","shared"),
    layer_order_dec  = c("shared","unshared")
  )
}

is_py_obj <- function(x) inherits(x, "python.builtin.object")

# -------------------------------------------------------------------
# TESTS
# -------------------------------------------------------------------

test_that("default returns contain imputed_dataset, raw_data, model", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  params <- minimal_params()

  res <- do.call(run_cissvae, c(list(df), params))

  expect_type(res, "list")
  expect_named(res, c("imputed_dataset", "raw_data", "model", "clusters"))

  expect_s3_class(res$imputed_dataset, "data.frame")
  expect_equal(dim(res$imputed_dataset), dim(df))
  expect_true(is_py_obj(res$model))
  expect_equal(length(res$clusters), nrow(df))
})

test_that("single-return mode still includes raw_data (always returned)", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  params <- minimal_params()

  res <- do.call(run_cissvae, c(
    list(
      df,
      return_model = FALSE,
      return_dataset = FALSE,
      return_clusters = FALSE,
      return_silhouettes = FALSE,
      return_history = FALSE
    ),
    params
  ))

  # clusters still returned because function forces them when clusters missing
  expect_named(res, c("imputed_dataset","raw_data","clusters"))
  expect_equal(dim(res$imputed_dataset), dim(df))
})

test_that("optional returns append correctly after imputed_dataset/raw_data", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  params <- minimal_params()

  cases <- list(
    list(
      flags = list(return_model=TRUE),
      expected = c("imputed_dataset","raw_data","model","clusters")
    ),
    list(
      flags = list(return_dataset=TRUE),
      expected = c("imputed_dataset","raw_data","model", "cluster_dataset","clusters")
    ),
    list(
      flags = list(return_silhouettes=TRUE),
      expected = c("imputed_dataset","raw_data", "model", "clusters", "silhouette_width")
    ),
    list(
      flags = list(return_history=TRUE),
      expected = c("imputed_dataset","raw_data", "model", "clusters", "training_history")
    ),
    list(
      flags = list(return_model=TRUE, return_dataset=TRUE),
      expected = c("imputed_dataset","raw_data","model","cluster_dataset","clusters")
    )
  )

  for (cs in cases) {
    res <- do.call(run_cissvae, c(list(df), cs$flags, params))
    expect_named(res, cs$expected)
  }
})

test_that("cluster labels are valid and contiguous 0..k-1", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  params <- minimal_params()

  res <- do.call(run_cissvae, c(list(df), params))

  cl <- as.integer(res$clusters)
  expect_false(anyNA(cl))
  expect_true(min(cl) >= 0)
  expect_equal(sort(unique(cl)), 0:(length(unique(cl)) - 1))
})

test_that("model architecture reflects parameters", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()

  params <- minimal_params()
  params$hidden_dims <- c(64L, 32L)
  params$latent_dim <- 10L
  params$latent_shared <- TRUE
  params$output_shared <- FALSE

  res <- do.call(run_cissvae, c(list(df, return_model = TRUE), params))

  vae <- res$model
  expect_equal(reticulate::py_to_r(vae$hidden_dims), c(64L,32L))
  expect_equal(reticulate::py_to_r(vae$latent_dim), 10L)
  expect_true(reticulate::py_to_r(vae$latent_shared))
  expect_false(reticulate::py_to_r(vae$output_shared))
})

test_that("proportion-matrix clustering path works", {
  skip_if_no_cissvae_py()
  df <- make_longitudinal_data()

  util <- reticulate::import("ciss_vae.utils.matrix", convert = FALSE)
  pm <- util$create_missingness_prop_matrix(
    df,
    repeat_feature_names = reticulate::r_to_py(c("y1","y2","y3"))
  )
  pm_r <- reticulate::py_to_r(pm$data)

  expect_equal(ncol(pm_r), 3)
  expect_equal(nrow(pm_r), nrow(df))

  params <- minimal_params()

  res <- do.call(run_cissvae, c(list(
    df,
    missingness_proportion_matrix = pm,
    return_clusters = TRUE
  ), params))

  expect_true(all(c("imputed_dataset","raw_data","clusters") %in% names(res)))
  expect_equal(length(res$clusters), nrow(df))
})

test_that("training history returns when requested", {
  skip_if_no_cissvae_py()
  df <- make_sample_data()
  params <- minimal_params()
  params$epochs <- 1L

  res <- do.call(run_cissvae, c(list(df, return_history = TRUE), params))

  expect_true("training_history" %in% names(res))
  expect_true(is.null(res$training_history) || inherits(res$training_history, "data.frame"))
})

test_that("integration test (slow) [skip on CRAN]", {
  skip_on_cran()
  skip_if_not(as.logical(Sys.getenv("RCISSVAE_RUN_SLOW","FALSE")),
              "Set RCISSVAE_RUN_SLOW=TRUE to run slow test")

  df <- make_large_sample_data()

  res <- run_cissvae(
    df,
    hidden_dims = c(100L,50L,25L),
    latent_dim = 15L,
    epochs = 2L,
    max_loops = 1L,
    epochs_per_loop = 1L,
    batch_size = 128L,
    return_model = TRUE,
    return_clusters = TRUE,
    return_dataset = TRUE,
    return_silhouettes = TRUE,
    return_history = TRUE,
    verbose = FALSE
  )

  expect_named(res, c(
    "imputed_dataset","raw_data","model","cluster_dataset",
    "silhouette_width","training_history","clusters"
  ))

  expect_equal(dim(res$imputed_dataset), dim(df))
  expect_false(anyNA(res$imputed_dataset))
})
