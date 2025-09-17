# tests/testthat/test-core.R
library(testthat)
library(rCISSVAE)
library(reticulate)

# ---------- helpers (fixtures) ----------
make_sample_data <- function() {
  set.seed(42)
  # 2 clusters in 2 informative dims + 18 noise dims
  cluster1 <- MASS::mvrnorm(50, mu = c(0, 0), Sigma = diag(2) * 0.3)
  cluster2 <- MASS::mvrnorm(50, mu = c(3, 3), Sigma = diag(2) * 0.3)
  noise    <- matrix(rnorm(100 * 18, sd = 0.5), nrow = 100, ncol = 18)
  X        <- cbind(rbind(cluster1, cluster2), noise)
  colnames(X) <- paste0("feature_", seq_len(ncol(X)) - 1)
  df <- as.data.frame(X)

  # ~5% missing
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
  data <- lapply(list(y1 = y1, y2 = y2, y3 = y3), function(traj) {
    out <- sapply(traj, function(m) rnorm(n, m, 0.3))
    out
  })
  # bind y1_1..y1_5, y2_1.., y3_1..
  mat <- do.call(cbind, data)
  colnames(mat) <- c(paste0("y1_", 1:t), paste0("y2_", 1:t), paste0("y3_", 1:t))
  df <- as.data.frame(mat)
  mask <- matrix(runif(length(df)) < 0.05, nrow(df), ncol(df))
  df[mask] <- NA
  df
}

make_large_sample_data <- function() {
  set.seed(42)
  m <- matrix(rnorm(1000 * 50), 1000, 50)
  df <- as.data.frame(m)
  colnames(df) <- paste0("feature_", seq_len(50) - 1)
  mask <- matrix(runif(1000 * 50) < 0.05, 1000, 50)
  df[mask] <- NA
  df
}

minimal_params <- function() {
  list(
    hidden_dims      = c(32L, 16L),
    latent_dim       = 8L,
    epochs           = 2L,
    batch_size       = 32L,
    max_loops        = 2L,
    patience         = 1L,
    epochs_per_loop  = 1L,
    verbose          = FALSE,
    n_clusters       = 2L,
    layer_order_enc  = c("unshared", "shared"),
    layer_order_dec  = c("shared", "unshared")
  )
}

is_py_obj <- function(x) inherits(x, "python.builtin.object")

# ---------- tests ----------
test_that("default returns: imputed + model", {
  sample_data <- make_sample_data()
  params <- minimal_params()

  res <- do.call(rCISSVAE::run_cissvae, c(list(sample_data), params))
  # R wrapper always returns a list
  expect_type(res, "list")
  expect_named(res, c("imputed", "model"))

  expect_s3_class(res$imputed, "data.frame")
  expect_equal(dim(res$imputed), dim(sample_data))
  expect_true(is_py_obj(res$model))
})

test_that("single return: imputed only when all flags FALSE", {
  sample_data <- make_sample_data()
  params <- minimal_params()

  res <- do.call(rCISSVAE::run_cissvae, c(list(
    sample_data,
    return_model = FALSE,
    return_clusters = FALSE,
    return_silhouettes = FALSE,
    return_history = FALSE,
    return_dataset = FALSE
  ), params))
  
  expect_type(res, "list")
  expect_named(res, "imputed")
  expect_s3_class(res$imputed, "data.frame")
  expect_equal(dim(res$imputed), dim(sample_data))
})

test_that("return combinations: names, order, and types", {
  sample_data <- make_sample_data()
  params <- minimal_params()

  # Each case: flags + expected name order + type-checks
  cases <- list(
    list(
      flags = list(return_model=TRUE, return_clusters=FALSE, return_silhouettes=FALSE, return_history=FALSE, return_dataset=FALSE),
      names = c("imputed", "model"),
      check = function(res) { expect_true(is_py_obj(res$model)) }
    ),
    list(
      flags = list(return_model=FALSE, return_clusters=TRUE, return_silhouettes=FALSE, return_history=FALSE, return_dataset=FALSE),
      names = c("imputed", "clusters"),
      check = function(res) { expect_true(is.integer(res$clusters) || is.numeric(res$clusters)) }
    ),
    list(
      flags = list(return_model=FALSE, return_clusters=FALSE, return_silhouettes=TRUE, return_history=FALSE, return_dataset=FALSE),
      names = c("imputed", "silhouettes"),
      check = function(res) { expect_true(is.null(res$silhouettes) || is.numeric(res$silhouettes)) }
    ),
    list(
      flags = list(return_model=FALSE, return_clusters=FALSE, return_silhouettes=FALSE, return_history=TRUE, return_dataset=FALSE),
      names = c("imputed", "history"),
      check = function(res) { expect_true(is.null(res$history) || inherits(res$history, "data.frame")) }
    ),
    list(
      flags = list(return_model=FALSE, return_clusters=FALSE, return_silhouettes=FALSE, return_history=FALSE, return_dataset=TRUE),
      names = c("imputed", "dataset"),
      check = function(res) { expect_true(is_py_obj(res$dataset)) }
    ),
    # Multiple
    list(
      flags = list(return_model=TRUE, return_clusters=TRUE, return_silhouettes=FALSE, return_history=FALSE, return_dataset=FALSE),
      names = c("imputed", "model", "clusters"),
      check = function(res) { expect_true(is_py_obj(res$model)); expect_true(is.numeric(res$clusters) || is.integer(res$clusters)) }
    ),
    list(
      flags = list(return_model=TRUE, return_clusters=FALSE, return_silhouettes=TRUE, return_history=FALSE, return_dataset=FALSE),
      names = c("imputed", "model", "silhouettes"),
      check = function(res) { expect_true(is_py_obj(res$model)); expect_true(is.null(res$silhouettes) || is.numeric(res$silhouettes)) }
    ),
    list(
      flags = list(return_model=FALSE, return_clusters=TRUE, return_silhouettes=TRUE, return_history=FALSE, return_dataset=FALSE),
      names = c("imputed", "clusters", "silhouettes"),
      check = function(res) { expect_true(is.numeric(res$clusters) || is.integer(res$clusters)) }
    ),
    list(
      flags = list(return_model=TRUE, return_dataset=TRUE, return_clusters=FALSE, return_silhouettes=FALSE, return_history=FALSE),
      names = c("imputed", "model", "dataset"),
      check = function(res) { expect_true(is_py_obj(res$model)); expect_true(is_py_obj(res$dataset)) }
    ),
    # All
    list(
      flags = list(return_model=TRUE, return_dataset=TRUE, return_clusters=TRUE, return_silhouettes=TRUE, return_history=TRUE),
      names = c("imputed", "model", "dataset", "clusters", "silhouettes", "history"),
      check = function(res) {
        expect_true(is_py_obj(res$model))
        expect_true(is_py_obj(res$dataset))
        expect_true(is.numeric(res$clusters) || is.integer(res$clusters))
        expect_true(is.null(res$silhouettes) || is.numeric(res$silhouettes))
        expect_true(is.null(res$history) || inherits(res$history, "data.frame"))
      }
    )
  )

  for (cs in cases) {
    res <- do.call(rCISSVAE::run_cissvae, c(list(sample_data), cs$flags, params))
    expect_named(res, cs$names)
    expect_s3_class(res$imputed, "data.frame")
    cs$check(res)
  }
})

test_that("return order consistency across different flag sets", {
  sample_data <- make_sample_data()
  params <- minimal_params()

  res1 <- do.call(rCISSVAE::run_cissvae, c(list(
    sample_data,
    return_model=TRUE, return_dataset=TRUE, return_clusters=TRUE,
    return_silhouettes=FALSE, return_history=FALSE), params
  ))
  expect_named(res1, c("imputed", "model", "dataset", "clusters"))
  expect_s3_class(res1$imputed, "data.frame")
  expect_true(is_py_obj(res1$model))
  expect_true(is_py_obj(res1$dataset))
  expect_true(is.numeric(res1$clusters) || is.integer(res1$clusters))

  res2 <- do.call(rCISSVAE::run_cissvae, c(list(
    sample_data,
    return_model = TRUE, return_dataset = FALSE, return_clusters = TRUE,
    return_silhouettes = TRUE, return_history = TRUE
  ), params))

  expect_named(res2, c("imputed", "model", "clusters", "silhouettes", "history"))
  expect_true(is_py_obj(res2$model))
  expect_true(is.numeric(res2$clusters) || is.integer(res2$clusters))
  expect_true(is.null(res2$silhouettes) || is.numeric(res2$silhouettes))
  expect_true(is.null(res2$history) || inherits(res2$history, "data.frame"))
})

test_that("data integrity and cluster labeling", {
  sample_data <- make_sample_data()
  params <- minimal_params()

  res <- do.call(rCISSVAE::run_cissvae, c(list(
    sample_data,
    return_model = TRUE, return_clusters = TRUE, return_dataset = TRUE
  ), params))

  expect_named(res, c("imputed", "model", "dataset", "clusters"))
  # Shapes
  expect_equal(dim(res$imputed), dim(sample_data))
  expect_equal(length(res$clusters), nrow(sample_data))
  # No NAs
  expect_false(anyNA(res$imputed))

  # clusters start at 0 and are contiguous
  uq <- sort(unique(as.integer(res$clusters)))
  expect_true(all(uq >= 0))
  expect_equal(uq, 0:(length(uq) - 1))

  # model sanity
  expect_true(is_py_obj(res$model))
})

test_that("model architecture parameters are respected", {
  sample_data <- make_sample_data()
  params <- minimal_params()
  params$hidden_dims <- c(64L, 32L)
  params$latent_dim <- 10L
  params$layer_order_enc <- c("unshared", "shared")
  params$layer_order_dec <- c("shared", "unshared")
  params$latent_shared <- TRUE
  params$output_shared <- FALSE

  res <- do.call(rCISSVAE::run_cissvae, c(list(
    sample_data, return_model = TRUE, return_clusters = FALSE
  ), params))

  expect_named(res, c("imputed", "model"))
  vae <- res$model
  expect_true(is_py_obj(vae))
  # Access Python attributes
  expect_equal(py_to_r(vae$hidden_dims), c(64L, 32L))
  expect_equal(py_to_r(vae$latent_dim), 10L)
  expect_true(py_to_r(vae$latent_shared))
  expect_false(py_to_r(vae$output_shared))
})

test_that("clustering parameters work (fixed n_clusters and Leiden)", {
  sample_data <- make_sample_data()
  params <- minimal_params()

  # with n_clusters = 2
  res1 <- do.call(rCISSVAE::run_cissvae, c(list(
  sample_data,
  return_clusters = TRUE, return_silhouettes = TRUE, return_model = FALSE
), params))

  expect_named(res1, c("imputed", "clusters", "silhouettes"))
  uq <- sort(unique(as.integer(res1$clusters)))
  expect_equal(length(uq), 2)

  # Leiden path: no n_clusters
  params2 <- params
  params2$n_clusters <- NULL
  params2$leiden_resolution <- 0.1

  res2 <- do.call(rCISSVAE::run_cissvae, c(list(
  sample_data, return_clusters = TRUE, return_model = FALSE
), params2))
  
  expect_named(res2, c("imputed", "clusters"))
  uq2 <- sort(unique(as.integer(res2$clusters)))
  expect_true(length(uq2) >= 1)
  expect_true(length(uq2) <= nrow(sample_data) %/% 2)
})

test_that("prop-matrix clustering path works", {
  longitudinal_data <- make_longitudinal_data()
  params <- minimal_params()

  # call Python util directly
  umatrix <- import("ciss_vae.utils.matrix", convert = FALSE)
  pm <- umatrix$create_missingness_prop_matrix(longitudinal_data, repeat_feature_names = r_to_py(c("y1","y2","y3")))
  pm_r <- py_to_r(pm$data)

  expect_equal(ncol(pm_r), 3)
  expect_equal(nrow(pm_r), nrow(longitudinal_data))

  res <- do.call(rCISSVAE::run_cissvae, c(list(
    longitudinal_data,
    return_clusters = TRUE, return_silhouettes = TRUE, return_model = FALSE,
    missingness_proportion_matrix = pm
  ), params))
  
  expect_named(res, c("imputed", "clusters", "silhouettes"))
  uq <- sort(unique(as.integer(res$clusters)))
  expect_equal(length(uq), 2)
  expect_equal(length(res$clusters), nrow(longitudinal_data))
})

test_that("training parameters don't break the pipeline and history may exist", {
  sample_data <- make_sample_data()
  params <- minimal_params()
  params$epochs <- 1L
  params$max_loops <- 1L
  params$epochs_per_loop <- 1L
  params$initial_lr <- 0.1
  params$decay_factor <- 0.9
  params$beta <- 0.01

  
  res <- do.call(rCISSVAE::run_cissvae, c(list(
    sample_data, return_model = TRUE, return_history = TRUE
  ), params))

  expect_true(all(c("imputed", "model", "history") %in% names(res)))
  expect_s3_class(res$imputed, "data.frame")
  expect_true(is_py_obj(res$model))
  expect_true(is.null(res$history) || inherits(res$history, "data.frame"))
})

test_that("full pipeline integration (slow) [skip on CRAN]", {
  skip_on_cran()
  skip_if_not(as.logical(Sys.getenv("RCISSVAE_RUN_SLOW", "FALSE")), "Set RCISSVAE_RUN_SLOW=TRUE to run slow test")

  large_data <- make_large_sample_data()
  res <- rCISSVAE::run_cissvae(
    large_data,
    hidden_dims=c(100L, 50L, 25L),
    latent_dim=15L,
    epochs=5L,
    max_loops=3L,
    epochs_per_loop=2L,
    batch_size=128L,
    return_model=TRUE,
    return_clusters=TRUE,
    return_silhouettes=TRUE,
    return_history=TRUE,
    return_dataset=TRUE,
    verbose=FALSE
  )

  expect_named(res, c("imputed","model","dataset","clusters","silhouettes","history"))
  expect_s3_class(res$imputed, "data.frame")
  expect_true(is_py_obj(res$model))
  expect_true(is_py_obj(res$dataset))
  expect_true(is.numeric(res$silhouettes) || is.null(res$silhouettes))
  expect_true(is.null(res$history) || inherits(res$history, "data.frame"))
  expect_equal(dim(res$imputed), dim(large_data))
  expect_false(anyNA(res$imputed))
})
