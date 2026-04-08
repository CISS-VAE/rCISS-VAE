# tests/testthat/test-run_cissvae.R
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
  mask <- matrix(runif(nrow(df) * ncol(df)) < 0.05,
               nrow = nrow(df),
               ncol = ncol(df))

  df[as.matrix(mask)] <- NA
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

## add fixture for categorical data


make_categorical_data <- function() {
  set.seed(42)

  df <- data.frame(
    x = rnorm(50),
    y = rnorm(50),

    # categorical dummy group
    C1_b1 = sample(c(0,1), 50, replace = TRUE),
    C1_b2 = sample(c(0,1), 50, replace = TRUE),

    # second categorical
    C2_b1 = sample(c(0,1), 50, replace = TRUE),
    C2_b2 = sample(c(0,1), 50, replace = TRUE)
  )

  # introduce missingness
  mask <- matrix(runif(nrow(df) * ncol(df)) < 0.05,
               nrow = nrow(df),
               ncol = ncol(df))

df[as.matrix(mask)] <- NA

  list(
    df = df,
    clusters = sample(0:1, 50, replace = TRUE),
    categorical_column_map = list(
      C1 = c("C1_b1", "C1_b2"),
      C2 = c("C2_b1", "C2_b2")
    ),
    binary_feature_mask = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)
  )
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
  expect_named(res, c('imputed_dataset', 'raw_data', 'model', 'cluster_dataset', 'clusters', 'val_data', 'val_imputed'))

  expect_s3_class(res$imputed_dataset, "data.frame")
  expect_equal(dim(res$imputed_dataset), dim(df))
  expect_true(is_py_obj(res$model))
  expect_equal(length(res$clusters), nrow(df))
})

test_that("binary_feature_matrix works", {
  skip_if_no_cissvae_py()
  df = data.frame(index = c(1, 2, 3, 4, 5), A = c(5, 7, 5, NA, 1), B = c(NA, 1, 0, 1, NA))
  binary_feature_mask = c(FALSE, FALSE, TRUE)
  names(binary_feature_mask) = colnames(df)
  res  = run_cissvae(
    data = df,
    index_col = "index",
    binary_feature_mask = binary_feature_mask,
    clusters = c(0, 0, 0, 1, 1),
    seed = 42, 
    hidden_dims = c(3, 2),
    latent_dim = 3,
    layer_order_enc = c("unshared", "shared"),
    layer_order_dec = c("unshared", "shared"),
    epochs = 2,
    return_dataset = TRUE
  )
  mask_out <- reticulate::py_to_r(res$cluster_dataset$binary_feature_mask)
  expect_true(is.logical(mask_out))
  expect_equal(ncol(df), ncol(res$imputed_dataset))
  expect_equal(length(binary_feature_mask), ncol(res$imputed_dataset))
  expect_true(is.data.frame(res$imputed_dataset))
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
      return_history = FALSE,
      return_validation_dataset = FALSE
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
      flags = list(return_model=TRUE, return_validation_dataset = FALSE),
      expected = c("imputed_dataset","raw_data","model","clusters")
    ),
    list(
      flags = list(return_dataset=TRUE, return_validation_dataset = FALSE),
      expected = c("imputed_dataset","raw_data","model", "cluster_dataset","clusters")
    ),
    list(
      flags = list(return_silhouettes=TRUE, return_validation_dataset = FALSE),
      expected = c("imputed_dataset","raw_data", "model", "clusters", "silhouette_width")
    ),
    list(
      flags = list(return_history=TRUE, return_validation_dataset = FALSE),
      expected = c("imputed_dataset","raw_data", "model", "clusters", "training_history")
    ),
    list(
      flags = list(return_model=TRUE, return_dataset=TRUE, return_validation_dataset = FALSE),
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

## =====================
## Add tests for the categorical data to make sure that it is working smoothly
## =====================

test_that("categorical_column_map correctly builds activation_groups", {
  skip_if_no_cissvae_py()

  td <- make_categorical_data()

  res <- run_cissvae(
    data = td$df,
    clusters = td$clusters,
    categorical_column_map = td$categorical_column_map,
    binary_feature_mask = td$binary_feature_mask,
    return_model = TRUE,
    verbose = FALSE,
    epochs = 1L
  )

  ag <- reticulate::py_to_r(res$model$activation_groups)

  expect_true("C1" %in% names(ag))
  expect_true("C2" %in% names(ag))

  expect_length(ag$C1, 2)
  expect_length(ag$C2, 2)
})

test_that("categorical activation_groups propagate to model", {
  skip_if_no_cissvae_py()

  td <- make_categorical_data()

  res <- run_cissvae(
    data = td$df,
    clusters = td$clusters,
    categorical_column_map = td$categorical_column_map,
    binary_feature_mask = td$binary_feature_mask,
    return_model = TRUE,
    verbose = FALSE,
    epochs = 1L
  )

  ag <- reticulate::py_to_r(res$model$activation_groups)

  expect_true(all(names(td$categorical_column_map) %in% names(ag)))
})

test_that("categorical_column_map requires binary_feature_mask", {
  skip_if_no_cissvae_py()

  td <- make_categorical_data()

  expect_error(
    run_cissvae(
      data = td$df,
      clusters = td$clusters,
      categorical_column_map = td$categorical_column_map,
      binary_feature_mask = NULL,
      verbose = FALSE,
      epochs = 1L
    ),
    "binary_feature_mask"
  )
})

test_that("categorical_column_map rejects invalid columns", {
  skip_if_no_cissvae_py()

  td <- make_categorical_data()

  bad_map <- list(C1 = c("NOT_A_COLUMN"))

  expect_error(
    run_cissvae(
      data = td$df,
      clusters = td$clusters,
      categorical_column_map = bad_map,
      binary_feature_mask = td$binary_feature_mask,
      verbose = FALSE,
      epochs = 1L
    ),
    "not in data"
  )
})

test_that("single-layer model works (length-1 hidden_dims + layer_order)", {
  skip_if_no_python()

  # Minimal synthetic dataset
  df <- data.frame(
    x1 = c(1, NA, 3, 4),
    x2 = c(NA, 2, 3, 4)
  )

  clusters <- c(0L, 0L, 1L, 1L)

  # This is the critical part: length-1 inputs
  res <- run_cissvae(
    data = df,
    clusters = clusters,
    hidden_dims = c(8),                 # length 1
    layer_order_enc = c("shared"),      # length 1
    layer_order_dec = c("shared"),      # length 1
    latent_dim = 2,
    epochs = 1,
    max_loops = 1,
    epochs_per_loop = 1,
    return_model = TRUE,
    verbose = FALSE
  )

  # Basic sanity checks
  expect_true(!is.null(res$imputed_dataset))
  expect_true(nrow(res$imputed_dataset) == nrow(df))
  expect_true(ncol(res$imputed_dataset) == ncol(df))

  # Model should exist and be valid Python object
  expect_true(!is.null(res$model))
  expect_true(reticulate::py_has_attr(res$model, "forward"))
})

test_that("single-layer inputs are passed as lists to Python", {
  skip_if_no_python()

  py <- reticulate::import_builtins()

  hidden_dims <- c(8)
  py_hidden <- reticulate::r_to_py(as.list(hidden_dims))

 
  expect_true(py$isinstance(py_hidden, py$list))
})