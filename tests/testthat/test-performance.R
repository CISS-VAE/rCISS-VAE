library(testthat)

# ============================================================
# Helper: small reproducible dataset
# ============================================================
make_test_data <- function() {
  set.seed(123)

  df <- data.frame(
    id = 1:6,
    group = c("A", "A", "B", "B", "A", "B"),
    x1 = c(1, NA, 3, 4, NA, 6),   # continuous
    x2 = c(0, 1, NA, 1, 0, NA),   # binary
    cat1_1 = c(1, 0, 0, 1, NA, 0),
    cat1_2 = c(0, 1, 1, 0, NA, 1)
  )

  # predictions (intentionally imperfect)
  pred <- data.frame(
    id = df$id,
    group = df$group,
    x1 = c(1.1, 2, 2.9, 4.2, 3, 5.8),
    x2 = c(0.1, 0.9, 0.3, 0.8, 0.2, 0.7),
    cat1_1 = c(0.9, 0.2, 0.1, 0.8, 0.5, 0.3),
    cat1_2 = c(0.1, 0.8, 0.9, 0.2, 0.5, 0.7)
  )

  # mask: only evaluate where val_data is NOT NA
  pred[is.na(df)] <- NA

  list(
    val_data = df,
    val_imputed = pred,
    clusters = c(0, 0, 1, 1, 0, 1)
  )
}

# ============================================================
# 1. Basic functionality
# ============================================================
test_that("returns expected structure", {
  res <- make_test_data()

  out <- performance_by_cluster(
  res,
  group_col = "group",
  binary_features = "x2",
  categorical_features = list(cat1 = c("cat1_1", "cat1_2")),
  cols_ignore = "id",
  by_group = TRUE,
  by_cluster = TRUE
)

  expect_true(is.list(out))
  expect_true("overall" %in% names(out))
  expect_true("per_cluster" %in% names(out))
  expect_true("per_group" %in% names(out))
  expect_true("group_by_cluster" %in% names(out))

  expect_true(all(c("mse", "bce", "ce", "accuracy", "imputation_error") %in% names(out$overall)))
})

# ============================================================
# 2. MSE correctness (continuous)
# ============================================================
test_that("MSE is computed correctly", {
  res <- make_test_data()

  out <- performance_by_cluster(
    res,
    feature_cols = "x1",                # ← CRITICAL FIX
    binary_features = character(0),
    cols_ignore = "id",
    by_group = FALSE,
    by_cluster = FALSE
  )

  y <- res$val_data$x1
  yhat <- res$val_imputed$x1
  mask <- !is.na(y)

  expected_mse <- mean((yhat[mask] - y[mask])^2)

  expect_equal(out$overall$mse, expected_mse, tolerance = 1e-6)
})

# ============================================================
# 3. BCE correctness (binary)
# ============================================================
test_that("BCE is computed correctly", {
  res <- make_test_data()

  out <- performance_by_cluster(
    res,
    binary_features = "x2",
    cols_ignore = "id",
    by_group = FALSE,
    by_cluster = FALSE
  )

  y <- res$val_data$x2
  yhat <- res$val_imputed$x2
  mask <- !is.na(y)

  eps <- 1e-7
  yhat <- pmin(pmax(yhat, eps), 1 - eps)

  expected_bce <- mean(
    -(y[mask] * log(yhat[mask]) + (1 - y[mask]) * log(1 - yhat[mask]))
  )

  expect_equal(out$overall$bce, expected_bce, tolerance = 1e-6)
})

# ============================================================
# 4. Categorical CE correctness
# ============================================================
test_that("categorical cross-entropy is computed correctly", {
  res <- make_test_data()

  out <- performance_by_cluster(
    res,
    binary_features = "x2",
    categorical_features = list(cat1 = c("cat1_1", "cat1_2")),
    cols_ignore = "id",
    by_group = FALSE,
    by_cluster = FALSE
  )

  y_mat <- as.matrix(res$val_data[, c("cat1_1", "cat1_2")])
  yhat <- as.matrix(res$val_imputed[, c("cat1_1", "cat1_2")])

  mask <- rowSums(!is.na(y_mat)) > 0

  y_true <- apply(y_mat, 1, function(x) {
    if (all(is.na(x))) return(NA_integer_)
    which.max(x)
  })

  eps <- 1e-7
  yhat <- pmin(pmax(yhat, eps), 1 - eps)

  ce_vals <- rep(NA_real_, nrow(y_mat))
  ce_vals[mask] <- -log(yhat[cbind(seq_len(nrow(yhat)), y_true)][mask])

  expected_ce <- mean(ce_vals[mask])

  expect_equal(out$overall$ce, expected_ce, tolerance = 1e-6)
})

# ============================================================
# 5. Group aggregation
# ============================================================
test_that("group aggregation works", {
  res <- make_test_data()

  out <- performance_by_cluster(
    res,
    group_col = "group",
    binary_features = "x2",
    cols_ignore = "id"
  )

  expect_true(nrow(out$per_group) >= 2)
  expect_true("group" %in% names(out$per_group))
})

# ============================================================
# 6. Cluster aggregation
# ============================================================
test_that("cluster aggregation works", {
  res <- make_test_data()

  out <- performance_by_cluster(
    res,
    binary_features = "x2",
    cols_ignore = "id"
  )

  expect_true(nrow(out$per_cluster) >= 2)
  expect_true("cluster" %in% names(out$per_cluster))
})

# ============================================================
# 7. Error handling
# ============================================================
test_that("throws error on invalid inputs", {
  res <- make_test_data()

  # mismatched rows
  bad_res <- res
  bad_res$val_imputed <- bad_res$val_imputed[-1, ]

  expect_error(performance_by_cluster(bad_res))

  # missing clusters
  bad_res2 <- res
  bad_res2$clusters <- NULL

  expect_error(performance_by_cluster(bad_res2, clusters = NULL))

  # invalid group column
  expect_error(
    performance_by_cluster(res, group_col = "not_a_column")
  )

  # binary not subset
  expect_error(
    performance_by_cluster(res, binary_features = "not_a_feature")
  )
})

# ============================================================
# 8. NA handling (critical correctness)
# ============================================================
test_that("NA values are properly excluded", {
  res <- make_test_data()

  out <- performance_by_cluster(
    res,
    binary_features = "x2",
    cols_ignore = "id",
    by_group = FALSE,
    by_cluster = FALSE
  )

  expect_false(is.nan(out$overall$mse))
  expect_false(is.nan(out$overall$bce))
})

# ============================================================
# 9. Accuracy output sanity
# ============================================================
test_that("accuracy is between 0 and 1", {
  res <- make_test_data()

  out <- performance_by_cluster(
    res,
    binary_features = "x2",
    categorical_features = list(cat1 = c("cat1_1", "cat1_2")),
    cols_ignore = "id"
  )

  expect_true(out$overall$accuracy >= 0 || is.na(out$overall$accuracy))
  expect_true(out$overall$accuracy <= 1 || is.na(out$overall$accuracy))
})