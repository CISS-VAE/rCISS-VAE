# tests/testthat/test-ignored_cols.R
# -------------------------------------------------------------------
# Tests for ignored columns behavior
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

get_last_non_na <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  as.numeric(tail(x, 1))
}

# ---------------------------------------------------------
# Helper: robust missingness injection
# ---------------------------------------------------------
add_missing <- function(df, n = 10) {
  mask <- matrix(FALSE, nrow(df), ncol(df))
  total_cells <- length(mask)

  idx <- sample(total_cells, min(n, total_cells), replace = FALSE)
  mask[idx] <- TRUE

  df[mask] <- NA
  df
}
test_that("training loss includes ignored columns", {
  skip_if_no_cissvae_py()

  set.seed(123)

  df <- data.frame(
    ignored = rnorm(50, mean = 0, sd = 1000),
    active  = rnorm(50, mean = 0, sd = 1)
  )

  df <- add_missing(df, 10)

  clusters <- sample(0:1, 50, replace = TRUE)

  res_ignore <- run_cissvae(
    data = df,
    clusters = clusters,
    columns_ignore = c("ignored"),
    epochs = 2,
    max_loops = 1,
    epochs_per_loop = 1,
    return_history = TRUE,
    verbose = FALSE
  )

  df_reduced <- df[, "active", drop = FALSE]

  res_reduced <- run_cissvae(
    data = df_reduced,
    clusters = clusters,
    columns_ignore = NULL,
    epochs = 2,
    max_loops = 1,
    epochs_per_loop = 1,
    return_history = TRUE,
    verbose = FALSE
  )

  hist_ignore  <- res_ignore$training_history
  hist_reduced <- res_reduced$training_history

  # --------------------------------------------------
  # DEBUG: Inspect structure
  # --------------------------------------------------
  # cat("\n===== IGNORE HISTORY STRUCTURE =====\n")
  # print(str(hist_ignore))
  # print(hist_ignore)

  # cat("\n===== REDUCED HISTORY STRUCTURE =====\n")
  # print(str(hist_reduced))
  # print(hist_reduced)

  # --------------------------------------------------
  # DEBUG: What columns exist?
  # --------------------------------------------------
  # cat("\n===== IGNORE NAMES =====\n")
  # print(names(hist_ignore))

  # cat("\n===== REDUCED NAMES =====\n")
  # print(names(hist_reduced))

  # --------------------------------------------------
  # DEBUG: train_loss specifically
  # --------------------------------------------------
  # cat("\n===== IGNORE train_loss =====\n")
  # print(hist_ignore$train_loss)

  # cat("\n===== REDUCED train_loss =====\n")
  # print(hist_reduced$train_loss)

  final_loss_ignore  <- get_last_non_na(hist_ignore$train_loss)
  final_loss_reduced <- get_last_non_na(hist_reduced$train_loss)

  #   cat("\n===== FINAL LOSS VALUES =====\n")
  # print(final_loss_ignore)
  # print(final_loss_reduced)

  expect_true(!is.na(final_loss_ignore))
  expect_true(!is.na(final_loss_reduced))

  expect_true(is.finite(final_loss_ignore))
  expect_true(is.finite(final_loss_reduced))

  expect_true(final_loss_ignore > final_loss_reduced)
})

test_that("imputation error excludes ignored columns", {
  skip_if_no_cissvae_py()

  set.seed(123)

  df <- data.frame(
    ignored = c(10, 20, 30, 40),
    active  = c(1, 2, 3, 4)
  )

  df <- add_missing(df, 2)

  clusters <- c(0, 0, 1, 1)

  res <- run_cissvae(
    data = df,
    clusters = clusters,
    columns_ignore = c("ignored"),
    epochs = 1,
    max_loops = 1,
    epochs_per_loop = 1,
    return_validation_dataset = TRUE,
    return_dataset = TRUE,
    verbose = FALSE
  )

  val_true <- res$val_data
  val_pred <- res$val_imputed

  mask <- !is.na(val_true$active)

  mse_before <- mean((val_pred$active[mask] - val_true$active[mask])^2)

  # Corrupt ignored column only
  val_pred$ignored <- 9999

  mse_after <- mean((val_pred$active[mask] - val_true$active[mask])^2)

  expect_equal(mse_before, mse_after)
})
test_that("imputation error excludes ignored columns (compare ignored vs full)", {
  skip_if_no_cissvae_py()

  set.seed(123)

  df <- data.frame(
    ignored = c(10, 20, 30, 40, 50, 60),
    active  = c(1, 2, 3, 4, 5, 6)
  )

  df <- add_missing(df, 4)

  clusters <- c(0, 0, 1, 1, 0, 1)

  # ---------------------------------------
  # Run WITH ignored column
  # ---------------------------------------
  res_ignore <- run_cissvae(
    data = df,
    clusters = clusters,
    columns_ignore = c("ignored"),
    epochs = 1,
    max_loops = 1,
    epochs_per_loop = 1,
    return_history = TRUE,
    verbose = FALSE
  )

  # ---------------------------------------
  # Run WITHOUT ignoring
  # ---------------------------------------
  res_full <- run_cissvae(
    data = df,
    clusters = clusters,
    columns_ignore = NULL,
    epochs = 1,
    max_loops = 1,
    epochs_per_loop = 1,
    return_history = TRUE,
    verbose = FALSE
  )

  hist_ignore <- res_ignore$training_history
  hist_full   <- res_full$training_history

  # ---------------------------------------
  # Extract NON-NA imputation errors
  # ---------------------------------------
  imp_err_ignore <- get_last_non_na(hist_ignore$imputation_error)
  imp_err_full   <- get_last_non_na(hist_full$imputation_error)

  # ---------------------------------------
  # Debug prints
  # ---------------------------------------
  # cat("\n===== IMPUTATION ERROR DEBUG =====\n")
  # cat("ignore history:\n")
  # print(hist_ignore$imputation_error)

  # cat("full history:\n")
  # print(hist_full$imputation_error)

  # cat(glue::glue(
  #   "\nfinal ignore: {round(imp_err_ignore, 6)} | final full: {round(imp_err_full, 6)}\n"
  # ))

  # ---------------------------------------
  # Assertions
  # ---------------------------------------
  expect_true(!is.na(imp_err_ignore))
  expect_true(!is.na(imp_err_full))

  expect_true(is.finite(imp_err_ignore))
  expect_true(is.finite(imp_err_full))

  # ---------------------------------------
  # Key test:
  # Imputation error should NOT take into account the ignored column 
  # ---------------------------------------
  expect_false(abs(imp_err_ignore - imp_err_full) < 1e-6)
})

test_that("ignored columns all types behavior (R)", {
  skip_if_no_cissvae_py()

  set.seed(123)

  n <- 80

  df <- data.frame(
    ignored_cont = rnorm(n, 0, 1000),
    active_cont  = rnorm(n, 0, 1),

    ignored_bin = rbinom(n, 1, 0.1),
    active_bin  = rbinom(n, 1, 0.5),

    cat_a = rep(c(1,0,0), length.out = n),
    cat_b = rep(c(0,1,0), length.out = n),
    cat_c = rep(c(0,0,1), length.out = n)
  )

  df <- add_missing(df, 40)

  clusters <- sample(0:1, n, replace = TRUE)

  categorical_map <- list(cat = c("cat_a", "cat_b", "cat_c"))

  binary_mask <- c(
    FALSE, FALSE,
    TRUE, TRUE,
    TRUE, TRUE, TRUE
  )

  # ----------------------------
  # RUN WITH ignored
  # ----------------------------
  res_ignore <- run_cissvae(
    data = df,
    clusters = clusters,
    columns_ignore = c("ignored_cont", "ignored_bin"),
    binary_feature_mask = binary_mask,
    categorical_column_map = categorical_map,
    epochs = 2,
    max_loops = 1,
    epochs_per_loop = 1,
    return_history = TRUE,
    verbose = FALSE
  )

  # ----------------------------
  # RUN WITHOUT ignored
  # ----------------------------
  res_full <- run_cissvae(
    data = df,
    clusters = clusters,
    columns_ignore = NULL,
    binary_feature_mask = binary_mask,
    categorical_column_map = categorical_map,
    epochs = 2,
    max_loops = 1,
    epochs_per_loop = 1,
    return_history = TRUE,
    verbose = FALSE
  )

  hist_ignore <- res_ignore$training_history
  hist_full   <- res_full$training_history

  get_last_non_na <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(NA)
    tail(x, 1)[[1]]
  }

  train_ignore <- get_last_non_na(hist_ignore$train_loss)
  train_full   <- get_last_non_na(hist_full$train_loss)

  train_mse_ignore <- get_last_non_na(hist_ignore$train_mse)
  train_mse_full   <- get_last_non_na(hist_full$train_mse)

  train_bce_ignore <- get_last_non_na(hist_ignore$train_bce)
  train_bce_full   <- get_last_non_na(hist_full$train_bce)

  cat("\n===== TRAIN DEBUG =====\n")
  cat("train_ignore:", train_ignore, "\n")
  cat("train_full:", train_full, "\n")
  cat("mse_ignore:", train_mse_ignore, "\n")
  cat("mse_full:", train_mse_full, "\n")
  cat("bce_ignore:", train_bce_ignore, "\n")
  cat("bce_full:", train_bce_full, "\n")

  expect_true(is.finite(train_ignore))
  expect_true(is.finite(train_full))

  # ✔ training should NOT differ
  expect_lt(abs(train_ignore - train_full), 0.5)
  expect_lt(abs(train_mse_ignore - train_mse_full), 10)
  expect_lt(abs(train_bce_ignore - train_bce_full), 10)

  # ----------------------------
  # VALIDATION / IMPUTATION
  # ----------------------------
  imp_ignore <- get_last_non_na(hist_ignore$imputation_error)
  imp_full   <- get_last_non_na(hist_full$imputation_error)

  mse_ignore <- get_last_non_na(hist_ignore$val_mse)
  mse_full   <- get_last_non_na(hist_full$val_mse)

  bce_ignore <- get_last_non_na(hist_ignore$val_bce)
  bce_full   <- get_last_non_na(hist_full$val_bce)

  cat("\n===== VAL DEBUG =====\n")
  cat("imp_ignore:", imp_ignore, "\n")
  cat("imp_full:", imp_full, "\n")
  cat("mse_ignore:", mse_ignore, "\n")
  cat("mse_full:", mse_full, "\n")
  cat("bce_ignore:", bce_ignore, "\n")
  cat("bce_full:", bce_full, "\n")

  expect_true(is.finite(imp_ignore))
  expect_true(is.finite(imp_full))

  # ✔ validation SHOULD differ
  expect_gt(abs(imp_ignore - imp_full), 10)
  expect_gt(abs(mse_ignore - mse_full), 10)
  expect_gt(abs(bce_ignore - bce_full), 0.01)
})

test_that("ignored categorical group behavior (R)", {
  skip_if_no_cissvae_py()

  set.seed(123)

  n <- 90

  df <- data.frame(
    ignored_cont = rnorm(n, 0, 100),
    active_cont  = rnorm(n, 0, 1),

    ignored_bin = rbinom(n, 1, 0.5),
    active_bin  = rbinom(n, 1, 0.5),

    # ignored categorical
    cat_ign_a = rep(c(1,0,0), length.out = n),
    cat_ign_b = rep(c(0,1,0), length.out = n),
    cat_ign_c = rep(c(0,0,1), length.out = n),

    # active categorical
    cat_act_a = rep(c(1,0), length.out = n),
    cat_act_b = rep(c(0,1), length.out = n)
  )

  df <- add_missing(df, 60)

  clusters <- sample(0:1, n, replace = TRUE)

  categorical_map <- list(
    ignored_cat = c("cat_ign_a", "cat_ign_b", "cat_ign_c"),
    active_cat  = c("cat_act_a", "cat_act_b")
  )

  binary_mask <- c(
    FALSE, FALSE,
    TRUE, TRUE,
    TRUE, TRUE, TRUE,
    TRUE, TRUE
  )

  ignored_cols <- c(
    "ignored_cont", "ignored_bin",
    "cat_ign_a", "cat_ign_b", "cat_ign_c"
  )

  res_ignore <- run_cissvae(
    data = df,
    clusters = clusters,
    columns_ignore = ignored_cols,
    binary_feature_mask = binary_mask,
    categorical_column_map = categorical_map,
    epochs = 2,
    max_loops = 1,
    epochs_per_loop = 1,
    return_history = TRUE,
    verbose = FALSE
  )

  res_full <- run_cissvae(
    data = df,
    clusters = clusters,
    columns_ignore = NULL,
    binary_feature_mask = binary_mask,
    categorical_column_map = categorical_map,
    epochs = 2,
    max_loops = 1,
    epochs_per_loop = 1,
    return_history = TRUE,
    verbose = FALSE
  )

  hist_ignore <- res_ignore$training_history
  hist_full   <- res_full$training_history

  get_last_non_na <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(NA)
    tail(x, 1)[[1]]
  }

  train_ignore <- get_last_non_na(hist_ignore$train_loss)
  train_full   <- get_last_non_na(hist_full$train_loss)

  train_mse_ignore <- get_last_non_na(hist_ignore$train_mse)
  train_mse_full   <- get_last_non_na(hist_full$train_mse)

  train_bce_ignore <- get_last_non_na(hist_ignore$train_bce)
  train_bce_full   <- get_last_non_na(hist_full$train_bce)

  train_ce_ignore <- get_last_non_na(hist_ignore$train_ce)
  train_ce_full   <- get_last_non_na(hist_full$train_ce)

  cat("\n===== TRAIN DEBUG =====\n")
  cat("train_ignore:", train_ignore, "\n")
  cat("train_full:", train_full, "\n")
  cat("train_ce_ignore:", train_ce_ignore, "\n")
  cat("train_ce_full:", train_ce_full, "\n")

  expect_lt(abs(train_ignore - train_full), 0.5)
  expect_lt(abs(train_mse_ignore - train_mse_full), 10)
  expect_lt(abs(train_bce_ignore - train_bce_full), 10)
  expect_lt(abs(train_ce_ignore - train_ce_full), 10)

  # ----------------------------
  # VALIDATION
  # ----------------------------
  imp_ignore <- get_last_non_na(hist_ignore$imputation_error)
  imp_full   <- get_last_non_na(hist_full$imputation_error)

  mse_ignore <- get_last_non_na(hist_ignore$val_mse)
  mse_full   <- get_last_non_na(hist_full$val_mse)

  bce_ignore <- get_last_non_na(hist_ignore$val_bce)
  bce_full   <- get_last_non_na(hist_full$val_bce)

  ce_ignore <- get_last_non_na(hist_ignore$val_ce)
  ce_full   <- get_last_non_na(hist_full$val_ce)

  cat("\n===== VAL DEBUG =====\n")
  cat("imp_ignore:", imp_ignore, "\n")
  cat("imp_full:", imp_full, "\n")
  cat("ce_ignore:", ce_ignore, "\n")
  cat("ce_full:", ce_full, "\n")

  expect_gt(abs(imp_ignore - imp_full), 10)
  expect_gt(abs(mse_ignore - mse_full), 10)
  expect_gt(abs(bce_ignore - bce_full), 0.01)
  expect_gt(abs(ce_ignore - ce_full), 0.01)
})