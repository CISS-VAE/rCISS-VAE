simulate_missingness <- function(
  df,
  mcar_frac    = 0.1,    # overall fraction MCAR
  mar_frac     = 0.1,    # overall fraction MAR
  mar_var      = NULL,   # name of a column in df (e.g. "sex" or "stage")
  mnar_frac    = 0.1,    # overall fraction MNAR (latent)
  latent_groups= 5,      # how many "ZIP‐code" groups to simulate
  target_vars  = NULL,   # which cols to corrupt; defaults to all numeric ≠ id, mar_var
  seed         = NULL    # for reproducibility
) {
  if(!is.null(seed)) set.seed(seed)
  
  df_out <- df
  n       <- nrow(df)
  
  # 1) decide which columns to touch
  if (is.null(target_vars)) {
    numeric_cols <- names(df)[sapply(df, is.numeric)]
    target_vars  <- setdiff(numeric_cols, c("id", mar_var))
  }
  
  # 2) MCAR: pick random cells across those columns
  total_cells <- n * length(target_vars)
  mcar_n      <- round(mcar_frac * total_cells)
  pos         <- sample(total_cells, mcar_n)
  row_idx     <- ((pos - 1) %% n) + 1
  col_idx     <- ((pos - 1) %/% n) + 1
  for(i in seq_along(pos)) {
    df_out[row_idx[i], target_vars[col_idx[i]]] <- NA
  }
  
  # 3) MAR: probability of missingness by mar_var
  if (!is.null(mar_var)) {
    # extract & standardize
    z        <- df[[mar_var]]
    if (!is.numeric(z)) z <- as.numeric(as.factor(z))
    z        <- (z - mean(z, na.rm=TRUE)) / sd(z, na.rm=TRUE)
    p_mar    <- plogis(qlogis(mar_frac) + z)
    # for each target column, drop with row‐specific p_mar
    for (col in target_vars) {
      miss_i <- runif(n) < p_mar
      df_out[miss_i, col] <- NA
    }
  }
  
  # 4) MNAR‐style: simulate a hidden ZIP code factor
  #    (we do NOT return zip; model is unaware)
  zip_codes   <- sample(10000:99999, latent_groups)
  assigned_zips <- sample(zip_codes, n, replace = TRUE)
  # give each ZIP a random “effect”
  zip_eff     <- rnorm(latent_groups)
  names(zip_eff) <- as.character(zip_codes)
  p_mnar      <- plogis(qlogis(mnar_frac) + zip_eff[as.character(assigned_zips)])
  for (col in target_vars) {
    miss_i <- runif(n) < p_mnar
    df_out[miss_i, col] <- NA
  }
  
  return(df_out)
}
