run_cissvae <- function(
  data,
  index_col              = NULL,
  val_proportion         = 0.1,
  replacement_value      = 0.0,
  columns_ignore         = NULL,
  print_dataset          = TRUE,
  ## Cluster stuff
  clusters               = NULL,
  n_clusters             = NULL,
  seed                   = 42,
  missingness_proportion_matrix = NULL,
  scale_features         = FALSE,
  k_neighbors            = 15L,
  leiden_resolution      = 0.5,
  leiden_objective       = "CPM",
  ## Model Stuff
  hidden_dims            = c(150, 120, 60),
  latent_dim             = 15,
  layer_order_enc        = c("unshared", "unshared", "unshared"),
  layer_order_dec        = c("shared", "shared", "shared"),
  latent_shared          = FALSE,
  output_shared          = FALSE,
  batch_size             = 4000,
  epochs                 = 500,
  initial_lr             = 0.01,
  decay_factor           = 0.999,
  beta                   = 0.001,
  device                 = NULL,
  max_loops              = 100,
  patience               = 2,
  epochs_per_loop        = NULL,
  initial_lr_refit       = NULL,
  decay_factor_refit     = NULL,
  beta_refit             = NULL,
  ## other params
  verbose                = FALSE,
  return_model           = TRUE,
  return_clusters        = FALSE,     # NEW
  return_silhouettes     = FALSE,
  return_history         = FALSE,
  return_dataset         = FALSE,
  ## NEw stuff from the ptyhon update
  do_not_impute_matrix   = NULL,
  
  debug                  = FALSE
){
  ## step 1: coerse numerics 
  seed            <- as.integer(seed)
  latent_dim      <- as.integer(latent_dim)
  batch_size      <- as.integer(batch_size)
  epochs          <- as.integer(epochs)
  max_loops       <- as.integer(max_loops)
  patience        <- as.integer(patience)
  n_clusters      <- if (!is.null(n_clusters)) as.integer(n_clusters) else NULL
  epochs_per_loop <- if (!is.null(epochs_per_loop)) as.integer(epochs_per_loop) else NULL
  hidden_dims     <- as.integer(hidden_dims)
  layer_order_enc <- as.character(layer_order_enc)
  layer_order_dec <- as.character(layer_order_dec)
  
  ## step 2: handle index col if passed
  if (!is.null(index_col)) {
    if (!index_col %in% colnames(data)) stop("`index_col` not found in data.")
    index_vals <- data[[index_col]]
    data       <- data[, setdiff(colnames(data), index_col), drop = FALSE]
  } else {
    index_vals <- NULL
  }
  ## grab original row and column names
  orig_rn <- if (is.data.frame(data) || is.matrix(data)) rownames(data) else NULL
  orig_cn <- if (is.data.frame(data) || is.matrix(data)) colnames(data) else NULL


  ## step 3: do python imports

  run_mod <- reticulate::import("ciss_vae.utils.run_cissvae", convert = FALSE)
  np      <- reticulate::import("numpy", convert = FALSE)
  pd      <- reticulate::import("pandas", convert = FALSE)


  ## step 4: handle whatever type of object `data` is (i.e. if it's a python object vs r dataframe vs matrix)
  is_py_obj <- function(x) inherits(x, "python.builtin.object")
  if (is_py_obj(data)) {
    # If it's already a pandas object with .isna, pass through
    if (reticulate::py_has_attr(data, "isna")) {
      data_py <- data
    } else {
      # Best effort: wrap into pandas DataFrame
      data_py <- pd$DataFrame(reticulate::r_to_py(as.data.frame(data)))
    }
  } else if (is.data.frame(data)) {
    data_py <- pd$DataFrame(reticulate::r_to_py(data))
  } else if (is.matrix(data)) {
    # Keep column names if present
    df_tmp <- as.data.frame(data, stringsAsFactors = FALSE)
    colnames(df_tmp) <- colnames(data)
    data_py <- pd$DataFrame(reticulate::r_to_py(df_tmp))
  } else {
    # fallback
    data_py <- pd$DataFrame(reticulate::r_to_py(as.data.frame(data)))
  }

  ## step 5: perpare python args
  if (!is.null(clusters)) { ## check for clusters
    if (is.data.frame(clusters)) clusters <- clusters[[1]]
    clusters <- as.vector(clusters)
    clusters_py <- np$array(as.integer(clusters))
  } else clusters_py <- NULL

  if (!is.null(missingness_proportion_matrix)) {
    if (is_py_obj(missingness_proportion_matrix)) {
      prop_matrix_py <- missingness_proportion_matrix
    } else if (is.data.frame(missingness_proportion_matrix)) {
      prop_matrix_py <- pd$DataFrame(reticulate::r_to_py(missingness_proportion_matrix))
    } else {
      prop_matrix_py <- reticulate::r_to_py(as.matrix(missingness_proportion_matrix))
    }
  } else prop_matrix_py <- NULL

  if (!is.null(do_not_impute_matrix)) {
    if (is_py_obj(do_not_impute_matrix)) {
      dni_py <- do_not_impute_matrix
    } else {
      dni_py <- reticulate::r_to_py(as.matrix(do_not_impute_matrix))
    }
  } else dni_py <- NULL

  py_args <- list(
    data                  = data_py,
    val_proportion        = val_proportion,
    replacement_value     = replacement_value,
    columns_ignore        = if (is.null(columns_ignore)) NULL else reticulate::r_to_py(columns_ignore),
    print_dataset         = print_dataset,
    clusters              = clusters_py,
    n_clusters            = n_clusters,
    seed                  = seed,
    missingness_proportion_matrix = prop_matrix_py,
    scale_features        = scale_features,
    hidden_dims           = reticulate::r_to_py(hidden_dims),
    latent_dim            = latent_dim,
    layer_order_enc       = reticulate::r_to_py(layer_order_enc),
    layer_order_dec       = reticulate::r_to_py(layer_order_dec),
    latent_shared         = latent_shared,
    output_shared         = output_shared,
    batch_size            = batch_size,
    return_model          = return_model,
    epochs                = epochs,
    initial_lr            = initial_lr,
    decay_factor          = decay_factor,
    beta                  = beta,
    device                = device,
    max_loops             = max_loops,
    patience              = patience,
    epochs_per_loop       = epochs_per_loop,
    initial_lr_refit      = initial_lr_refit,
    decay_factor_refit    = decay_factor_refit,
    beta_refit            = beta_refit,
    verbose               = verbose,
    return_clusters       = return_clusters,
    return_silhouettes    = return_silhouettes,
    return_history        = return_history,
    return_dataset        = return_dataset,
    do_not_impute_matrix = dni_py,
    k_neighbors           = as.integer(k_neighbors),
    leiden_resolution     = as.numeric(leiden_resolution),
    leiden_objective      = as.character(leiden_objective),
    debug                 = debug
  )
  ## drop nulls from python args
  py_args <- py_args[!vapply(py_args, is.null, logical(1))]

  ## call python w/ these args

  res_py = do.call(run_mod$run_cissvae, py_args)

  ## res_py[0] = imputed df, res_py[1] is model
  
  res_r = reticulate::py_to_r(res_py)
  imputed_df = res_r[[1]]

  ## put index back on
  if (!is.null(index_vals) && length(index_vals) == nrow(imputed_df)) {
    imputed_df[[index_col]] <- index_vals
    imputed_df <- imputed_df[, c(index_col, setdiff(names(imputed_df), index_col)), drop = FALSE]
  } else if (!is.null(index_vals) && length(index_vals) != nrow(imputed_df) && isTRUE(verbose)) {
    message("run_cissvae(): index_col length mismatch; not attaching index_col.")
  }

  out = list(imputed_dataset = imputed_df)
  
  i = 2 ## starting from second entry in res_r
  if(return_model){
    out[["model"]] = res_r[i]
    i = i+1
  }
  if(return_dataset){
    out[["dataset"]] = res_r[i]
    i = i+1
  }
  if(return_clusters){
    out[["clusters"]] = res_r[i]
    i = i+1
  }
  if(return_silhouettes){
    out[["silhouette_width"]] = res_r[i]
    i = i+1
  }
  if(return_history){
    out[["training_history"]] = res_r[i]
    i = i+1
  }

    
  return(out)

}