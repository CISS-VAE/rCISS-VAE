#' Save a trained Python CISS-VAE model to disk
#'
#' Uses reticulate to call Python's torch.save on a model object returned
#' from run_cissvae (or any Python model in the R session).
#'
#' @param model_py Python model object (e.g., res$model from run_cissvae)
#' @param file Path where the model will be saved (e.g., "trained_vae.pt")
#' @return NULL. Called for side effects.
#' @importFrom reticulate import
#' @examples
#' \donttest{
#' ## Requires a working Python environment via reticulate
#' ## Examples are wrapped in try() to avoid failures on CRAN check
#' try({
#'   reticulate::use_virtualenv("cissvae_environment", required = TRUE)
#'   data(df_missing)
#'   data(clusters)
#'
#'   ## Run CISS-VAE training
#'   dat <- try(
#'     run_cissvae(
#'       data = df_missing,
#'       index_col = "index",
#'       val_proportion = 0.1,
#'       columns_ignore = c(
#'         "Age", "Salary", "ZipCode10001", "ZipCode20002", "ZipCode30003"
#'       ),
#'       clusters = clusters$clusters,
#'       epochs = 5,
#'       return_silhouettes = FALSE,
#'       return_history = TRUE,
#'       verbose = FALSE,
#'       return_model = TRUE,
#'       device = "cpu",
#'       layer_order_enc = c("unshared", "shared", "unshared"),
#'       layer_order_dec = c("shared", "unshared", "shared")
#'     ),
#'     silent = TRUE
#'   )
#'
#'   ## Save the trained model to a temporary file (CRAN-safe)
#'   tmpfile <- tempfile(fileext = ".pt")
#'   try(save_cissvae_model(dat$model, tmpfile), silent = TRUE)
#' })}
#' @export
save_cissvae_model <- function(model_py, file) {
  # Import torch via reticulate
  torch <- reticulate::import("torch")
  # Save full model (architecture + weights)
  torch$save(model_py, file)
}


#' Load a saved Python CISS-VAE model
#'
#' Loads a full CISSVAE model into R via reticulate.
#'
#' @param file Path to the saved model file (e.g., "trained_vae.pt")
#' @param python_env Optional: Python virtualenv or conda env to activate
#' @return CISSVAE model object
#' @examples
#' \donttest{
#' ## Requires a working Python environment via reticulate
#' ## Wrapped in try() and donttest to avoid CRAN check failures
#' try({
#'   # Activate the Python virtualenv or conda env where CISSVAE is installed
#'   reticulate::use_virtualenv("cissvae_environment", required = TRUE)
#'
#'   # Path to a previously saved model file
#'   model_file <- "trained_vae.pt"
#'
#'   # Load the CISS-VAE model
#'   loaded_model <- try(
#'     load_cissvae_model(file = model_file, python_env = "cissvae_environment")
#'   )
#'
#' })
#' }
#' @export
load_cissvae_model <- function(file, python_env = NULL) {
  if (!is.null(python_env)) {
    reticulate::use_virtualenv(python_env, required = TRUE)
  }

  # Import torch and ensure VAE class is registered
  torch <- reticulate::import("torch")
  reticulate::import("ciss_vae.classes.vae")

  # Load and put model into eval mode
  model <- torch$load(file, map_location = "cpu", weights_only = FALSE)
  model$eval()
  return(model)
}




#' Impute new data with a loaded Python CISS-VAE model
#'
#' Given a loaded model, an R data frame, and a vector of cluster labels,
#' this builds the Python ClusterDataset and DataLoader, runs inference,
#' and returns an imputed data frame in R.
#'
#' @param model_py Python model object loaded via load_cissvae_model()
#' @param data R data.frame with missing values
#' @param index_col String name of index column to preserve (optional)
#' @param columns_ignore Character vector of column names to exclude from imputation scoring.
#' @param clusters Integer vector of cluster labels for rows of data
#' @param imputable_matrix Logical matrix indicating entries allowed to be imputed.
#' @param binary_feature_mask Logical vector marking which columns are binary.
#' @param val_proportion Proportion of non-missing data to hold out for validation.
#' @param replacement_value Numeric value used to replace missing entries before model input.
#' @param batch_size Batch size passed to Python DataLoader (default 4000L)
#' @param seed Base random seed for reproducible results
#' @return Imputed R data.frame
#' @section Tips:
#' \itemize{
#'   \item Use same ClusterDataset parameters as for initial model training. 
#'   \item Clusters must have same labels as clusters used for model training
#'   \item 'binary_feature_mask' is required for correct imputation of binary columns. 
#' }
#' @examples
#' ## Requires a working Python environment via reticulate
#' ## Wrapped in try() and donttest to avoid CRAN check failures
#' \donttest{
#' try({
#'   # Activate your reticulate Python environment with ciss-vae installed
#'   reticulate::use_virtualenv("cissvae_environment", required = TRUE)
#'
#'   # Load example data and clusters (replace with your own)
#'   data(df_missing)
#'   data(clusters)
#'
#'   # Load a previously saved model
#'   model <- try(load_cissvae_model("model.pt", python_env = "cissvae_environment"))
#'
#'   # Perform imputation on new data
#'   imputed_df <- try(
#'     impute_with_cissvae(
#'       model_py = model,
#'       data = df_missing,
#'       index_col = "index",
#'       columns_ignore = c("Age", "Salary"),
#'       clusters = clusters$clusters,
#'       imputable_matrix = NULL,
#'       binary_feature_mask = NULL,
#'       val_proportion = 0.1,
#'       replacement_value = 0,
#'       batch_size = 4000L,
#'       seed = 42
#'     )
#'   )
#' })
#' }
#' @export
impute_with_cissvae <- function(model_py,  data, index_col = NULL, columns_ignore= NULL,  
  clusters, imputable_matrix = NULL, binary_feature_mask = NULL, val_proportion = 0.1, replacement_value = 0, batch_size = 4000L, seed = 42) {

  ## --------------- Setup Stuff ----------------------
  ## Check if reticulate has initialized Python
  if (is.null(reticulate::py_config()$python)) {
      stop(
        "Python is not initialized in this session. ",
        "Please activate a reticulate Python environment before calling this function, ",
        "for example by calling:\n",
        "  reticulate::use_virtualenv(\"your/env/path\", required = TRUE)\n",
        "or\n",
        "  reticulate::use_condaenv(\"your_env_name\", required = TRUE)\n",
        "and then re-run the function.",
        call. = FALSE
      )
    }   

  ## Python obj chekcer
  is_py_obj <- function(x) inherits(x, "python.builtin.object")
  ## -------------- End Setup Stuff --------------------

  ## -------------- Handle Index column -----------------

  if (!is.null(index_col)) {
    if (!index_col %in% colnames(data)) stop("`index_col` not found in data.")
    index_vals <- data[[index_col]]
    data       <- data[, setdiff(colnames(data), 
      index_col), drop = FALSE]
    imputable_matrix = imputable_matrix[, setdiff(colnames(imputable_matrix), 
      index_col), drop = FALSE]
    ## handle index col in binary_feature_mask
    if(!is.null(binary_feature_mask) & !is.null(names(binary_feature_mask))){
      binary_feature_mask =  binary_feature_mask[ setdiff(names(binary_feature_mask), 
        index_col), drop = FALSE]
      if(debug){
        cat("Binary feature mask: ", 
        paste0(names(binary_feature_mask), collapse = ", "))
      }
    }
  } else index_vals <- NULL

  # Ensure data and mask columns match EXACTLY (after index removal) 
if (!is.null(imputable_matrix)) {
  if (!identical(colnames(data), colnames(imputable_matrix))) {
    stop("`data` and `imputable_matrix` columns must match exactly (same names & order) after removing index_col.")
  }
}

  ## --------------- End Handle Index Column ---------------

  ## --------------- Prepare data and python imports ----------
  np       <- reticulate::import("numpy", convert = FALSE)
  pd       <- reticulate::import("pandas", convert = FALSE)
  CD_mod   <- reticulate::import("ciss_vae.classes.cluster_dataset", 
  convert = FALSE)$ClusterDataset
  helpers <- import("ciss_vae.utils.helpers")
  DataLoader <- import("torch.utils.data")$DataLoader

  data[is.na(data)] <- NaN
  data_py <- pd$DataFrame(data = data, dtype = "float64")
  clusters_py <- np$array(as.integer(clusters), dtype = "int64")

  # columns_ignore as a Python list (safer when convert = FALSE)
  cols_ignore_py <- if (is.null(columns_ignore)) NULL else reticulate::r_to_py(as.character(columns_ignore))

  if (!is.null(imputable_matrix)) {
      dni_py <- pd$DataFrame(imputable_matrix)
  } else dni_py <- NULL

  ## Build dataset + loader
  dataset_py   <- CD_mod(
    data = data_py, 
    cluster_labels = clusters_py, 
    val_proportion = val_proportion, 
    replacement_value = replacement_value, 
    columns_ignore = cols_ignore_py, 
    imputable = dni_py,
    val_seed = seed,
    binary_feature_mask =  reticulate::r_to_py(binary_feature_mask))
  data_loader  <- DataLoader(dataset_py, batch_size = as.integer(batch_size))

  ## Run model inference & get imputed df
  imputed_py   <- helpers$get_imputed_df(model_py, data_loader)

  ## Convert Python pandas DataFrame back to R
  imputed_r    <- reticulate::py_to_r(imputed_py)
  return(imputed_r)
}