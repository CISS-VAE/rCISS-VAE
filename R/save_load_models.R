#' Save a trained CISS-VAE model to disk
#'
#' Saves a trained \code{CISSVAE} PyTorch model to disk. The default method
#' (\code{method = "state_dict"}) saves only the model weights alongside an
#' automatically captured architecture config, which is the recommended
#' approach for portability and long-term reproducibility. A full-object save
#' (\code{method = "full"}) is also available but is less portable across
#' Python / PyTorch versions.
#'
#' @param model A \code{CISSVAE} Python object (reticulate proxy), e.g.,
#'   \code{res$model} from \code{run_cissvae()}.
#' @param file Character string. File path for the saved model. For
#'   \code{method = "state_dict"} two files are written:
#'   \itemize{
#'     \item \code{<file>} — the PyTorch \code{state_dict} checkpoint
#'       (\code{.pt}).
#'     \item \code{<file>.config.rds} — the architecture config needed to
#'       reconstruct the model on load.
#'   }
#'   For \code{method = "full"} only \code{<file>} is written.
#' @param method Character string. One of \code{"state_dict"} (default) or
#'   \code{"full"}.
#'   \describe{
#'     \item{\code{"state_dict"}}{Saves weights and architecture config
#'       separately. Recommended. Requires \code{load_cissvae_model()} to
#'       reload.}
#'     \item{\code{"full"}}{Saves the entire Python object with
#'       \code{torch.save}. Less portable; may break across PyTorch versions.}
#'   }
#' @param overwrite Logical. If \code{FALSE} (default), an error is raised
#'   when output files already exist.
#'
#' @return \code{NULL}, invisibly. Called for its side effects.
#'
#' @seealso \code{\link{load_cissvae_model}}, \code{\link{save_impute_result}}
#'
#' @importFrom reticulate import py_has_attr py_to_r
#'
#' @export
#'
#' @examples
#' \donttest{
#' try({
#'   reticulate::use_virtualenv("cissvae_environment", required = TRUE)
#'   data(df_missing)
#'   data(clusters)
#'
#'   dat <- run_cissvae(
#'     data             = df_missing,
#'     index_col        = "index",
#'     val_proportion   = 0.1,
#'     clusters         = clusters$clusters,
#'     epochs           = 5,
#'     return_model     = TRUE,
#'     device           = "cpu",
#'     layer_order_enc  = c("unshared", "shared", "unshared"),
#'     layer_order_dec  = c("shared",   "unshared", "shared")
#'   )
#'
#'   # default: saves state_dict + config
#'   save_cissvae_model(dat$model, file = tempfile(fileext = ".pt"))
#'
#'   # alternative: full object save
#'   save_cissvae_model(dat$model, file = tempfile(fileext = ".pt"),
#'                      method = "full")
#' })
#' }
save_cissvae_model <- function(model,
                               file,
                               method    = c("state_dict", "full"),
                               overwrite = FALSE) {

  method <- match.arg(method)
  torch  <- reticulate::import("torch", delay_load = FALSE)

  config_path <- paste0(file, ".config.rds")

  if (!overwrite) {
    existing <- c(
      if (file.exists(file))        file,
      if (method == "state_dict" &&
          file.exists(config_path)) config_path
    )
    if (length(existing) > 0) {
      stop(
        "File(s) already exist: ",
        paste(existing, collapse = ", "),
        ". Use overwrite = TRUE."
      )
    }
  }

  if (method == "state_dict") {

    if (!reticulate::py_has_attr(model, "state_dict")) {
      stop("`model` does not appear to be a torch nn.Module (missing state_dict).")
    }

    # capture architecture config from the live model
    config <- .cissvae_config(model)

    torch$save(model$state_dict(), file)
    saveRDS(config, config_path)

    message("Saved state_dict to '", file, "'")
    message("Saved model config to '", config_path, "'")

  } else {

    torch$save(model, file)
    message("Saved full model object to '", file, "'")

  }

  invisible(NULL)
}


#' Load a saved CISS-VAE model
#'
#' Loads a \code{CISSVAE} model previously saved by
#' \code{\link{save_cissvae_model}}. When the model was saved with
#' \code{method = "state_dict"} (the default), the architecture is
#' automatically reconstructed from the paired \code{.config.rds} file and
#' the weights are loaded via \code{load_state_dict}. When the model was saved
#' with \code{method = "full"}, the entire Python object is deserialised
#' directly.
#'
#' The \code{CISSVAE} class is imported from the installed
#' \code{ciss_vae.classes} using \code{reticulate::import}
#' \code{reticulate::import}).
#'
#' @param file Character string. Path to the saved model file (\code{.pt}).
#'   For \code{method = "state_dict"}, a paired \code{<file>.config.rds} must
#'   exist in the same location (written automatically by
#'   \code{\link{save_cissvae_model}}).
#' @param method Character string. One of \code{"state_dict"} (default) or
#'   \code{"full"}. Must match the method used when saving.
#' @param device Character string. PyTorch device string passed to
#'   \code{torch.load} as \code{map_location}. Defaults to \code{"cpu"}.
#'   Use \code{"cuda"} to load directly onto GPU.
#' @param python_env Optional character string. Name of a virtualenv or conda
#'   environment to activate before loading. If \code{NULL} (default), the
#'   currently active Python session is used.
#'
#' @return A \code{CISSVAE} Python object in evaluation mode
#'   (\code{model.eval()} has been called).
#'
#' @seealso \code{\link{save_cissvae_model}}, \code{\link{load_impute_result}}
#'
#' @importFrom reticulate import py_run_string use_virtualenv
#'
#' @export
#'
#' @examples
#' \donttest{
#' try({
#'   reticulate::use_virtualenv("cissvae_environment", required = TRUE)
#'
#'   pt_file <- tempfile(fileext = ".pt")
#'
#'   # save first
#'   save_cissvae_model(dat$model, file = pt_file)
#'
#'   # reload in same or new session
#'   model <- load_cissvae_model(file = pt_file, device = "cpu")
#' })
#' }
load_cissvae_model <- function(file,
                               method     = c("state_dict", "full"),
                               device     = "cpu",
                               python_env = NULL) {

  method <- match.arg(method)

  if (!is.null(python_env)) {
    reticulate::use_virtualenv(python_env, required = TRUE)
  }

  if (!file.exists(file)) {
    stop("Model file not found: '", file, "'.")
  }

  torch <- reticulate::import("torch", delay_load = FALSE)

  if (method == "state_dict") {

    config_path <- paste0(file, ".config.rds")
    if (!file.exists(config_path)) {
      stop(
        "Config file not found: '", config_path, "'. ",
        "Expected alongside '", file, "' when method = 'state_dict'. ",
        "If the model was saved with method = 'full', use method = 'full' here."
      )
    }

    cfg <- readRDS(config_path)

    if (is.null(cfg$activation_groups)) {
      stop("Missing activation_groups in saved model config.")
    }
    if (!is.list(cfg$activation_groups)) {
      stop("activation_groups must be a list.")
    }
    
    mod <- reticulate::import("ciss_vae.classes", delay_load = FALSE)

    model <- mod$CISSVAE(
      input_dim           = cfg$input_dim,
      hidden_dims         = cfg$hidden_dims,
      layer_order_enc     = cfg$layer_order_enc,
      layer_order_dec     = cfg$layer_order_dec,
      latent_shared       = cfg$latent_shared,
      latent_dim          = cfg$latent_dim,
      output_shared       = cfg$output_shared,
      num_clusters        = cfg$num_clusters,
      ## Now using activation_groups instead of bfm!!
      activation_groups = cfg$activation_groups,
      debug               = FALSE
    )

    sd <- torch$load(file, map_location = device)
    model$load_state_dict(sd)

  } else {

    model <- torch$load(file, map_location = device, weights_only = FALSE)

  }

  model$eval()
  model
}



#' Impute new data with a loaded Python CISS-VAE model
#'
#' Given a loaded model, an R data frame, and a vector of cluster labels,
#' this builds the Python ClusterDataset and DataLoader, runs inference,
#' and returns an imputed data frame in R.
#'
#' @param model Python model object loaded via load_cissvae_model()
#' @param data R data.frame with missing values
#' @param index_col String name of index column to preserve (optional)
#' @param cols_ignore Character vector of column names to exclude from imputation scoring.
#' @param clusters Integer vector of cluster labels for rows of data
#' @param imputable_matrix Logical matrix indicating entries allowed to be imputed.
#' @param binary_feature_mask Logical vector marking which columns are binary.
#' @param categorical_column_map Named list. Maps generated dummy variable column names to the original categorical column name.
#' @param replacement_value Numeric value used to replace missing entries before model input.
#' @param batch_size Batch size passed to Python DataLoader. If NULL, batch_size = nrow(data)
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
#'   # Activate your reticulate Python environment with ciss_vae installed
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
#'       model = model,
#'       data = df_missing,
#'       index_col = "index",
#'       cols_ignore = c("Age", "Salary"),
#'       clusters = clusters$clusters,
#'       imputable_matrix = NULL,
#'       binary_feature_mask = NULL,
#'       replacement_value = 0,
#'       batch_size = 4000L,
#'       seed = 42
#'     )
#'   )
#' })
#' }
#' @export
impute_with_cissvae <- function(model,  data, index_col = NULL, cols_ignore= NULL,  
  clusters, imputable_matrix = NULL, binary_feature_mask = NULL, categorical_column_map = NULL,
  replacement_value = 0, batch_size = NULL, seed = 42) {

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
    if (!is.null(imputable_matrix)) {
      imputable_matrix <- imputable_matrix[, setdiff(colnames(imputable_matrix), index_col), drop = FALSE]
    }
    ## handle index col in binary_feature_mask
    if(!is.null(binary_feature_mask) && !is.null(names(binary_feature_mask))){
      binary_feature_mask =  binary_feature_mask[ setdiff(names(binary_feature_mask), 
        index_col), drop = FALSE]
      
    }
  } else index_vals <- NULL

  # Ensure data and mask columns match EXACTLY (after index removal) 
if (!is.null(imputable_matrix)) {
  if (!identical(colnames(data), colnames(imputable_matrix))) {
    stop("`data` and `imputable_matrix` columns must match exactly (same names & order) after removing index_col.")
  }
}

  ## --------------- End Handle Index Column ---------------

  ## ----------------- Set batch size to dataset size if not defined ----------------
  if(is.null(batch_size)){
    batch_size = as.integer(nrow(data))
  }
  if(batch_size > nrow(data)){
     batch_size = as.integer(nrow(data))
  }
  batch_size = as.integer(batch_size)

  ## --------------- Prepare data and python imports ----------
  np       <- reticulate::import("numpy", convert = FALSE)
  pd       <- reticulate::import("pandas", convert = FALSE)
  CD_mod   <- reticulate::import("ciss_vae.classes.cluster_dataset", 
  convert = FALSE)$ClusterDataset
  helpers <- reticulate::import("ciss_vae.utils.helpers")
  DataLoader <- reticulate::import("torch.utils.data")$DataLoader

  data[is.na(data)] <- NaN
  data_py <- pd$DataFrame(data = data, dtype = "float64")
  clusters_py <- np$array(as.integer(clusters), dtype = "int64")
  seed <- as.integer(seed)
  # cols_ignore as a Python list (safer when convert = FALSE)
  cols_ignore_py <- if (is.null(cols_ignore)) NULL else reticulate::r_to_py(as.character(cols_ignore))

  if (!is.null(imputable_matrix)) {
      dni_py <- pd$DataFrame(imputable_matrix)
  } else dni_py <- NULL

  ## ---------- make sure everything is valid before sending ot python -------------------------

# Ensure we have at least 1 feature column after all filtering
if (ncol(data) == 0L) {
  stop(
    "After dropping index_col/cols_ignore, there are 0 feature columns left. ",
    "Check `index_col` and `cols_ignore` against your input data."
  )
}

# Ensure no duplicated column names (pandas can behave badly here)
if (anyDuplicated(colnames(data))) {
  stop("Duplicate column names detected after preprocessing. Resolve duplicates before imputing.")
}

# If you have a binary feature mask, it MUST match the number of feature cols being sent
if (!is.null(binary_feature_mask)) {
  if (length(binary_feature_mask) != ncol(data)) {
    stop(
      "binary_feature_mask length (", length(binary_feature_mask),
      ") does not match number of feature columns sent to python (", ncol(data), "). ",
      "The mask must be defined AFTER column dropping/reordering."
    )
  }
}

# clusters must match rows
if (!is.null(clusters) && length(clusters) != nrow(data)) {
  stop("`clusters` length must equal number of rows in the data being imputed.")
}

# Ensure seed is an integer for python
if (!is.null(seed)) seed <- as.integer(seed)

# Ensure batch size is a positive integer 
if (!is.null(batch_size)) {
  batch_size <- as.integer(batch_size)
  if (is.na(batch_size) || batch_size < 1L) stop("`batch_size` must be a positive integer.")
}

## --------------- send things to python =-=--------------
  ## Build dataset + loader
  dataset_py   <- CD_mod(
    data = data_py, 
    cluster_labels = clusters_py, 
    val_proportion = 0L, 
    replacement_value = replacement_value, 
    columns_ignore = cols_ignore_py, 
    imputable = dni_py,
    val_seed = seed,
    binary_feature_mask =  reticulate::r_to_py(binary_feature_mask),
    categorical_column_map = reticulate::r_to_py(categorical_column_map))
  data_loader  <- DataLoader(dataset_py, batch_size = as.integer(batch_size))

    ## Sanity checks: loaded model must have a non-empty layer order
  lo_enc <- tryCatch(model$layer_order_enc, error = function(e) NULL)
  lo_dec <- tryCatch(model$layer_order_dec, error = function(e) NULL)

  len0 <- function(x) is.null(x) || length(x) == 0

  if (len0(lo_enc) || len0(lo_dec)) {
    stop(
      "Loaded model has empty `layer_order_enc`/`layer_order_dec`. ",
      "This usually means the R load routine reconstructed the model incorrectly ",
      "(e.g., defaults) rather than restoring the original architecture/config."
    )
  }

  ## Run model inference & get imputed df
  imputed_py   <- helpers$get_imputed_df(model, data_loader)

  ## Convert Python pandas DataFrame back to R
  imputed_r    <- reticulate::py_to_r(imputed_py)

  if(!is.null(index_col)){
      imputed_r[[index_col]] = index_vals
  imputed_r <- imputed_r[c(index_col, setdiff(names(imputed_r), index_col))]
  }

  return(imputed_r)
}