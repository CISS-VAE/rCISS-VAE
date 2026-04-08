#' CISS-VAE imputation result object
#'
#' @description
#' An S3 class returned by the CISS-VAE imputation function. Inherits from
#' \code{list}, so all standard list operations (e.g., \code{$}, \code{[[})
#' work as expected. The class exists primarily to enable type-safe save/load
#' methods that correctly handle the embedded PyTorch model.
#'
#' @section Fields:
#' The following fields may be present depending on which \code{return_*}
#' arguments were set when the model was run:
#' \describe{
#'   \item{\code{imputed_dataset}}{Data frame. The full imputed dataset.}
#'   \item{\code{raw_data}}{Data frame. The original data before imputation.}
#'   \item{\code{model}}{Python object (\code{CISSVAE}). The trained PyTorch
#'     model. Present if \code{return_model = TRUE}.}
#'   \item{\code{cluster_dataset}}{Python object (\code{ClusterDataset}).
#'     Present if \code{return_dataset = TRUE}.}
#'   \item{\code{clusters}}{Integer vector. Cluster assignment per row.
#'     Present if \code{return_clusters = TRUE}.}
#'   \item{\code{silhouette_width}}{Numeric vector. Per-sample silhouette
#'     widths. Present if \code{return_silhouettes = TRUE}.}
#'   \item{\code{training_history}}{Data frame. Loss values recorded during
#'     training. Present if \code{return_history = TRUE}.}
#'   \item{\code{val_data}}{Data frame. Validation data with observed entries
#'     replaced by \code{NA}. Present if \code{return_validation_dataset = TRUE}.}
#'   \item{\code{val_imputed}}{Data frame. Model reconstructions of the
#'     validation entries. Present if \code{return_validation_dataset = TRUE}.}
#' }
#'
#' @seealso
#' \code{\link{save_impute_result}}, \code{\link{load_impute_result}},
#' \code{\link{print.impute_result}}, \code{\link{performance_by_cluster}}
#'
#' @name impute_result
NULL

# ============================================================
# Constructor (internal)
# ============================================================

#' Construct an \code{impute_result} object
#'
#' @description
#' Internal constructor. End users receive an \code{impute_result} from the
#' main imputation function; they do not need to call this directly.
#'
#' @param x List. A result list containing fields to include in the result object. 
#' @param ... Named fields to include in the result object. See
#'   \code{\link{impute_result}} for a description of recognised fields.
#'
#' @return An object of class \code{impute_result}.
#'
#' @seealso \code{\link{impute_result}}
#'
#' @keywords internal
new_impute_result <- function(x = NULL, ...) {
  if (!is.null(x)) {
    if (!is.list(x)) stop("`x` must be a list.")
    if (length(list(...)) > 0L) {
      stop("Supply either `x` or `...`, not both.")
    }
    return(structure(x, class = "impute_result"))
  }

  structure(list(...), class = "impute_result")
}

# ============================================================
# print method
# ============================================================

#' Print an \code{impute_result} object
#'
#' @description
#' Prints a structured summary of an \code{impute_result} object, including
#' the dimensions of the imputed dataset, a preview of the imputed data,
#' cluster composition, and which optional components are present.
#'
#' @param x An object of class \code{impute_result}.
#' @param n Integer. Number of rows of the imputed dataset to preview.
#'   Defaults to \code{6}.
#' @param ... Further arguments passed to or from other methods (currently
#'   unused).
#'
#' @return \code{x}, invisibly.
#'
#' @method print impute_result
#' @export
#' @examples
#' \donttest{
#' try({
#'   res <- ciss_vae(data = my_data, ...)
#'   print(res)      # default: 6 row preview
#'   print(res, n = 10)
#' })
#' }
print.impute_result <- function(x, n = 6, ...) {

  ruler <- paste0(rep("-", 60), collapse = "")

  # ── Header ────────────────────────────────────────────────
  cat("<impute_result>\n")
  cat(ruler, "\n")

  # ── Imputed dataset ───────────────────────────────────────
  if (!is.null(x$imputed_dataset)) {
    d <- x$imputed_dataset
    cat(sprintf(
      "Imputed dataset : %d rows x %d columns\n",
      nrow(d), ncol(d)
    ))

    n_show <- min(n, nrow(d))
    cat(sprintf("Preview (first %d rows):\n", n_show))
    print(d[seq_len(n_show), , drop = FALSE], row.names = TRUE)
  } else {
    cat("Imputed dataset : <not present>\n")
  }

  cat(ruler, "\n")

  # ── Cluster summary ───────────────────────────────────────
  if (!is.null(x$clusters)) {
    tbl <- table(x$clusters)
    cat("Clusters        :", length(tbl), "cluster(s)\n")
    cluster_summary <- paste(
      sprintf("  cluster %s: %d rows", names(tbl), as.integer(tbl)),
      collapse = "\n"
    )
    cat(cluster_summary, "\n")
  } else {
    cat("Clusters        : <not present>\n")
  }

  cat(ruler, "\n")

  # ── Optional components ───────────────────────────────────
  cat("Optional components:\n")

  .present <- function(field) {
    if (is.null(x[[field]])) "[ ]" else "[x]"
  }

  cat(sprintf(
    "  %s model              (CISSVAE PyTorch object)\n",   .present("model")))
  cat(sprintf(
    "  %s cluster_dataset    (ClusterDataset Python object)\n", .present("cluster_dataset")))
  cat(sprintf(
    "  %s silhouette_width   (%s)\n",
    .present("silhouette_width"),
    if (!is.null(x$silhouette_width))
      sprintf("n = %d", length(x$silhouette_width))
    else ""))
  cat(sprintf(
    "  %s training_history   (%s)\n",
    .present("training_history"),
    if (!is.null(x$training_history))
      sprintf("%d epochs x %d columns", nrow(x$training_history), ncol(x$training_history))
    else ""))
  cat(sprintf(
    "  %s val_data           (%s)\n",
    .present("val_data"),
    if (!is.null(x$val_data))
      sprintf("%d rows x %d cols", nrow(x$val_data), ncol(x$val_data))
    else ""))
  cat(sprintf(
    "  %s val_imputed        (%s)\n",
    .present("val_imputed"),
    if (!is.null(x$val_imputed))
      sprintf("%d rows x %d cols", nrow(x$val_imputed), ncol(x$val_imputed))
    else ""))
  if (!is.null(x$model)) {
    ag <- reticulate::py_to_r(x$model$activation_groups)
    cat("\nActivation groups:\n")
    print(ag)
  }

  cat(ruler, "\n")
  cat("Use save_impute_result() to save this object.\n")

  invisible(x)
}

# ============================================================
# Internal helper: extract CISSVAE constructor args from model
# ============================================================

#' Extract CISSVAE constructor arguments from a live model object
#'
#' @description
#' Reads the architecture-defining attributes from a live \code{CISSVAE}
#' Python object and returns them as a plain R list suitable for storage
#' in an RDS file and for passing back to the \code{CISSVAE} constructor
#' during model reconstruction.
#'
#' Special care is taken when converting \code{activation_groups} from Python
#' to R. Due to reticulate's automatic type simplification, Python lists of
#' length 1 may be converted to scalar values in R. This function ensures that
#' all activation group entries are returned as integer vectors, even when they
#' contain only a single column index.
#'
#' @param model A \code{CISSVAE} Python object (reticulate proxy).
#'
#' @return A named list with the following elements:
#' \describe{
#'   \item{\code{input_dim}}{Integer. Number of input features.}
#'   \item{\code{hidden_dims}}{Integer vector. Width of each hidden layer.}
#'   \item{\code{layer_order_enc}}{Character vector. Per-layer encoder
#'     directives (\code{"shared"} or \code{"unshared"}).}
#'   \item{\code{layer_order_dec}}{Character vector. Per-layer decoder
#'     directives (\code{"shared"} or \code{"unshared"}).}
#'   \item{\code{latent_shared}}{Logical. Whether latent heads are shared
#'     across clusters.}
#'   \item{\code{latent_dim}}{Integer. Dimensionality of the latent space.}
#'   \item{\code{output_shared}}{Logical. Whether the output layer is shared
#'     across clusters.}
#'   \item{\code{num_clusters}}{Integer. Number of clusters.}
#'   \item{\code{activation_groups}}{Named list mapping feature groups to
#'     integer vectors of column indices. Always returned as vectors, even
#'     for groups of size one.}
#' }
#'
#' @seealso \code{\link{save_impute_result}}
#'
#' @keywords internal
.cissvae_config <- function(model) {

  # Extract activation_groups from Python
  activation_groups <- reticulate::py_to_r(model$activation_groups)

  # force all groups to be integer vectors
  activation_groups <- lapply(activation_groups, function(x) {

  if (is.null(x)) {
    return(integer(0))
  }

  x <- unlist(x, use.names = FALSE)

  if (length(x) == 1L) {
    return(as.integer(c(x)))
  }

  return(as.integer(x))
})

  list(
    input_dim           = as.integer(reticulate::py_to_r(model$input_dim)),
    hidden_dims         = as.integer(reticulate::py_to_r(model$hidden_dims)),
    layer_order_enc     = as.character(reticulate::py_to_r(model$layer_order_enc)),
    layer_order_dec     = as.character(reticulate::py_to_r(model$layer_order_dec)),
    latent_shared       = as.logical(reticulate::py_to_r(model$latent_shared)),
    latent_dim          = as.integer(reticulate::py_to_r(model$latent_dim)),
    output_shared       = as.logical(reticulate::py_to_r(model$output_shared)),
    num_clusters        = as.integer(reticulate::py_to_r(model$num_clusters)),
    activation_groups   = activation_groups
  )
}

# ============================================================
# save_impute_result
# ============================================================

#' Save an \code{impute_result} object to disk
#'
#' @description
#' Saves an \code{impute_result} to a directory. R-native fields are written
#' to an RDS file. The \code{CISSVAE} PyTorch model, if present, is saved
#' separately as a \code{state_dict} checkpoint (\code{.pt}) so that the live
#' Python object is never passed to \code{saveRDS} (which cannot serialise it).
#' The model architecture is captured automatically from the live object and
#' stored inside the RDS so that \code{\link{load_impute_result}} can
#' reconstruct it without any additional input from the user.
#'
#' @param x An object of class \code{impute_result}.
#' @param dir Character string. Path to the directory in which to write files.
#'   Created recursively if it does not exist.
#' @param name Character string. Base name used for output files. Defaults to
#'   \code{"result"}. Two files are written:
#'   \itemize{
#'     \item \code{<name>.rds} — R data (all fields except the live model).
#'     \item \code{<name>_state_dict.pt} — PyTorch model weights.
#'   }
#' @param overwrite Logical. If \code{FALSE} (default), an error is raised when
#'   output files already exist. Set to \code{TRUE} to overwrite silently.
#'
#' @return Invisibly, a named list with elements \code{meta} (path to the RDS)
#'   and \code{model} (path to the \code{.pt} file, or \code{NULL} if no model
#'   was present).
#'
#' @seealso \code{\link{load_impute_result}}, \code{\link{impute_result}}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' res <- ciss_vae(data = my_data, return_model = TRUE, ...)
#' save_impute_result(res, dir = "run_001", name = "imputation_result")
#' }
save_impute_result <- function(x, dir, name = "result", overwrite = FALSE) {
  stopifnot(inherits(x, "impute_result"))
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)

  meta_path  <- file.path(dir, paste0(name, ".rds"))
  model_path <- file.path(dir, paste0(name, "_state_dict.pt"))

  if (!overwrite && (file.exists(meta_path) || file.exists(model_path))) {
    stop("Files already exist in '", dir, "'. Use overwrite = TRUE.")
  }

  x_to_save <- x
  

  if (!is.null(x_to_save$model)) {
    torch     <- reticulate::import("torch", delay_load = FALSE)
    x_to_save$model_config <- .cissvae_config(x_to_save$model)
    torch$save(x_to_save$model$state_dict(), model_path)

    x_to_save$model      <- NULL
    x_to_save$model_path <- basename(model_path)
  }

  saveRDS(x_to_save, meta_path)

  message("Saved to '", dir, "'")
  invisible(list(
    meta  = meta_path,
    model = if (file.exists(model_path)) model_path else NULL
  ))
}
# ============================================================
# load_impute_result
# ============================================================

#' Load an \code{impute_result} object from disk
#'
#' @description
#' Reads an \code{impute_result} previously saved by
#' \code{\link{save_impute_result}}. R-native fields are restored from the RDS
#' file. If a \code{CISSVAE} model was saved, the architecture is rebuilt from
#' the stored configuration, the \code{state_dict} weights are loaded, and the
#' model is placed in evaluation mode (\code{model.eval()}).
#'
#' The \code{CISSVAE} class is imported from the installed
#' \code{ciss_vae.classes} Python package using \code{reticulate::import}.
#'
#' @param dir Character string. Path to the directory written by
#'   \code{\link{save_impute_result}}.
#' @param name Character string. Base name used when saving.
#' @param device Character string. PyTorch device ("cpu" or "cuda").
#'
#' @return An \code{impute_result} object with model restored if present.
#'
#' @export
load_impute_result <- function(dir, name = "result", device = "cpu") {

  meta_path <- file.path(dir, paste0(name, ".rds"))
  if (!file.exists(meta_path)) {
    stop("No file found at '", meta_path, "'.")
  }

  x <- readRDS(meta_path)
  class(x) <- "impute_result"

  if (!is.null(x$model_path)) {

    if (is.null(x$model_config)) {
      stop("Saved object lacks model_config; cannot rebuild CISSVAE.")
    }

    cfg <- x$model_config

    # ============================================================
    # 🔴 STRICT VALIDATION (MUST COME FIRST)
    # ============================================================
    if (is.null(cfg$activation_groups) || !is.list(cfg$activation_groups)) {
      stop("activation_groups must be a non-null list.")
    }

    for (name in names(cfg$activation_groups)) {

      g <- cfg$activation_groups[[name]]

      if (is.null(g)) {
        stop(sprintf("activation_groups[['%s']] is NULL.", name))
      }

      if (!is.numeric(unlist(g))) {
        stop(sprintf("activation_groups[['%s']] must be numeric.", name))
      }
    }

    # ============================================================
    # NORMALIZE (safe AFTER validation)
    # ============================================================
    cfg$activation_groups <- lapply(cfg$activation_groups, function(x) {
      as.integer(unlist(x, use.names = FALSE))
    })

    # ============================================================
    # IMPORT PYTHON
    # ============================================================
    torch <- reticulate::import("torch", delay_load = FALSE)
    mod   <- reticulate::import("ciss_vae.classes", delay_load = FALSE)

    # ============================================================
    # BUILD MODEL (NO conversion hacks needed)
    # ============================================================
    model <- mod$CISSVAE(
      input_dim         = cfg$input_dim,
      hidden_dims       = cfg$hidden_dims,
      layer_order_enc   = cfg$layer_order_enc,
      layer_order_dec   = cfg$layer_order_dec,
      latent_shared     = cfg$latent_shared,
      latent_dim        = cfg$latent_dim,
      output_shared     = cfg$output_shared,
      num_clusters      = cfg$num_clusters,
      activation_groups = cfg$activation_groups,
      debug             = FALSE
    )

    # ============================================================
    # LOAD WEIGHTS
    # ============================================================
    state_path <- file.path(dir, x$model_path)

    if (!file.exists(state_path)) {
      stop("State dict file not found at '", state_path, "'.")
    }

    state_dict <- torch$load(state_path, map_location = device)

    model$load_state_dict(state_dict)
    model$eval()

    x$model <- model
    x$model_path <- NULL
  }

  return(x)
}