
## helper function to extract the vae metadata
#' Extract CISSVAE architecture metadata
#'
#' @param py_model A reticulate-imported Python CISSVAE object.
#' @return A data.frame with one row per layer:
#'   - phase: "encoder", "latent", "decoder", or "output"  
#'   - layer_idx: integer index within the phase  
#'   - type: "shared" or "unshared"  
#'   - cluster: NA for shared layers; cluster ID for unshared  
#'   - size: integer number of units  
#' @export
extract_cissvae_arch <- function(py_model) {
  # Encoder ---------------------------------------------------------------
  hidden_dims   <- as.integer(unlist(reticulate::py_to_r(py_model$hidden_dims)))
  layer_order_e <- unlist(reticulate::py_to_r(py_model$layer_order_enc))
  
  enc_list <- lapply(seq_along(hidden_dims), function(i) {
    data.frame(
      phase     = "encoder",
      layer_idx = i,
      type      = tolower(layer_order_e[i]),
      cluster   = if (tolower(layer_order_e[i]) == "shared") NA_integer_ else NA_integer_,
      size      = hidden_dims[i],
      stringsAsFactors = FALSE
    )
  })
  # Now fill cluster for unshared: count how many unshared before and assign cluster keys afterward
  # But since all unshared encoder layers are duplicated per cluster in Python,
  # we'll just mark them as "unshared" without cluster here.
  
  enc_df <- do.call(rbind, enc_list)
  
  # Latent ----------------------------------------------------------------
  latent_dim    <- as.integer(py_model$latent_dim)
  latent_shared <- as.logical(py_model$latent_shared)
  
  if (latent_shared) {
    lat_df <- data.frame(
      phase     = "latent",
      layer_idx = 1L,
      type      = "shared",
      cluster   = NA_integer_,
      size      = latent_dim,
      stringsAsFactors = FALSE
    )
  } else {
    # one row per cluster
    num_clust <- as.integer(py_model$num_clusters)
    lat_df <- data.frame(
      phase     = "latent",
      layer_idx = 1L,
      type      = "unshared",
      cluster   = seq_len(num_clust) - 1L,
      size      = latent_dim,
      stringsAsFactors = FALSE
    )
  }
  
  # Decoder ----------------------------------------------------------------
  layer_order_d <- unlist(reticulate::py_to_r(py_model$layer_order_dec))
  # decoder hidden dims reversed
  dec_sizes_rev <- rev(hidden_dims)
  
  dec_list <- lapply(seq_along(dec_sizes_rev), function(i) {
    data.frame(
      phase     = "decoder",
      layer_idx = i,
      type      = tolower(layer_order_d[i]),
      cluster   = if (tolower(layer_order_d[i]) == "shared") NA_integer_ else NA_integer_,
      size      = dec_sizes_rev[i],
      stringsAsFactors = FALSE
    )
  })
  dec_df <- do.call(rbind, dec_list)
  
  # Output -----------------------------------------------------------------
  output_shared <- as.logical(py_model$output_shared)
  input_dim     <- as.integer(py_model$input_dim)
  
  if (output_shared) {
    out_df <- data.frame(
      phase     = "output",
      layer_idx = 1L,
      type      = "shared",
      cluster   = NA_integer_,
      size      = input_dim,
      stringsAsFactors = FALSE
    )
  } else {
    num_clust <- as.integer(py_model$num_clusters)
    out_df <- data.frame(
      phase     = "output",
      layer_idx = 1L,
      type      = "unshared",
      cluster   = seq_len(num_clust) - 1L,
      size      = input_dim,
      stringsAsFactors = FALSE
    )
  }
  
  # Combine all
  architecture <- rbind(enc_df, lat_df, dec_df, out_df)
  architecture
}
