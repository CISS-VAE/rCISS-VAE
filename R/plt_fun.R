## Plot functions go here

#' Plot a beeswarm of silhouette scores
#'
#' @param silhouettes Numeric vector of silhouette scores.
#' @param sample_ids Optional character vector of sample identifiers. If omitted,
#'   samples will be numbered sequentially.
#' @return A \code{ggplot} object showing a beeswarm of silhouette scores.
#' @examples
#' \dontrun{
#'   sil <- run_cissvae(df, return_silhouettes = TRUE)$silhouettes
#'   plot_silhouette_beeswarm(sil)
#' }
#' @import ggplot2
#' @import ggbeeswarm
#' @export
plot_silhouette_beeswarm <- function(silhouettes, sample_ids = NULL) {
  if (!is.numeric(silhouettes)) {
    stop("`silhouettes` must be a numeric vector.")
  }
  if (is.null(sample_ids)) {
    sample_ids <- seq_along(silhouettes)
  }
  if (length(sample_ids) != length(silhouettes)) {
    stop("Length of `sample_ids` must match length of `silhouettes`.")
  }

  df <- data.frame(
    sample     = sample_ids,
    silhouette = silhouettes,
    stringsAsFactors = FALSE
  )

  ggplot2::ggplot(df, ggplot2::aes(x = 1, y = silhouette)) +
    ggbeeswarm::geom_beeswarm(cex = 1.2) +
    ggplot2::labs(
      x     = NULL,
      y     = "Silhouette score",
      title = "Beeswarm of silhouette scores"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text.x  = ggplot2::element_blank(),
      axis.ticks.x = ggplot2::element_blank()
    )
}

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

library(dplyr)
library(ggplot2)
library(grid)

library(ggplot2)
library(dplyr)

# Assumes extract_cissvae_arch() as defined earlier.

#' Plot CISSVAE architecture with ggplot2 to mirror the Python figure
#'
#' @param model A CISSVAE Python object via reticulate.
#' @param title Optional title.
#' @export
plot_cissvae_arch_gg <- function(model, title = NULL) {
  # Parameters same as Python
  x_gap       <- 1
  cluster_gap <- 1
  box_w       <- 0.8
  box_h       <- 0.5
  
  # Pull metadata
  arch <- extract_cissvae_arch(model)
  n_clusters <- as.integer(model$num_clusters)
  hidden_dims <- unlist(reticulate::py_to_r(model$hidden_dims))
  enc_order   <- unlist(reticulate::py_to_r(model$layer_order_enc))
  dec_order   <- unlist(reticulate::py_to_r(model$layer_order_dec))
  input_dim   <- as.integer(model$input_dim)
  
  # Build a data.frame of every layer box:
  rows <- list()
  x <- 1  # start at x=1
  
  # Input
  rows[[length(rows)+1]] <- tibble(
    phase   = "input",
    x       = x,
    y       = 0,
    label   = paste0("Input\n", input_dim),
    fill    = "lightgreen",
    track   = "shared"
  )
  
  # Encoder
  x <- x + x_gap
  shared_i <- 1L
  unshared_i <- 1L
  for (i in seq_along(enc_order)) {
    ty <- enc_order[i]
    dim <- hidden_dims[i]
    if (ty == "shared") {
      rows[[length(rows)+1]] <- tibble(
        phase = "encoder",
        x     = x,
        y     = 0,
        label = paste0("Enc ", i, "\n", dim),
        fill  = "skyblue",
        track = "shared"
      )
      shared_i <- shared_i + 1L
    } else {
      for (c in seq_len(n_clusters)-1L) {
        rows[[length(rows)+1]] <- tibble(
          phase = "encoder",
          x     = x,
          y     = (c - (n_clusters-1)/2)*cluster_gap,
          label = paste0("Enc ", i, "\nC", c, "\n", dim),
          fill  = "lightcoral",
          track = paste0("c", c)
        )
      }
      unshared_i <- unshared_i + 1L
    }
    x <- x + x_gap
  }
  encoder_start <- 2
  encoder_end   <- x - x_gap
  
  # Latent
  latent_shared <- isTRUE(model$latent_shared)
  latent_dim <- if (latent_shared) {
    as.integer(model$latent_dim)
  } else {
    hidden_dims[1] # any cluster yields same; we'll override label below
  }
  if (latent_shared) {
    rows[[length(rows)+1]] <- tibble(
      phase = "latent",
      x     = x,
      y     = 0,
      label = paste0("Latent\nμ/σ²\n", latent_dim),
      fill  = "gold",
      track = "shared"
    )
  } else {
    for (c in seq_len(n_clusters)-1L) {
      rows[[length(rows)+1]] <- tibble(
        phase = "latent",
        x     = x,
        y     = (c - (n_clusters-1)/2)*cluster_gap,
        label = paste0("Latent\nC", c, "\nμ/σ²\n", latent_dim),
        fill  = "gold",
        track = paste0("c", c)
      )
    }
  }
  x <- x + x_gap
  
  # Decoder
  decoder_start <- x
  for (i in seq_along(dec_order)) {
    ty  <- dec_order[i]
    dim <- hidden_dims[length(hidden_dims)-i+1]
    if (ty == "shared") {
      rows[[length(rows)+1]] <- tibble(
        phase = "decoder",
        x     = x,
        y     = 0,
        label = paste0("Dec ", i, "\n", dim),
        fill  = "skyblue",
        track = "shared"
      )
    } else {
      for (c in seq_len(n_clusters)-1L) {
        rows[[length(rows)+1]] <- tibble(
          phase = "decoder",
          x     = x,
          y     = (c - (n_clusters-1)/2)*cluster_gap,
          label = paste0("Dec ", i, "\nC", c, "\n", dim),
          fill  = "lightcoral",
          track = paste0("c", c)
        )
      }
    }
    x <- x + x_gap
  }
  decoder_end <- x - x_gap
  
  # Output
  out_shared <- isTRUE(model$output_shared)
  if (out_shared) {
    rows[[length(rows)+1]] <- tibble(
      phase = "output",
      x     = x,
      y     = 0,
      label = paste0("Output\n", input_dim),
      fill  = "lightgreen",
      track = "shared"
    )
  } else {
    for (c in seq_len(n_clusters)-1L) {
      rows[[length(rows)+1]] <- tibble(
        phase = "output",
        x     = x,
        y     = (c - (n_clusters-1)/2)*cluster_gap,
        label = paste0("Output\nC", c, "\n", input_dim),
        fill  = "lightgreen",
        track = paste0("c", c)
      )
    }
  }
  
  df <- bind_rows(rows)
  
  # Build arrows
  arrows <- df %>%
    arrange(x) %>%
    group_by(track) %>%
    mutate(xend = lead(x), yend = lead(y)) %>%
    filter(!is.na(xend)) %>%
    ungroup()
  
  # Section frames
  sections <- tribble(
    ~phase,    ~xmin,    ~xmax,
    "encoder", encoder_start - box_w, encoder_end + box_w,
    "decoder", decoder_start - box_w, decoder_end + box_w
  )
  
  # Plot
  p <- ggplot() +
    # section boxes
    geom_rect(data=sections,
              aes(xmin=xmin, xmax=xmax, ymin=-Inf, ymax=Inf),
              fill=NA, color="gray50", linetype="dashed") +
    # boxes
    geom_rect(data=df,
              aes(xmin=x-box_w/2, xmax=x+box_w/2,
                  ymin=y-box_h/2, ymax=y+box_h/2,
                  fill=fill),
              color="black") +
    geom_text(data=df,
              aes(x=x, y=y, label=label),
              size=3) +
    # arrows
    geom_segment(data=arrows,
                 aes(x=x+box_w/2, y=y,
                     xend=xend-box_w/2, yend=yend),
                 arrow=arrow(length=unit(0.15,"cm")), size=0.5) +
    # axis
    scale_fill_identity() +
    scale_x_continuous(breaks=c(1,
                                mean(c(encoder_start,encoder_end)),
                                x - x_gap - 1,  # latent center
                                mean(c(decoder_start,decoder_end)),
                                x),
                       labels=c("input","encoder","latent","decoder","output"),
                       expand=expansion(add=0.5)) +
    scale_y_continuous(expand=expansion(add=1)) +
    labs(x=NULL,y=NULL, title=title) +
    theme_minimal() +
    theme(panel.grid=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks=element_blank())
  
  print(p)
  invisible(p)
}
