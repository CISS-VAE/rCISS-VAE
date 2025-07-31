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


## plot cissvae architecture -------------------------------------

library(ggplot2)

#' Plot a CISS‐VAE architecture as a horizontal schematic
#'
#' @param model_or_arch
#'   Either a reticulate‐imported Python CISSVAE object, or
#'   a data.frame from extract_cissvae_arch().
#' @param title        Optional plot title
#' @param color_shared   Fill for shared layers
#' @param color_unshared Fill for unshared layers
#' @param color_input    Fill override for the Input layer
#' @param color_latent   Fill override for the Latent layer(s)
#' @param color_output   Fill override for the Output layer
#' @param x_gap         Horizontal spacing
#' @param y_gap         Vertical spacing between clusters
#' @return A ggplot object
#' @export
plot_cissvae_arch <- function(model_or_arch,
                              title        = NULL,
                              color_shared   = "skyblue",
                              color_unshared = "lightcoral",
                              color_input    = "lightgreen",
                              color_latent   = "gold",
                              color_output   = "lightgreen",
                              x_gap         = 2,
                              y_gap         = 1.5) {

  # 1⃣  Accept either a Python model or a pre‐computed df
  if (inherits(model_or_arch, "python.builtin.object")) {
    # you must have extract_cissvae_arch() in your environment
    arch_df <- extract_cissvae_arch(model_or_arch)
  } else if (is.data.frame(model_or_arch)) {
    arch_df <- model_or_arch
  } else {
    stop("`model_or_arch` must be a Python CISSVAE object or an architecture data.frame.")
  }

  # ensure 'phase' factor ordering
  arch_df$phase <- factor(
    arch_df$phase,
    levels = c("encoder", "latent", "decoder", "output")
  )

  # how many clusters in total?
  n_clusters <- max(arch_df$cluster, na.rm = TRUE) + 1

  # 2⃣  Compute horizontal positions:
  #    find how many layers per phase, then offset each phase
  max_idx <- tapply(arch_df$layer_idx, arch_df$phase, max)
  # cumulative offsets
  offsets <- cumsum(c(0, max_idx[-length(max_idx)] + 1)) * x_gap
  names(offsets) <- levels(arch_df$phase)
  # assign x coordinate
  arch_df$x <- mapply(function(ph, idx) {
    offsets[as.character(ph)] + (idx - 1) * x_gap
  }, arch_df$phase, arch_df$layer_idx)

  # 3⃣  Vertical positions:
  # shared → y = 0; unshared → spread by cluster index
  arch_df$y <- ifelse(
    arch_df$type == "shared",
    0,
    (arch_df$cluster - (n_clusters - 1)/2) * y_gap
  )

  # 4⃣  Box sizes
  bw <- x_gap * 0.8
  bh_un <- y_gap * 0.8
  bh_sh <- if (n_clusters>1) y_gap*(n_clusters-1) + 0.5 else bh_un

  # 5⃣  Color assignment:
  arch_df$fill <- with(arch_df, ifelse(
    phase == "encoder" & layer_idx == 1, color_input,                # Input
    ifelse(phase == "output" & layer_idx == max(layer_idx[phase=="output"]), color_output,  # Output
           ifelse(phase == "latent", color_latent,                   # Latent
                  ifelse(type=="shared", color_shared, color_unshared)))
  ))

  # 6⃣  Prepare a rect + label data.frame
  rects <- transform(arch_df,
    xmin = x - bw/2,
    xmax = x + bw/2,
    ymin = ifelse(type=="shared", -bh_sh/2, y - bh_un/2),
    ymax = ifelse(type=="shared",  bh_sh/2, y + bh_un/2),
    label = ifelse(
      phase=="encoder",
      paste0("ENC ", layer_idx, ifelse(type=="unshared", paste0("\nC",cluster), ""), "\n", size),
      ifelse(phase=="decoder",
             paste0("DEC ", layer_idx, ifelse(type=="unshared", paste0("\nC",cluster), ""), "\n", size),
             ifelse(phase=="latent",
                    ifelse(type=="shared",
                           paste0("Latent\nμ/σ²\n", size),
                           paste0("Latent C",cluster,"\nμ/σ²\n", size)),
                    paste0("OUT", ifelse(type=="unshared", paste0(" C",cluster), ""), "\n", size)
             )
      )
    )
  )

  # 7⃣  Build arrows between successive x‐columns
  xs <- sort(unique(rects$x))
  arrows <- do.call(rbind, lapply(seq_along(xs)[-length(xs)], function(i) {
    curx <- xs[i]; nxtx <- xs[i+1]
    from <- subset(rects, x==curx)
    to   <- subset(rects, x==nxtx)
    # connect boxes if they share a cluster OR either is shared
    pairs <- merge(
      from, to,
      by = NULL,
      suffixes = c("_from", "_to")
    )
    keep <- with(pairs, type_from=="shared" | type_to=="shared" | cluster_from==cluster_to)
    pairs <- pairs[keep, ]
    data.frame(
      x    = pairs$x_from + bw/2,
      xend = pairs$x_to   - bw/2,
      y    = pairs$y_from,
      yend = pairs$y_to
    )
  }))

  # 8⃣  Plot
  p <- ggplot() +
    geom_rect(
      data = rects,
      aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, fill=fill),
      color="black", size=0.8
    ) +
    geom_text(
      data = rects,
      aes(x=(xmin+xmax)/2, y=(ymin+ymax)/2, label=label),
      size=3, fontface="bold", lineheight=0.9
    ) +
    geom_segment(
      data = arrows,
      aes(x=x, y=y, xend=xend, yend=yend),
      arrow = arrow(length = unit(0.15, "inches")), size=0.8
    ) +
    scale_fill_identity() +
    coord_equal() +
    theme_void() +
    theme(
      plot.title   = element_text(size=16, face="bold", hjust=0.5),
      plot.margin  = margin(10,10,10,10)
    )

  if (!is.null(title)) p <- p + ggtitle(title)
  return(p)
}
