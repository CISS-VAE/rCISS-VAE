library(ggplot2)
library(dplyr)
library(purrr)
library(grid)  # for arrow()

#' Plot CISS-VAE architecture schematic
#'
#' @param model_or_arch Python CISSVAE object (reticulate)
#' @param title         Optional title
#' @param color_shared   Fill for shared layers
#' @param color_unshared Fill for unshared layers
#' @param color_input    Fill override for Input
#' @param color_latent   Fill override for Latent
#' @param color_output   Fill override for Output
#' @param layer_gap     Horizontal gap between layers in the same phase
#' @param phase_gap     Horizontal gap between phases
#' @param y_gap         Vertical gap between cluster rows
#' @param box_w         Box width
#' @param box_h         Box height for unshared layers
#' @export
plot_cissvae_arch <- function(model_or_arch,
                              title         = NULL,
                              color_shared   = "skyblue",
                              color_unshared = "lightcoral",
                              color_input    = "lightgreen",
                              color_latent   = "gold",
                              color_output   = "lightgreen",
                              layer_gap      = 2.0,
                              phase_gap      = 0.5,
                              y_gap          = 2.0,
                              box_w          = 1.2,
                              box_h          = 1.0) {
  # 1️⃣ Extract architecture
  if (!inherits(model_or_arch, "python.builtin.object")) {
    stop("Please supply a reticulate‐imported CISSVAE object.")
  }
  arch_df  <- extract_cissvae_arch(model_or_arch)
  py_model <- model_or_arch
  n_clust   <- as.integer(py_model$num_clusters)
  input_dim <- as.integer(py_model$input_dim)

  # 2️⃣ Build Input row
  input_row <- tibble(
    phase     = "input",   layer_idx = 0L,
    type      = "shared",  cluster   = NA_integer_,
    size      = input_dim
  )

  # 3️⃣ Expand encoder per cluster
  enc_raw <- filter(arch_df, phase == "encoder")
  enc_expanded <- map_dfr(seq_len(nrow(enc_raw)), function(i) {
    r <- enc_raw[i, ]
    if (r$type == "shared") {
      r$cluster <- NA_integer_; r
    } else {
      map_dfr(0:(n_clust - 1), function(cl) {
        r2 <- r; r2$cluster <- cl; r2
      })
    }
  })

  # 4️⃣ Latent rows
  lat_raw       <- filter(arch_df, phase == "latent")
  latent_dim    <- lat_raw$size[1]
  latent_shared <- lat_raw$type[1] == "shared"
  lat_rows <- if (latent_shared) {
    tibble(phase="latent", layer_idx=1L, type="shared",
           cluster=NA_integer_, size=latent_dim)
  } else {
    tibble(phase="latent", layer_idx=1L, type="unshared",
           cluster=0:(n_clust-1), size=latent_dim)
  }

  # 5️⃣ Expand decoder per cluster
  dec_raw <- filter(arch_df, phase == "decoder")
  dec_expanded <- map_dfr(seq_len(nrow(dec_raw)), function(i) {
    r <- dec_raw[i, ]
    if (r$type == "shared") {
      r$cluster <- NA_integer_; r
    } else {
      map_dfr(0:(n_clust - 1), function(cl) {
        r2 <- r; r2$cluster <- cl; r2
      })
    }
  })

  # 6️⃣ Output rows
  out_raw       <- filter(arch_df, phase == "output") %>% slice(1)
  output_shared <- out_raw$type == "shared"
  out_dim       <- out_raw$size
  out_rows <- if (output_shared) {
    tibble(phase="output", layer_idx=1L, type="shared",
           cluster=NA_integer_, size=out_dim)
  } else {
    tibble(phase="output", layer_idx=1L, type="unshared",
           cluster=0:(n_clust-1), size=out_dim)
  }

  # 7️⃣ Combine & ensure numeric
  all_df <- bind_rows(input_row,
                      enc_expanded,
                      lat_rows,
                      dec_expanded,
                      out_rows) %>%
    mutate(layer_idx = as.integer(layer_idx))

  # 8️⃣ Compute x/y positions
  phase_order <- c("input","encoder","latent","decoder","output")
  # Count layers per phase
  phase_counts <- all_df %>%
    group_by(phase) %>%
    summarise(max_idx = max(layer_idx)) %>%
    ungroup() %>%
    mutate(phase = factor(phase, levels=phase_order)) %>%
    arrange(phase)

  # Build offsets sequentially
  offsets <- numeric(length(phase_order))
  names(offsets) <- phase_order
  offsets["input"] <- 0
  for (i in seq(2, length(phase_order))) {
    prev <- phase_order[i - 1]
    # (prev_max - 1)*layer_gap is total width of phase, plus phase_gap
    offsets[i] <- offsets[i - 1] +
      (phase_counts$max_idx[phase_counts$phase == prev] - 1) * layer_gap +
      phase_gap
  }

  # Shared box full height
  shared_h <- box_h + (n_clust - 1) * y_gap

  all_df <- all_df %>%
    mutate(
      x     = offsets[phase] + (layer_idx - 1) * layer_gap,
      y     = ifelse(type=="shared", 0,
                     (cluster - (n_clust - 1)/2) * y_gap),
      xmin  = x - box_w/2,
      xmax  = x + box_w/2,
      ymin  = ifelse(type=="shared", -shared_h/2,  y - box_h/2),
      ymax  = ifelse(type=="shared",  shared_h/2,  y + box_h/2)
    )

  # 9️⃣ Arrow helper
  connect_phase <- function(df_from, df_to) {
    map_dfr(seq_len(nrow(df_from)), function(i) {
      f <- df_from[i, ]
      sel <- if (f$type == "shared") {
        df_to
      } else {
        filter(df_to, type=="shared" |
                          (type=="unshared" & cluster==f$cluster))
      }
      tibble(
        x    = rep(f$x + box_w/2, nrow(sel)),
        y    = rep(f$y,        nrow(sel)),
        xend = sel$x - box_w/2,
        yend = sel$y
      )
    })
  }

  connect_seq <- function(df_phase) {
    idxs <- sort(unique(df_phase$layer_idx))
    map_dfr(idxs[-length(idxs)], function(i) {
      connect_phase(
        filter(df_phase, layer_idx == i),
        filter(df_phase, layer_idx == i + 1)
      )
    })
  }

  # 1️⃣0️⃣ Build all arrows
  df_in  <- filter(all_df, phase=="input")
  df_enc <- filter(all_df, phase=="encoder")
  df_lat <- filter(all_df, phase=="latent")
  df_dec <- filter(all_df, phase=="decoder")
  df_out <- filter(all_df, phase=="output")

  arrows <- bind_rows(
    connect_phase(df_in,  filter(df_enc,  layer_idx==1)),
    connect_seq(df_enc),
    connect_phase(filter(df_enc, layer_idx==max(layer_idx)), df_lat),
    connect_phase(df_lat, filter(df_dec, layer_idx==1)),
    connect_seq(df_dec),
    connect_phase(filter(df_dec, layer_idx==max(layer_idx)), df_out)
  )

  # 1️⃣1️⃣ Draw
  # compute one dashed region around encoder+decoder
  encdec <- filter(all_df, phase %in% c("encoder","decoder"))
  region <- list(
    xmin = min(encdec$xmin) - 0.2,
    xmax = max(encdec$xmax) + 0.2,
    ymin = min(encdec$ymin) - 0.2,
    ymax = max(encdec$ymax) + 0.2
  )

  p <- ggplot() +
    # single dashed box
    annotate("rect",
      xmin     = region$xmin, xmax = region$xmax,
      ymin     = region$ymin, ymax = region$ymax,
      fill     = NA,
      color    = "gray50",
      linetype = "dashed",
      size     = 0.8
    ) +
    # all boxes
    geom_rect(
      data = all_df,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = type),
      color = "black", size = 0.8
    ) +
    # labels
    geom_text(
      data = all_df,
      aes(x = x, y = y,
          label = case_when(
            phase == "input"   ~ paste0("Input\n", size),
            phase == "encoder" ~ paste0("Enc ", layer_idx,
                                        ifelse(type=="unshared",
                                               paste0("\nC",cluster), ""),
                                        "\n", size),
            phase == "latent"  ~ ifelse(type=="shared",
                                        paste0("Latent\nμ/σ²\n", size),
                                        paste0("Latent C",cluster,"\nμ/σ²\n", size)),
            phase == "decoder" ~ paste0("Dec ", layer_idx,
                                        ifelse(type=="unshared",
                                               paste0("\nC",cluster), ""),
                                        "\n", size),
            TRUE               ~ ifelse(type=="shared",
                                        paste0("Output\n", size),
                                        paste0("Output C",cluster, "\n", size))
          )
      ),
      size=3, fontface="bold", lineheight=0.9
    ) +
    # arrows
    geom_segment(
      data        = arrows,
      aes(x=x, y=y, xend=xend, yend=yend),
      arrow       = arrow(length = unit(0.15, "inches")),
      size        = 0.8,
      inherit.aes = FALSE
    ) +
    # fills
    scale_fill_manual(values = c(shared=color_shared,
                                 unshared=color_unshared)) +
    coord_equal() +
    theme_void() +
    theme(legend.position="none")

  if (!is.null(title)) p <- p + ggtitle(title)
  return(p)
}

