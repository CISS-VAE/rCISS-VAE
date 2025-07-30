# plt_fun.R

library(ggplot2)
library(dplyr)
library(purrr)
library(grid)    # for arrow()

#' Plot CISS-VAE architecture schematic
#'
#' @param model_or_arch Python CISSVAE object (reticulate) or df from extract_cissvae_arch()
#' @param title         Optional plot title
#' @param color_shared   Fill for shared layers
#' @param color_unshared Fill for unshared layers
#' @param color_input    Fill override for Input
#' @param color_latent   Fill override for Latent
#' @param color_output   Fill override for Output
#' @param x_gap         Horizontal spacing
#' @param y_gap         Vertical spacing
#' @param box_w         Box width
#' @param box_h         Box height for unshared (shared auto-spans)
#' @export
plot_cissvae_arch <- function(model_or_arch,
                              title         = NULL,
                              color_shared   = "skyblue",
                              color_unshared = "lightcoral",
                              color_input    = "lightgreen",
                              color_latent   = "gold",
                              color_output   = "lightgreen",
                              x_gap          = 2,
                              y_gap          = 1.5,
                              box_w          = 0.8,
                              box_h          = 0.8) {
  # ─── 1) Get architecture table ──────────────────────────────────────────────
  if (inherits(model_or_arch, "python.builtin.object")) {
    arch_df  <- extract_cissvae_arch(model_or_arch)
    py_model <- model_or_arch
  } else {
    stop("Please pass a Python CISSVAE object, not a plain data.frame.")
  }
  n_clust   <- as.integer(py_model$num_clusters)
  input_dim <- as.integer(py_model$input_dim)

  # ─── 2) Build the Input row ────────────────────────────────────────────────
  input_row <- tibble(
    phase     = "input",
    layer_idx = 0L,
    type      = "shared",
    cluster   = NA_integer_,
    size      = input_dim
  )

  # ─── 3) Expand encoder rows per cluster ───────────────────────────────────
  enc_raw <- arch_df %>% filter(phase == "encoder")
  enc_expanded <- map_dfr(seq_len(nrow(enc_raw)), function(i) {
    r <- enc_raw[i, ]
    if (r$type == "shared") {
      r$cluster <- NA_integer_; r
    } else {
      map_dfr(0:(n_clust-1), function(cl) {
        r2 <- r; r2$cluster <- cl; r2
      })
    }
  })

  # ─── 4) Build latent rows ─────────────────────────────────────────────────
  lat_raw       <- arch_df %>% filter(phase == "latent")
  latent_dim    <- as.integer(lat_raw$size[1])
  latent_shared <- lat_raw$type[1] == "shared"
  lat_rows <- if (latent_shared) {
    tibble(phase="latent", layer_idx=1L, type="shared",
           cluster=NA_integer_, size=latent_dim)
  } else {
    tibble(phase="latent", layer_idx=1L, type="unshared",
           cluster=0:(n_clust-1), size=latent_dim)
  }

  # ─── 5) Expand decoder rows per cluster ───────────────────────────────────
  dec_raw <- arch_df %>% filter(phase == "decoder")
  dec_expanded <- map_dfr(seq_len(nrow(dec_raw)), function(i) {
    r <- dec_raw[i, ]
    if (r$type == "shared") {
      r$cluster <- NA_integer_; r
    } else {
      map_dfr(0:(n_clust-1), function(cl) {
        r2 <- r; r2$cluster <- cl; r2
      })
    }
  })

  # ─── 6) Build output rows ─────────────────────────────────────────────────
  out_raw       <- arch_df %>% filter(phase == "output") %>% slice(1)
  out_dim       <- as.integer(out_raw$size)
  output_shared <- out_raw$type == "shared"
  out_rows <- if (output_shared) {
    tibble(phase="output", layer_idx=1L, type="shared",
           cluster=NA_integer_, size=out_dim)
  } else {
    tibble(phase="output", layer_idx=1L, type="unshared",
           cluster=0:(n_clust-1), size=out_dim)
  }

  # ─── 7) Combine and ensure numeric layer_idx ───────────────────────────────
  all_df <- bind_rows(input_row,
                      enc_expanded,
                      lat_rows,
                      dec_expanded,
                      out_rows) %>%
    mutate(layer_idx = as.integer(layer_idx))

  # ─── 8) Compute x, y coordinates ──────────────────────────────────────────
  phase_order  <- c("input","encoder","latent","decoder","output")
  phase_counts <- all_df %>%
    group_by(phase) %>%
    summarise(max_idx = max(layer_idx, na.rm=TRUE)) %>%
    ungroup() %>%
    mutate(phase = factor(phase, levels = phase_order)) %>%
    arrange(phase)

  offsets <- setNames(
    cumsum(c(0, (phase_counts$max_idx[-nrow(phase_counts)] + 1) * x_gap)),
    phase_counts$phase
  )

  all_df <- all_df %>%
    mutate(
      x = offsets[phase] + (layer_idx - 1) * x_gap,
      y = ifelse(
        type=="shared", 0,
        (cluster - (n_clust - 1)/2) * y_gap
      )
    )

  # ─── 9) Arrow helpers ─────────────────────────────────────────────────────
  connect_phase <- function(df_from, df_to) {
    # for each row f in df_from, connect to matching rows in df_to
    map_dfr(seq_len(nrow(df_from)), function(i) {
      f <- df_from[i, ]
      to <- if (f$type == "shared") {
        df_to
      } else {
        df_to %>%
          filter(
            type=="shared" |
            (type=="unshared" & cluster==f$cluster)
          )
      }
      if (nrow(to)==0) return(NULL)
      data.frame(
        x    = rep(f$x + box_w/2, nrow(to)),
        y    = rep(f$y,        nrow(to)),
        xend = to$x - box_w/2,
        yend = to$y
      )
    })
  }

  connect_seq <- function(df_phase) {
    inds <- sort(unique(df_phase$layer_idx))
    # skip last index
    map_dfr(inds[-length(inds)], function(i) {
      connect_phase(
        df_phase %>% filter(layer_idx == i),
        df_phase %>% filter(layer_idx == i+1)
      )
    })
  }

  # ─── 1️⃣0️⃣ Build every arrow segment ───────────────────────────────────────
  df_in  <- all_df %>% filter(phase == "input")
  df_enc <- all_df %>% filter(phase == "encoder")
  df_lat <- all_df %>% filter(phase == "latent")
  df_dec <- all_df %>% filter(phase == "decoder")
  df_out <- all_df %>% filter(phase == "output")

  arrows <- bind_rows(
    connect_phase(df_in,  df_enc   %>% filter(layer_idx==1)),
    connect_seq(df_enc),
    connect_phase(df_enc   %>% filter(layer_idx==max(layer_idx)), df_lat),
    connect_phase(df_lat, df_dec   %>% filter(layer_idx==1)),
    connect_seq(df_dec),
    connect_phase(df_dec   %>% filter(layer_idx==max(layer_idx)), df_out)
  )

  # ─── 1️⃣1️⃣ Plot ──────────────────────────────────────────────────────────
  p <- ggplot() +
    # dashed outlines for encoder & decoder
    lapply(c("encoder","decoder"), function(ph) {
      d <- all_df %>% filter(phase==ph)
      annotate("rect",
               xmin = min(d$x)-box_w/2-0.2,
               xmax = max(d$x)+box_w/2+0.2,
               ymin = min(d$y)-box_h/2-0.2,
               ymax = max(d$y)+box_h/2+0.2,
               fill     = NA,
               color    = "gray50",
               linetype = "dashed",
               size     = 0.8)
    }) +
    geom_rect(
      data = all_df,
      aes(xmin = x - box_w/2, xmax = x + box_w/2,
          ymin = y - box_h/2, ymax = y + box_h/2,
          fill = type),
      color = "black", size = 0.8
    ) +
    geom_text(
      data = all_df,
      aes(x=x, y=y,
          label = case_when(
            phase=="input"   ~ paste0("Input\n", size),
            phase=="encoder" ~ paste0("Enc ", layer_idx,
                                     ifelse(type=="unshared",
                                            paste0("\nC",cluster), ""),
                                     "\n", size),
            phase=="latent"  ~ ifelse(type=="shared",
                                     paste0("Latent\nμ/σ²\n", size),
                                     paste0("Latent C",cluster,"\nμ/σ²\n", size)),
            phase=="decoder" ~ paste0("Dec ", layer_idx,
                                     ifelse(type=="unshared",
                                            paste0("\nC",cluster), ""),
                                     "\n", size),
            TRUE             ~ ifelse(type=="shared",
                                     paste0("Output\n", size),
                                     paste0("Output C",cluster, "\n", size))
          )
      ),
      size=3, fontface="bold", lineheight=0.9
    ) +
    geom_segment(
      data        = arrows,
      aes(x=x, y=y, xend=xend, yend=yend),
      arrow       = arrow(length = unit(0.15, "inches")),
      size        = 0.8,
      inherit.aes = FALSE
    ) +
    scale_fill_manual(
      values = c(shared=color_shared, unshared=color_unshared)
    ) +
    coord_equal() +
    theme_void() +
    theme(legend.position="none")

  if (!is.null(title)) p <- p + ggtitle(title)
  p
}

