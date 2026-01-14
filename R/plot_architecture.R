#' Plot VAE Architecture Diagram
#'
#' Creates a horizontal schematic diagram of the CISS-VAE architecture, showing 
#' shared and cluster-specific layers. This function wraps the Python 
#' `plot_vae_architecture` function from the ciss_vae package.
#'
#' @param model A trained CISSVAE model object (Python object)
#' @param title Title of the plot. If NULL, no title is displayed. Default NULL.
#' @param color_shared Color for shared hidden layers. Default "skyblue".
#' @param color_unshared Color for unshared (cluster-specific) hidden layers. Default "lightcoral".
#' @param color_latent Color for latent layer. Default "gold".
#' @param color_input Color for input layer. Default "lightgreen".
#' @param color_output Color for output layer. Default "lightgreen".
#' @param figsize Size of the matplotlib figure as c(width, height). Default c(16, 8).
#' @param save_path Optional path to save the plot as PNG. If NULL, plot is displayed. Default NULL.
#' @param dpi Resolution for saved PNG file. Default 300.
#' @param return_plot Logical; if TRUE, returns the plot as an R object using reticulate. Default FALSE.
#' @param display_plot Logical; if TRUE, displays the plot. Set to FALSE when only saving. Default TRUE.
#'
#' @return If return_plot is TRUE, returns a Python matplotlib figure object that can be 
#'   further manipulated. Otherwise returns NULL invisibly.
#' 
#' 
#' @section Tips:
#' \itemize{
#'   \item If you get a TCL or TK error, run: `reticulate::py_run_string("import matplotlib; matplotlib.use('Agg')")` to change the matplotlib backend to use 'Agg' instead.
#' }
#'
#'
#' @examples
#' \donttest{
#' try({
#'   # Train a model first
#'   result <- run_cissvae(my_data, return_model = TRUE)
#'
#'   # Basic plot
#'   plot_vae_architecture(result$model)
#'
#'   # Save plot to file
#'   plot_vae_architecture(
#'     model = result$model,
#'     title = "CISS-VAE Architecture",
#'     save_path = "vae_architecture.png",
#'     dpi = 300
#'   )
#'
#'   # Return plot object for further manipulation
#'   fig <- plot_vae_architecture(
#'     model = result$model,
#'     return_plot = TRUE,
#'     display_plot = FALSE
#'   )
#' })
#' }
#' @export
plot_vae_architecture <- function(model,
  title = NULL,
  color_shared = "skyblue",
  color_unshared = "lightcoral", 
  color_latent = "gold",
  color_input = "lightgreen",
  color_output = "lightgreen",
  figsize = c(16, 8),
  save_path = NULL,
  dpi = 300,
  return_plot = FALSE,
  display_plot = TRUE) {
  

# Check if model is provided
if (missing(model) || is.null(model)) {
stop("model parameter is required")
}

# Validate figsize
if (length(figsize) != 2 || !is.numeric(figsize)) {
stop("figsize must be a numeric vector of length 2")
}

# Import the plotting function from the package
tryCatch({
# Import the module that contains plot_vae_architecture
plot_mod <- reticulate::import("ciss_vae.utils.helpers", convert = FALSE)
plot_func <- plot_mod$plot_vae_architecture
}, error = function(e3) {
stop("Failed to import plot_vae_architecture function. Make sure ciss_vae package is installed and the function is accessible. Tried multiple import paths.")
})


# Convert figsize to Python tuple
figsize_py <- reticulate::tuple(figsize[1], figsize[2])

# Call the Python function with return_fig=True to get the figure object
tryCatch({
fig <- plot_func(
model = model,
title = title,
color_shared = color_shared,
color_unshared = color_unshared,
color_latent = color_latent,
color_input = color_input,
color_output = color_output,
figsize = figsize_py,
return_fig = TRUE  # Always return fig so we can save/display as needed
)
}, error = function(e) {
stop("Failed to create plot. Error: ", e$message)
})

# Handle saving
if (!is.null(save_path)) {
tryCatch({
# Ensure the directory exists
dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)

# Save the figure
fig$savefig(save_path, dpi = as.integer(dpi), bbox_inches = "tight")

message("Plot saved to: ", save_path)
}, error = function(e) {
warning("Failed to save plot to ", save_path, ". Error: ", e$message)
})
}

# Handle display
if (display_plot) {
tryCatch({
# Import matplotlib.pyplot and show the plot
plt <- reticulate::import("matplotlib.pyplot", convert = FALSE)
plt$tight_layout()
plt$show()
}, error = function(e) {
warning("Failed to display plot. Error: ", e$message)
})
}

# Return figure object if requested
if (return_plot) {
return(fig)
} else {
# Close the figure to free memory if not returning it
tryCatch({
plt <- reticulate::import("matplotlib.pyplot", convert = FALSE)
plt$close(fig)
}, error = function(e) {
# Ignore closing errors
})
return(invisible(NULL))
}
}