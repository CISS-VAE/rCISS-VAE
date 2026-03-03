# rCISSVAE v0.0.5
## Updates
- `performance_by_cluster()` imputation error calculation now matches python implementation
- changed all `columns_ignore` parameters to `cols_ignore`. Keeping `columns_ignore` as an alias for continuity.

## New additions
- `save_cissvae_model()` will save CISSVAE models to disk
- `load_cissvae_model()` loads a saved CISSVAE model from disk
- `impute_with_cissvae()` accepts a model and R data.frame with missingness and uses the model to impute the data



# rCISSVAE v0.0.4
## Updates
- added missingness heatmap function `cluster_heatmap()` and associated vignette
- Updated the examples to be 'donttest{}' not 'dontrun{}'
- Added checks for reticulate to functions requiring reticulate
- Added tutorial for loading and imputing with saved model

# rCISSVAE v0.0.3
## Updates
- `binary_feature_mask` is now correctly recognized by both `autotune_cissvae()` and `run_cissvae()`
- Added tests for making sure binary_feature_mask is correctly recognized

# rCISSVAE v0.0.2.0000
### Updating for CRAN readiness
## Updates
- Improved Vignettes  
    - Added extdata files for sample results
- Added documentation for undcumented things

# rCISSVAE v0.0.1.0000
## Initial Package Version
## Updates
- gtsummary-like cluster_summary function