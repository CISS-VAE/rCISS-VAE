# rCISS-VAE - WORK IN PROGRESS!

The R implementation of the Clustering-Informed Shared-Structure Variational Autoencoder (CISS-VAE). The Python implementation can be found [here](https://github.com/CISS-VAE/CISS-VAE-python).

## Installation 


Install devtools or remotes if not already installed: 

```{.r}
install.packages("remotes")
# or
install.packages("devtools")

```

The rCISSVAE package can be installed with:

```{.r}
remotes::install_github("CISS-VAE/rCISS-VAE")
# or
devtools::install_github("CISS-VAE/rCISS-VAE")

```

## Ensuring correct virtual environment for reticulate

This package uses `reticulate` to interface with the python version of the package `cissvae`. 

Therefore, it is necessary to make sure that you have a venv or conda environment set up that has the `cissvae` package installed. 

If you are comfortable creating an environment and installing the package, great! Then all you need to do is tell reticulate where to point. 

**For Venv**
```{.r}
reticulate::use_virtualenv("./.venv", required = TRUE)

```

**For conda**

```{.r}
reticulate::use_condaenv("myenv", required = TRUE)
```

### Virtual environment helper function 

If you do not want to manually create the virtual environment, you can use the helper function `create_cissvae_env()` to create a virtual environment (venv) in your current working directory. 

```{.r}
create_cissvae_env(
  envname = "cissvae_environment", ## name of environment
  path = NULL, ## add path to wherever you want virtual environment to be
  install_python = FALSE, ## set to TRUE if you want create_cisssvae_env to install python for you
  python_version = "3.10" ## set to whatever version you want >=3.10. Python 3.10 or 3.11 recommended
)
```

Once the environment is created, activate it using:

```{.r}
reticulate::use_virtualenv("./cissvae_environment", required = TRUE)

# In other words,
# reticulate::use_virtualenv("./your_environment_name", required = TRUE)


```


