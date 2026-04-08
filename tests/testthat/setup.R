Sys.setenv(TORCHINDUCTOR_CACHE_DIR = tempdir())

withr::defer(
  unlink(list.files(tempdir(), pattern = "torchinductor", full.names = TRUE),
         recursive = TRUE),
  teardown_env()
)