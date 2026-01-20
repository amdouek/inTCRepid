# TCR-Epitope Prediction Model: Environment Setup
#
# USAGE:
#   1. Run at START of FRESH R session (Ctrl+Shift+F10 to restart)
#   2. Do NOT load reticulate before running
#
# SECTIONS:
#   A: First-time setup (run once)
#   B: Session initialization (run each session)
#   C: Verification
#   D: R package management
#
# V10 ADDITIONS:
#   - HuggingFace transformers for ESM-2 protein language model
#   - h5py for HDF5 embedding cache storage
#   - sentencepiece (transformers dependency)
#

# ===== Helper Functions =====

#' Get paths for conda environment
#' @param env_name Conda environment name
#' @return List with miniconda, env, python, and site_packages paths
get_env_paths <- function(env_name = "tcr_epitope") {

  miniconda <- file.path(Sys.getenv("LOCALAPPDATA"), "r-miniconda")
  env <- file.path(miniconda, "envs", env_name)
  list(
    miniconda = miniconda,
    env = env,
    python = file.path(env, "python.exe"),
    site_packages = file.path(env, "Lib", "site-packages")
  )
}

# ===== SECTION A: First-Time Setup =====

#' Check if first-time setup is needed
#' @param env_name Conda environment name
#' @param check_esm Also check for ESM-2 dependencies
#' @return TRUE if setup needed
check_setup_needed <- function(env_name = "tcr_epitope", check_esm = TRUE) {

  paths <- get_env_paths(env_name)

  if (!file.exists(paths$python)) {
    message("Setup needed: Environment '", env_name, "' not found")
    return(TRUE)
  }

  # Core packages
  required <- c("numpy", "pandas", "torch", "sklearn")
  for (pkg in required) {
    if (!dir.exists(file.path(paths$site_packages, pkg))) {
      message("Setup needed: Package '", pkg, "' missing")
      return(TRUE)
    }
  }

  # ESM-2 packages (V10)
  if (check_esm) {
    esm_required <- c("transformers", "h5py")
    for (pkg in esm_required) {
      if (!dir.exists(file.path(paths$site_packages, pkg))) {
        message("Setup needed: ESM-2 dependency '", pkg, "' missing")
        return(TRUE)
      }
    }
  }

  message("Environment '", env_name, "' configured")
  FALSE
}


#' First-time setup of conda environment
#'
#' Creates conda environment and installs required Python packages,
#' including ESM-2 dependencies for V10.
#'
#' Run ONCE, then restart R.
#'
#' @param env_name Conda environment name
#' @param python_version Python version
#' @param use_gpu Install CUDA-enabled PyTorch
#' @param force_reinstall Reinstall even if present
#' @param include_esm Install ESM-2 dependencies (default TRUE for V10+)
#' @return TRUE if successful
first_time_setup <- function(env_name = "tcr_epitope",
                             python_version = "3.10",
                             use_gpu = TRUE,
                             force_reinstall = FALSE,
                             include_esm = TRUE) {

  cat("\n", strrep("=", 60), "\n")
  cat("TCR-EPITOPE: FIRST-TIME SETUP\n")
  cat(strrep("=", 60), "\n\n")

  library(reticulate)
  paths <- get_env_paths(env_name)

  # Step 1: Miniconda
  if (!dir.exists(paths$miniconda)) {
    message("[1/4] Installing miniconda...")
    install_miniconda()
  } else {
    message("[1/4] Miniconda exists")
  }

  # Step 2: Conda environment
  if (!dir.exists(paths$env) || force_reinstall) {
    message("[2/4] Creating environment '", env_name, "'...")
    if (dir.exists(paths$env) && force_reinstall) {
      tryCatch(conda_remove(envname = env_name),
               error = function(e) unlink(paths$env, recursive = TRUE))
      Sys.sleep(2)
    }
    conda_create(envname = env_name, python_version = python_version)
  } else {
    message("[2/4] Environment exists")
  }

  # Step 3: Core Python packages
  message("[3/4] Installing core packages...")

  # Core packages via conda
  conda_install(env_name, c("numpy", "pandas", "scipy", "scikit-learn"),
                channel = "conda-forge")

  # PyTorch with appropriate CUDA version
  if (use_gpu) {
    message("  Installing PyTorch with CUDA support...")
    # CUDA 12.1 for RTX 2000 Ada and newer GPUs
    conda_install(env_name,
                  c("pytorch", "torchvision", "torchaudio", "pytorch-cuda=12.1"),
                  channel = c("pytorch", "nvidia"))
  } else {
    message("  Installing PyTorch (CPU only)...")
    conda_install(env_name,
                  c("pytorch", "torchvision", "torchaudio", "cpuonly"),
                  channel = "pytorch")
  }

  # Additional core packages via pip
  conda_install(env_name, "biopython", pip = TRUE)

  # Step 4: ESM-2 dependencies (V10)
  if (include_esm) {
    message("[4/4] Installing ESM-2 dependencies...")

    # HDF5 for embedding cache
    conda_install(env_name, "h5py", channel = "conda-forge")

    # HuggingFace transformers for ESM-2
    # Using pip for latest version with all ESM-2 support
    conda_install(env_name,
                  c("transformers", "sentencepiece", "protobuf"),
                  pip = TRUE)

    message("  ESM-2 dependencies installed")
  } else {
    message("[4/4] Skipping ESM-2 dependencies (include_esm=FALSE)")
  }

  cat("\n", strrep("=", 60), "\n")
  cat("SETUP COMPLETE - Restart R before continuing\n")
  cat("RStudio: Session -> Restart R (Ctrl+Shift+F10)\n")
  cat(strrep("=", 60), "\n\n")

  TRUE
}


#' Install ESM-2 dependencies only
#'
#' For existing environments that need ESM-2 support added.
#'
#' @param env_name Conda environment name
#' @return TRUE if successful
install_esm_dependencies <- function(env_name = "tcr_epitope") {

  cat("\n", strrep("=", 60), "\n")
  cat("INSTALLING ESM-2 DEPENDENCIES\n")
  cat(strrep("=", 60), "\n\n")

  library(reticulate)

  paths <- get_env_paths(env_name)
  if (!file.exists(paths$python)) {
    stop("Environment '", env_name, "' not found. Run first_time_setup() first.")
  }

  message("Installing to environment: ", env_name)

  # HDF5 for embedding cache
  message("\n[1/2] Installing h5py...")
  conda_install(env_name, "h5py", channel = "conda-forge")

  # HuggingFace transformers
  message("\n[2/2] Installing transformers...")
  conda_install(env_name,
                c("transformers", "sentencepiece", "protobuf"),
                pip = TRUE)

  cat("\n", strrep("=", 60), "\n")
  cat("ESM-2 DEPENDENCIES INSTALLED\n")
  cat("Restart R before continuing\n")
  cat(strrep("=", 60), "\n\n")

  TRUE
}

# ===== SECTION B: Session Initialization =====

#' Initialize Python environment for current session
#'
#' MUST be called at session start, BEFORE loading reticulate.
#'
#' @param env_name Conda environment name
#' @return TRUE if successful
initialize_session <- function(env_name = "tcr_epitope") {

  # Check for already-initialized Python
  if ("reticulate" %in% loadedNamespaces()) {
    if (reticulate::py_available(initialize = FALSE)) {
      stop("Python already initialized. Restart R and call initialize_session() first.")
    }
  }

  paths <- get_env_paths(env_name)
  python_path <- normalizePath(paths$python, winslash = "/", mustWork = FALSE)

  if (!file.exists(python_path)) {
    stop("Environment '", env_name, "' not found. Run first_time_setup().")
  }

  Sys.setenv(RETICULATE_PYTHON = python_path)
  library(reticulate)
  message("Session initialized: ", env_name)
  TRUE
}

## ===== Quick Start (copy/paste at session start) =====
Sys.setenv(RETICULATE_PYTHON = normalizePath(
  file.path(Sys.getenv("LOCALAPPDATA"),
            "r-miniconda/envs/tcr_epitope/python.exe"),
  winslash = "/"
))
library(reticulate)

# ===== SECTION C: Verification =====

#' Verify Python environment configuration
#' @param check_esm Also verify ESM-2 dependencies
#' @return List with environment status
verify_environment <- function(check_esm = TRUE) {

  cat("\n", strrep("=", 60), "\n")
  cat("ENVIRONMENT VERIFICATION\n")
  cat(strrep("=", 60), "\n")

  config <- py_config()
  cat("\nPython:", config$python, "\n")
  ver_str <- tryCatch({
    if (is.list(config$version)) {
      paste(config$version$major, config$version$minor, config$version$patch, sep = ".")
    } else {
      as.character(config$version)
    }
  }, error = function(e) "unknown")
  cat("Version:", ver_str, "\n")

  # Check environment
  env_ok <- grepl("tcr_epitope", config$python)
  cat("Environment:", ifelse(env_ok, "tcr_epitope ✓", "WARNING: wrong env"), "\n")

  # Check core packages
  cat("\n--- Core Packages ---\n")
  core_pkgs <- c(numpy = "numpy", pandas = "pandas", torch = "torch",
                 sklearn = "sklearn", scipy = "scipy")
  status <- list()

  for (nm in names(core_pkgs)) {
    avail <- py_module_available(core_pkgs[nm])
    status[[nm]] <- avail
    if (avail) {
      ver <- tryCatch({
        mod <- import(core_pkgs[nm])
        as.character(mod$`__version__`)
      }, error = function(e) "?")
      cat("  ", nm, ": ", ver, " ✓\n", sep = "")
    } else {
      cat("  ", nm, ": MISSING ✗\n", sep = "")
    }
  }

  # Check ESM-2 packages
  esm_status <- list()
  if (check_esm) {
    cat("\n--- ESM-2 Packages (V10) ---\n")
    esm_pkgs <- c(transformers = "transformers", h5py = "h5py")

    for (nm in names(esm_pkgs)) {
      avail <- py_module_available(esm_pkgs[nm])
      esm_status[[nm]] <- avail
      if (avail) {
        ver <- tryCatch({
          mod <- import(esm_pkgs[nm])
          as.character(mod$`__version__`)
        }, error = function(e) "?")
        cat("  ", nm, ": ", ver, " ✓\n", sep = "")
      } else {
        cat("  ", nm, ": MISSING ✗\n", sep = "")
      }
    }
  }

  # PyTorch device info
  if (status[["torch"]]) {
    torch <- import("torch")
    cuda <- torch$cuda$is_available()
    cat("\n--- Compute Device ---\n")
    if (cuda) {
      gpu_name <- torch$cuda$get_device_name(0L)
      gpu_mem <- as.numeric(torch$cuda$get_device_properties(0L)$total_memory) / 1024^3
      cat("  Device: GPU ✓\n")
      cat("  GPU: ", gpu_name, "\n", sep = "")
      cat("  VRAM: ", sprintf("%.1f GB", gpu_mem), "\n", sep = "")
    } else {
      cat("  Device: CPU (no CUDA)\n")
    }
  }

  # Summary
  all_core_ok <- all(unlist(status))
  all_esm_ok <- if (check_esm) all(unlist(esm_status)) else TRUE
  all_ok <- all_core_ok && all_esm_ok

  cat("\n", strrep("-", 40), "\n", sep = "")
  cat("Status: ", ifelse(all_ok, "READY ✓", "ISSUES FOUND ✗"), "\n", sep = "")

  if (!all_core_ok) {
    cat("  Missing core packages - run first_time_setup()\n")
  }
  if (check_esm && !all_esm_ok) {
    cat("  Missing ESM-2 packages - run install_esm_dependencies()\n")
  }

  cat(strrep("=", 60), "\n\n")

  invisible(list(
    python = config$python,
    version = config$version,
    core_packages = status,
    esm_packages = esm_status,
    cuda_available = if (status[["torch"]]) torch$cuda$is_available() else FALSE,
    ready = all_ok
  ))
}


#' Verify ESM-2 can load and run
#'
#' Quick functional test of ESM-2 model loading and inference.
#' Useful to confirm GPU memory is sufficient before full pipeline.
#'
#' @param model_name ESM-2 model to test
#' @param device 'cuda' or 'cpu'
#' @return TRUE if successful
verify_esm2_functional <- function(model_name = "facebook/esm2_t30_150M_UR50D",
                                   device = "cuda") {

  cat("\n", strrep("=", 60), "\n")
  cat("ESM-2 FUNCTIONAL TEST\n")
  cat(strrep("=", 60), "\n\n")

  # Check CUDA
  torch <- import("torch")
  if (device == "cuda" && !torch$cuda$is_available()) {
    message("CUDA not available, falling back to CPU")
    device <- "cpu"
  }

  # Memory before
  if (device == "cuda") {
    mem_before <- torch$cuda$memory_allocated() / 1024^3
    cat("GPU memory before: ", sprintf("%.2f GB", mem_before), "\n")
  }

  cat("Loading model: ", model_name, "\n")
  cat("Device: ", device, "\n\n")

  tryCatch({
    transformers <- import("transformers")

    # Load tokenizer
    cat("[1/4] Loading tokenizer...")
    tokenizer <- transformers$EsmTokenizer$from_pretrained(model_name)
    cat(" ✓\n")

    # Load model
    cat("[2/4] Loading model...")
    model <- transformers$EsmModel$from_pretrained(model_name)
    model <- model$to(device)
    model$eval()
    cat(" ✓\n")

    # Memory after loading
    if (device == "cuda") {
      mem_after <- torch$cuda$memory_allocated() / 1024^3
      cat("  GPU memory after loading: ", sprintf("%.2f GB", mem_after), "\n")
      cat("  Model size: ", sprintf("%.2f GB", mem_after - mem_before), "\n")
    }

    # Test inference
    cat("[3/4] Testing inference...")
    test_seqs <- c("CASSLAPGATNEKLFF", "GILGFVFTL")
    encoded <- tokenizer(test_seqs, return_tensors = "pt", padding = TRUE)
    input_ids <- encoded$input_ids$to(device)

    with(torch$no_grad(), {
      outputs <- model(input_ids = input_ids)
    })

    hidden_states <- outputs$last_hidden_state
    shape_list <- hidden_states$shape
    shape_vec <- c(
      as.integer(shape_list[0]),
      as.integer(shape_list[1]),
      as.integer(shape_list[2])
    )
    cat(" ✓\n")
    cat("  Output shape: (", paste(shape_vec, collapse = ", "), ")\n", sep = "")
    cat("  Embedding dim: ", shape_vec[3], "\n", sep = "")

    # Cleanup
    cat("[4/4] Cleanup...")
    rm(model, tokenizer, outputs)
    if (device == "cuda") {
      torch$cuda$empty_cache()
    }
    gc()
    cat(" ✓\n")

    if (device == "cuda") {
      mem_final <- torch$cuda$memory_allocated() / 1024^3
      cat("  GPU memory after cleanup: ", sprintf("%.2f GB", mem_final), "\n")
    }

    cat("\n", strrep("-", 40), "\n", sep = "")
    cat("ESM-2 FUNCTIONAL TEST: PASSED ✓\n")
    cat(strrep("=", 60), "\n\n")

    TRUE

  }, error = function(e) {
    cat(" ✗\n")
    cat("\nError: ", e$message, "\n")
    cat("\n", strrep("-", 40), "\n", sep = "")
    cat("ESM-2 FUNCTIONAL TEST: FAILED ✗\n")
    cat(strrep("=", 60), "\n\n")

    # Cleanup on failure
    if (device == "cuda") {
      torch$cuda$empty_cache()
    }
    gc()

    FALSE
  })
}


#' Import standard Python modules
#' @return List of imported modules
import_python_modules <- function() {

  message("Importing Python modules...")
  modules <- list(
    np = import("numpy", convert = TRUE),
    pd = import("pandas", convert = TRUE),
    torch = import("torch", convert = TRUE),
    sklearn = import("sklearn", convert = TRUE),
    scipy = import("scipy", convert = TRUE)
  )
  message("Done.")
  modules
}

# ===== SECTION D: R Package Management =====

#' Install and load required R packages
#' @param load_packages If TRUE, load packages after installing
#' @return Invisible NULL
setup_r_packages <- function(load_packages = TRUE) {

  required <- c(
    "tidyverse", "data.table", "reticulate", "caret", "pROC",
    "uwot", "ggrepel", "jsonlite", "httr", "Matrix", "pheatmap",
    "patchwork"
  )

  bioc_required <- c("Biostrings")

  # Install CRAN packages
  for (pkg in required) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
  }

  # Install Bioconductor packages
  for (pkg in bioc_required) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
      }
      BiocManager::install(pkg)
    }
  }

  # Load core packages
  if (load_packages) {
    suppressPackageStartupMessages({
      library(tidyverse)
      library(data.table)
      library(reticulate)
      library(caret)
      library(pROC)
      library(uwot)
      library(Matrix)
    })
    message("Core R packages loaded.")
  }

  invisible(NULL)
}

# ===== SECTION E: Utility Functions =====

#' Get GPU memory info
#' @return List with total, allocated, cached memory in GB
get_gpu_memory <- function() {

  torch <- import("torch")

  if (!torch$cuda$is_available()) {
    return(list(available = FALSE))
  }

  props <- torch$cuda$get_device_properties(0L)

  list(
    available = TRUE,
    device_name = torch$cuda$get_device_name(0L),
    total_gb = props$total_memory / 1024^3,
    allocated_gb = torch$cuda$memory_allocated() / 1024^3,
    cached_gb = torch$cuda$memory_reserved() / 1024^3,
    free_gb = (props$total_memory - torch$cuda$memory_allocated()) / 1024^3
  )
}


#' Print GPU memory status
#' @export
print_gpu_memory <- function() {

  mem <- get_gpu_memory()

  if (!mem$available) {
    cat("GPU: Not available\n")
    return(invisible(mem))
  }

  cat("\n--- GPU Memory Status ---\n")
  cat("Device: ", mem$device_name, "\n", sep = "")
  cat("Total:     ", sprintf("%6.2f GB\n", mem$total_gb))
  cat("Allocated: ", sprintf("%6.2f GB\n", mem$allocated_gb))
  cat("Cached:    ", sprintf("%6.2f GB\n", mem$cached_gb))
  cat("Free:      ", sprintf("%6.2f GB\n", mem$free_gb))
  cat("-------------------------\n\n")

  invisible(mem)
}


#' Clear GPU cache
#' @export
clear_gpu_cache <- function() {

  torch <- import("torch")

  if (torch$cuda$is_available()) {
    before <- torch$cuda$memory_allocated() / 1024^3
    torch$cuda$empty_cache()
    gc()
    after <- torch$cuda$memory_allocated() / 1024^3

    message(sprintf("GPU cache cleared: %.2f GB -> %.2f GB", before, after))
  } else {
    message("CUDA not available")
  }

  invisible(NULL)
}


# ===== WORKFLOW REFERENCE =====

# FIRST-TIME SETUP (V10):
#   1. Restart R (Ctrl+Shift+F10)
#   2. source("R/env_setup_config.R")
#   3. first_time_setup(use_gpu = TRUE, include_esm = TRUE)
#   4. Restart R
#   5. verify_environment(check_esm = TRUE)
#   6. verify_esm2_functional()  # Optional but recommended
#
# ADD ESM-2 TO EXISTING ENVIRONMENT:
#   1. Restart R
#   2. source("R/env_setup_config.R")
#   3. install_esm_dependencies()
#   4. Restart R
#   5. verify_environment(check_esm = TRUE)
#
# REGULAR SESSION:
#   1. Restart R (Ctrl+Shift+F10)
#   2. Run quick start block (Section B) OR:
#      source("R/env_setup_config.R")
#      initialize_session()
#   3. verify_environment()  # Optional
#   4. setup_r_packages()    # If not already loaded
#
# V10 PIPELINE:
#   1. Initialize session (above)
#   2. source("R/source_all.R")
#   3. verify_esm2_functional()  # Confirm ESM-2 works
#   4. source("R/run_v10_pipeline.R")
