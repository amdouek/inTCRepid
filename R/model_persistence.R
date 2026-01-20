# Model persistence: save/load trained models and inference packages (V7)

library(reticulate)

# ===== Core Save/Load Functions =====

#' Save trained V7 model to disk
#'
#' @param model PyTorch model object
#' @param trainer TCRTrainerV7 object
#' @param model_dir Directory to save model files
#' @param model_name Base name for model files
#' @param history Training history (optional)
#' @param config Training configuration (optional)
#' @return List of saved file paths
#' @export
save_model <- function(model, trainer = NULL,
                       model_dir = "models",
                       model_name = "tcr_epitope_v7",
                       history = NULL,
                       config = NULL) {

  if (!dir.exists(model_dir)) dir.create(model_dir, recursive = TRUE)

  weights_path <- file.path(model_dir, paste0(model_name, "_weights.pt"))
  config_path <- file.path(model_dir, paste0(model_name, "_config.rds"))

  message("Saving model to: ", model_dir)

  # Save PyTorch weights

  torch <- import("torch")
  torch$save(model$state_dict(), weights_path)
  message("  Weights: ", basename(weights_path))

  # Extract model architecture info
  model_config <- list(
    version = "v7",
    model_class = class(model)[1],
    output_dim = as.integer(model$output_dim),
    timestamp = Sys.time(),
    r_version = R.version.string,
    torch_version = as.character(torch$`__version__`)
  )

  # Add training config if provided
  if (!is.null(config)) {
    model_config$training_config <- config
  }

  saveRDS(model_config, config_path)
  message("  Config: ", basename(config_path))

  saved <- list(weights = weights_path, config = config_path)

  # Save history if provided
  if (!is.null(history)) {
    history_path <- file.path(model_dir, paste0(model_name, "_history.rds"))
    history_r <- lapply(history, unlist)
    saveRDS(history_r, history_path)
    message("  History: ", basename(history_path))
    saved$history <- history_path
  }

  message("Model saved successfully.")
  invisible(saved)
}

#' Load trained V7 model from disk
#'
#' @param model_dir Directory containing model files
#' @param model_name Base name of model files
#' @param trb_vocab TRB V/J vocabulary
#' @param tra_vocab TRA V/J vocabulary
#' @param device Device to load model on
#' @return List with model, config, history
#' @export
load_model <- function(model_dir = "models",
                       model_name = "tcr_epitope_v7",
                       trb_vocab = NULL,
                       tra_vocab = NULL,
                       device = "cpu") {

  weights_path <- file.path(model_dir, paste0(model_name, "_weights.pt"))
  config_path <- file.path(model_dir, paste0(model_name, "_config.rds"))

  if (!file.exists(weights_path)) stop("Weights not found: ", weights_path)
  if (!file.exists(config_path)) stop("Config not found: ", config_path)

  message("Loading model from: ", model_dir)

  config <- readRDS(config_path)
  message("  Version: ", config$version)
  message("  Saved: ", config$timestamp)

  # Recreate model architecture
  tc <- config$training_config

  if (is.null(trb_vocab) || is.null(tra_vocab)) {
    # Try to load from package
    vocab_path <- file.path(model_dir, paste0(model_name, "_vocabs.rds"))
    if (file.exists(vocab_path)) {
      vocabs <- readRDS(vocab_path)
      trb_vocab <- vocabs$trb
      tra_vocab <- vocabs$tra
      message("  Loaded vocabularies from package")
    } else {
      stop("V/J vocabularies required. Provide trb_vocab and tra_vocab or include in package.")
    }
  }

  model <- create_tcr_epitope_model(
    vocab_size = tc$vocab_size %||% 22L,
    embed_dim = tc$token_embedding_dim %||% 128L,
    hidden_dim = tc$hidden_dim %||% 256L,
    output_dim = tc$output_dim %||% 256L,
    dropout = tc$dropout %||% 0.3,
    trb_v_vocab_size = as.integer(trb_vocab$v$size),
    trb_j_vocab_size = as.integer(trb_vocab$j$size),
    tra_v_vocab_size = as.integer(tra_vocab$v$size),
    tra_j_vocab_size = as.integer(tra_vocab$j$size),
    v_embed_dim = tc$v_embed_dim %||% 32L,
    j_embed_dim = tc$j_embed_dim %||% 16L,
    fusion = tc$fusion_type %||% "concat",
    use_atchley_init = FALSE,  # Don't reinit when loading
    use_blosum_reg = tc$use_blosum_reg %||% FALSE
  )

  # Load weights
  torch <- import("torch")
  state_dict <- torch$load(weights_path, map_location = device)
  model$load_state_dict(state_dict)
  model$eval()

  message("  Weights loaded")

  # Load history if available
  history_path <- file.path(model_dir, paste0(model_name, "_history.rds"))
  history <- if (file.exists(history_path)) readRDS(history_path) else NULL

  message("Model loaded successfully.")

  list(model = model, config = config, history = history,
       trb_vocab = trb_vocab, tra_vocab = tra_vocab)
}

# ===== Epitope Reference =====

#' Save epitope reference database
#'
#' @param epitope_reference Reference from build_epitope_reference()
#' @param output_path Output file path
#' @return Path to saved file
#' @export
save_epitope_reference <- function(epitope_reference, output_path) {

  dir_path <- dirname(output_path)
  if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE)

  message("Saving epitope reference...")
  message("  Epitopes: ", epitope_reference$n_epitopes)
  message("  Embedding dim: ", epitope_reference$embedding_dim)

  saveRDS(epitope_reference, output_path)

  size_mb <- round(file.size(output_path) / 1024^2, 2)
  message("  Saved: ", output_path, " (", size_mb, " MB)")

  invisible(output_path)
}

#' Load epitope reference database
#'
#' @param input_path Path to reference file
#' @return Epitope reference list
#' @export
load_epitope_reference <- function(input_path) {

  if (!file.exists(input_path)) stop("Reference not found: ", input_path)

  ref <- readRDS(input_path)
  message("Loaded epitope reference: ", ref$n_epitopes, " epitopes")

  ref
}

# ============================================================================
# Complete Model Package (V7)
# ============================================================================

#' Save complete V7 model package for deployment
#'
#' Saves everything needed for inference:
#' - Model weights and architecture config
#' - V/J vocabularies (TRA and TRB)
#' - Epitope reference database
#' - Training metadata and evaluation metrics
#'
#' @param pipeline_result Result from run_transfer_learning_pipeline()
#' @param package_dir Base directory for packages
#' @param package_name Name for this package
#' @return List of saved file paths
#' @export
save_model_package <- function(pipeline_result,
                               package_dir = "models",
                               package_name = "tcr_epitope_v7") {

  pkg_path <- file.path(package_dir, package_name)
  if (!dir.exists(pkg_path)) dir.create(pkg_path, recursive = TRUE)

  message("\n", strrep("=", 60))
  message("Saving Model Package: ", package_name)
  message(strrep("=", 60))

  saved <- list()

  # 1. Model weights and config
  message("\n[1/5] Saving model...")
  model_files <- save_model(
    model = pipeline_result$phase1$model,
    model_dir = pkg_path,
    model_name = "model",
    history = pipeline_result$phase1$history,
    config = pipeline_result$config
  )
  saved$model <- model_files

  # 2. V/J vocabularies
  message("\n[2/5] Saving V/J vocabularies...")
  vocabs <- list(
    trb = pipeline_result$data_splits$trb_vocab,
    tra = pipeline_result$data_splits$tra_vocab
  )
  vocab_path <- file.path(pkg_path, "model_vocabs.rds")
  saveRDS(vocabs, vocab_path)
  message("  Saved: model_vocabs.rds")
  message("    TRB-V: ", vocabs$trb$v$size, " genes")
  message("    TRB-J: ", vocabs$trb$j$size, " genes")
  message("    TRA-V: ", vocabs$tra$v$size, " genes")
  message("    TRA-J: ", vocabs$tra$j$size, " genes")
  saved$vocabs <- vocab_path

  # 3. Epitope reference
  message("\n[3/5] Building epitope reference...")
  epi_ref <- build_epitope_reference(
    trainer = pipeline_result$phase1$trainer,
    unique_epitopes = pipeline_result$data_splits$unique_epitopes,
    unique_epitope_idx = pipeline_result$data_splits$unique_epitope_idx
  )
  ref_path <- file.path(pkg_path, "epitope_reference.rds")
  save_epitope_reference(epi_ref, ref_path)
  saved$epitope_reference <- ref_path

  # 4. Epitope mappings
  message("\n[4/5] Saving epitope mappings...")
  mappings <- list(
    epitope_to_idx = pipeline_result$data_splits$epitope_to_idx,
    idx_to_epitope = pipeline_result$data_splits$idx_to_epitope,
    unique_epitopes = pipeline_result$data_splits$unique_epitopes,
    n_epitopes = length(pipeline_result$data_splits$unique_epitopes)
  )
  mappings_path <- file.path(pkg_path, "epitope_mappings.rds")
  saveRDS(mappings, mappings_path)
  message("  Saved: epitope_mappings.rds")
  saved$mappings <- mappings_path

  # 5. Package manifest
  message("\n[5/5] Saving manifest...")

  # Get evaluation metrics
  eval_metrics <- if (!is.null(pipeline_result$phase1$evaluation)) {
    pipeline_result$phase1$evaluation$overall
  } else NULL

  manifest <- list(
    package_name = package_name,
    version = "v7",
    created = Sys.time(),
    r_version = R.version.string,
    config = pipeline_result$config,
    evaluation = eval_metrics,
    data_summary = list(
      n_train = nrow(pipeline_result$data_splits$train$data),
      n_validation = nrow(pipeline_result$data_splits$validation$data),
      n_test = nrow(pipeline_result$data_splits$test$data),
      n_epitopes = mappings$n_epitopes
    ),
    files = c(
      "model_weights.pt", "model_config.rds", "model_history.rds",
      "model_vocabs.rds", "epitope_reference.rds", "epitope_mappings.rds",
      "manifest.rds"
    )
  )
  manifest_path <- file.path(pkg_path, "manifest.rds")
  saveRDS(manifest, manifest_path)
  message("  Saved: manifest.rds")
  saved$manifest <- manifest_path

  # Summary
  message("\n", strrep("=", 60))
  message("Package saved: ", normalizePath(pkg_path))
  message(strrep("=", 60))

  files <- list.files(pkg_path, full.names = TRUE)
  total_size <- sum(file.size(files)) / 1024^2
  message(sprintf("Total size: %.1f MB (%d files)", total_size, length(files)))

  invisible(saved)
}

#' Load complete V7 model package
#'
#' @param package_dir Base directory containing packages
#' @param package_name Name of package to load
#' @param device Device for model ("cpu" or "cuda")
#' @return List with model, trainer, epitope_reference, vocabs, config
#' @export
load_model_package <- function(package_dir = "models",
                               package_name = "tcr_epitope_v7",
                               device = "cpu") {

  pkg_path <- file.path(package_dir, package_name)
  if (!dir.exists(pkg_path)) stop("Package not found: ", pkg_path)

  message("\n", strrep("=", 60))
  message("Loading Model Package: ", package_name)
  message(strrep("=", 60))

  # Load manifest
  manifest_path <- file.path(pkg_path, "manifest.rds")
  if (!file.exists(manifest_path)) stop("Invalid package: manifest.rds not found")
  manifest <- readRDS(manifest_path)

  message("\nPackage info:")
  message("  Version: ", manifest$version)
  message("  Created: ", manifest$created)

  # Load vocabularies
  message("\n[1/4] Loading vocabularies...")
  vocabs <- readRDS(file.path(pkg_path, "model_vocabs.rds"))

  # Load model
  message("\n[2/4] Loading model...")
  model_result <- load_model(
    model_dir = pkg_path,
    model_name = "model",
    trb_vocab = vocabs$trb,
    tra_vocab = vocabs$tra,
    device = device
  )

  # Create trainer for inference
  trainer <- py$TCRTrainerV7(
    model = model_result$model,
    device = device,
    loss_type = manifest$config$loss_type %||% "focal",
    focal_gamma = manifest$config$focal_gamma %||% 2.0,
    label_smoothing = manifest$config$label_smoothing %||% 0.0,
    ewc_lambda = 0.0,
    blosum_lambda = 0.0
  )

  # Load epitope reference
  message("\n[3/4] Loading epitope reference...")
  epi_ref <- load_epitope_reference(file.path(pkg_path, "epitope_reference.rds"))

  # Load mappings
  message("\n[4/4] Loading epitope mappings...")
  mappings <- readRDS(file.path(pkg_path, "epitope_mappings.rds"))

  message("\n", strrep("=", 60))
  message("Package loaded successfully")
  message(strrep("=", 60))

  # Show metrics
  if (!is.null(manifest$evaluation)) {
    message("\nTraining performance:")
    message("  Accuracy: ", round(manifest$evaluation$accuracy * 100, 1), "%")
    message("  Top-5: ", round(manifest$evaluation$top5_accuracy * 100, 1), "%")
  }

  list(
    model = model_result$model,
    trainer = trainer,
    epitope_reference = epi_ref,
    trb_vocab = vocabs$trb,
    tra_vocab = vocabs$tra,
    epitope_to_idx = mappings$epitope_to_idx,
    idx_to_epitope = mappings$idx_to_epitope,
    unique_epitopes = mappings$unique_epitopes,
    config = model_result$config,
    history = model_result$history,
    manifest = manifest
  )
}

# ===== Convenience Functions =====

#' Quick inference from saved package
#'
#' @param query_data Data frame with paired chain columns
#' @param package_dir Package directory
#' @param package_name Package name
#' @param top_k Number of predictions
#' @param device Compute device
#' @return Predictions tibble
#' @export
quick_predict <- function(query_data,
                          package_dir = "models",
                          package_name = "tcr_epitope_v7",
                          top_k = 5,
                          device = "cpu") {

  pkg <- load_model_package(package_dir, package_name, device)

  predict_epitopes(
    query_data = query_data,
    trainer = pkg$trainer,
    epitope_reference = pkg$epitope_reference,
    tra_vocab = pkg$tra_vocab,
    trb_vocab = pkg$trb_vocab,
    top_k = top_k
  )
}

#' List available model packages
#'
#' @param package_dir Directory to search
#' @return Tibble of available packages with metadata
#' @export
list_model_packages <- function(package_dir = "models") {

  if (!dir.exists(package_dir)) return(tibble())

  dirs <- list.dirs(package_dir, recursive = FALSE, full.names = TRUE)

  packages <- lapply(dirs, function(d) {
    manifest_path <- file.path(d, "manifest.rds")
    if (!file.exists(manifest_path)) return(NULL)

    m <- readRDS(manifest_path)
    tibble(
      name = basename(d),
      version = m$version %||% "unknown",
      created = m$created,
      n_epitopes = m$data_summary$n_epitopes %||% NA,
      accuracy = m$evaluation$accuracy %||% NA
    )
  })

  bind_rows(packages)
}

# Helper for null-coalescing
`%||%` <- function(x, y) if (is.null(x)) y else x
