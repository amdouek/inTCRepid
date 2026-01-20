# model_persistence_legacy_v5-6.R
# Model persistence for V5/V6 models - DEPRECATED
# ============================================================================

library(reticulate)

#' Save V5/V6 model - DEPRECATED
#' @export
save_model_legacy <- function(trainer, model_dir = "models",
                              model_name = "tcr_epitope_model",
                              save_history = TRUE, history = NULL) {

  .Deprecated("save_model", msg = "Use save_model() from model_persistence.R for V7")

  if (!dir.exists(model_dir)) dir.create(model_dir, recursive = TRUE)

  weights_path <- file.path(model_dir, paste0(model_name, "_weights.pt"))
  config_path <- file.path(model_dir, paste0(model_name, "_config.rds"))

  torch <- import("torch")
  torch$save(trainer$model$state_dict(), weights_path)

  config <- list(
    model_class = "TCREpitopeModelV5",
    embedding_dim = as.integer(trainer$model$cdr3_encoder$fc$out_features),
    timestamp = Sys.time()
  )
  saveRDS(config, config_path)

  saved <- list(weights = weights_path, config = config_path)

  if (save_history && !is.null(history)) {
    history_path <- file.path(model_dir, paste0(model_name, "_history.rds"))
    saveRDS(lapply(history, unlist), history_path)
    saved$history <- history_path
  }

  message("Model saved to: ", model_dir)
  invisible(saved)
}

#' Load V5/V6 model - DEPRECATED
#' @export
load_model_legacy <- function(model_dir = "models",
                              model_name = "tcr_epitope_model",
                              device = "cpu",
                              vj_vocab = NULL) {

  .Deprecated("load_model", msg = "Use load_model() from model_persistence.R for V7")

  weights_path <- file.path(model_dir, paste0(model_name, "_weights.pt"))
  config_path <- file.path(model_dir, paste0(model_name, "_config.rds"))

  if (!file.exists(weights_path)) stop("Weights not found: ", weights_path)
  if (!file.exists(config_path)) stop("Config not found: ", config_path)

  config <- readRDS(config_path)

  # Detect model version
  is_v6 <- grepl("V6", config$model_class %||% "")

  if (is_v6 && !is.null(vj_vocab)) {
    model <- create_tcr_epitope_model_v6(
      embed_dim = config$embedding_dim %||% 128L,
      v_gene_vocab_size = as.integer(vj_vocab$v$size),
      j_gene_vocab_size = as.integer(vj_vocab$j$size)
    )
  } else {
    model <- create_tcr_epitope_model_v5(
      embed_dim = config$embedding_dim %||% 128L
    )
  }

  torch <- import("torch")
  state_dict <- torch$load(weights_path, map_location = device)
  model$load_state_dict(state_dict)
  model$eval()

  # Create appropriate trainer
  if (is_v6) {
    trainer <- py$TCRTrainerV6(model, device = device)
  } else {
    trainer <- py$TCRTrainerV5(model, device = device)
  }

  history_path <- file.path(model_dir, paste0(model_name, "_history.rds"))
  history <- if (file.exists(history_path)) readRDS(history_path) else NULL

  list(model = model, trainer = trainer, config = config, history = history)
}

#' Save V5/V6 model package - DEPRECATED
#' @export
save_model_package_legacy <- function(pipeline_result, package_dir = "models",
                                      package_name = "tcr_epitope_v5") {

  .Deprecated("save_model_package", msg = "Use save_model_package() for V7")

  pkg_path <- file.path(package_dir, package_name)
  if (!dir.exists(pkg_path)) dir.create(pkg_path, recursive = TRUE)

  # Save model
  save_model_legacy(
    trainer = pipeline_result$trainer,
    model_dir = pkg_path,
    model_name = "model",
    save_history = TRUE,
    history = pipeline_result$history
  )

  # Build epitope reference
  epi_ref <- build_epitope_reference_legacy(
    trainer = pipeline_result$trainer,
    unique_epitopes = pipeline_result$data_splits$unique_epitopes
  )
  saveRDS(epi_ref, file.path(pkg_path, "epitope_reference.rds"))

  # Save splits info
  splits_info <- list(
    epitope_to_idx = pipeline_result$data_splits$epitope_to_idx,
    idx_to_epitope = pipeline_result$data_splits$idx_to_epitope,
    unique_epitopes = pipeline_result$data_splits$unique_epitopes
  )
  saveRDS(splits_info, file.path(pkg_path, "data_splits_info.rds"))

  # Manifest
  manifest <- list(
    package_name = package_name,
    version = "v5",
    created = Sys.time(),
    config = pipeline_result$config
  )
  saveRDS(manifest, file.path(pkg_path, "manifest.rds"))

  message("Package saved: ", pkg_path)
}

#' Load V5/V6 model package - DEPRECATED
#' @export
load_model_package_legacy <- function(package_dir = "models",
                                      package_name = "tcr_epitope_v5",
                                      device = "cpu",
                                      vj_vocab = NULL) {

  .Deprecated("load_model_package", msg = "Use load_model_package() for V7")

  pkg_path <- file.path(package_dir, package_name)
  if (!dir.exists(pkg_path)) stop("Package not found: ", pkg_path)

  model_result <- load_model_legacy(pkg_path, "model", device, vj_vocab)
  epi_ref <- readRDS(file.path(pkg_path, "epitope_reference.rds"))
  splits_info <- readRDS(file.path(pkg_path, "data_splits_info.rds"))
  manifest <- readRDS(file.path(pkg_path, "manifest.rds"))

  list(
    model = model_result$model,
    trainer = model_result$trainer,
    epitope_reference = epi_ref,
    splits_info = splits_info,
    config = model_result$config,
    history = model_result$history,
    manifest = manifest
  )
}

#' Quick predict V5/V6 - DEPRECATED
#' @export
quick_predict_legacy <- function(query_cdr3, package_dir = "models",
                                 package_name = "tcr_epitope_v5",
                                 top_k = 5, device = "cpu") {

  .Deprecated("quick_predict", msg = "Use quick_predict() for V7")

  pkg <- load_model_package_legacy(package_dir, package_name, device)
  predict_epitopes_legacy(query_cdr3, pkg$trainer, pkg$epitope_reference, top_k)
}
