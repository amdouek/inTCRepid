# Inference pipeline for TCR-epitope prediction (V7: dual-chain)

library(dplyr)
library(tidyr)
library(stringr)
library(reticulate)

# ===== Sequence Validation =====

#' Clean and validate CDR3 sequences for inference
#'
#' @param sequences Character vector of CDR3 sequences
#' @param min_length Minimum valid length (default 8)
#' @param max_length Maximum valid length (default 25)
#' @return Cleaned sequences (invalid entries become NA)
#' @export
clean_cdr3_for_inference <- function(sequences, min_length = 8, max_length = 25) {

  valid_aa <- "ACDEFGHIKLMNPQRSTVWY"
  invalid_pattern <- paste0("[^", valid_aa, "]")


  cleaned <- sequences %>%
    str_to_upper() %>%
    str_trim()

  # Validation checks
  has_invalid_chars <- str_detect(cleaned, invalid_pattern)
  has_invalid_chars[is.na(has_invalid_chars)] <- TRUE

  too_short <- nchar(cleaned) < min_length
  too_long <- nchar(cleaned) > max_length
  too_short[is.na(too_short)] <- TRUE
  too_long[is.na(too_long)] <- TRUE

  invalid <- has_invalid_chars | too_short | too_long | is.na(cleaned)
  cleaned[invalid] <- NA

  cleaned
}

#' Validate paired chain data for V7 inference
#'
#' @param query_data Data frame with CDR3 and V/J columns
#' @param require_alpha If TRUE, require TRA chain (default FALSE)
#' @return Validated data with validity flags
#' @export
validate_paired_chain_data <- function(query_data, require_alpha = FALSE) {

  # Required columns
  required <- c("cdr3_beta", "v_beta", "j_beta")
  if (require_alpha) {
    required <- c(required, "cdr3_alpha", "v_alpha", "j_alpha")
  }

  missing <- setdiff(required, names(query_data))
  if (length(missing) > 0) {
    stop("Missing required columns: ", paste(missing, collapse = ", "))
  }

  # Validate beta chain (required)
  query_data <- query_data %>%
    mutate(
      cdr3_beta_clean = clean_cdr3_for_inference(cdr3_beta),
      beta_valid = !is.na(cdr3_beta_clean) & !is.na(v_beta) & !is.na(j_beta)
    )

  # Validate alpha chain (optional unless required)
  if ("cdr3_alpha" %in% names(query_data)) {
    query_data <- query_data %>%
      mutate(
        cdr3_alpha_clean = clean_cdr3_for_inference(cdr3_alpha),
        alpha_valid = !is.na(cdr3_alpha_clean) & !is.na(v_alpha) & !is.na(j_alpha),
        is_paired = beta_valid & alpha_valid
      )
  } else {
    query_data <- query_data %>%
      mutate(
        cdr3_alpha_clean = NA_character_,
        alpha_valid = FALSE,
        is_paired = FALSE
      )
  }

  # Overall validity
  if (require_alpha) {
    query_data$valid <- query_data$beta_valid & query_data$alpha_valid
  } else {
    query_data$valid <- query_data$beta_valid
  }

  n_valid <- sum(query_data$valid)
  n_paired <- sum(query_data$is_paired, na.rm = TRUE)

  message(sprintf("Validated %d/%d sequences (%d paired)",
                  n_valid, nrow(query_data), n_paired))

  query_data
}

# ===== Epitope Reference Database =====

#' Build epitope reference database from training data
#'
#' @param trainer Trained TCRTrainerV7 object
#' @param unique_epitopes Character vector of unique epitope sequences
#' @param unique_epitope_idx Encoded epitope indices matrix
#' @return List with epitope sequences and embeddings
#' @export
build_epitope_reference <- function(trainer, unique_epitopes, unique_epitope_idx) {

  message("Building epitope reference database...")
  message("  Epitopes: ", length(unique_epitopes))

  # Get embeddings via trainer
  trainer$model$eval()

  py_run_string("
import torch
import numpy as np

def get_epitope_embeddings(model, epitope_idx, device):
    model.eval()
    with torch.no_grad():
        epi_tensor = torch.tensor(np.array(epitope_idx, copy=True),
                                  dtype=torch.long).to(device)
        emb = model.encode_epitope(epi_tensor)
    return emb.cpu().numpy()
")

  embeddings <- py$get_epitope_embeddings(
    trainer$model,
    unique_epitope_idx,
    trainer$device
  )

  message("  Embedding dim: ", ncol(embeddings))

  list(
    epitopes = unique_epitopes,
    embeddings = embeddings,
    n_epitopes = length(unique_epitopes),
    embedding_dim = ncol(embeddings)
  )
}

# ===== V7 Inference (Dual Chain) =====

#' Predict epitopes for paired TCR chains (V7)
#'
#' @param query_data Data frame with cdr3_alpha, cdr3_beta, v_alpha, j_alpha, v_beta, j_beta
#' @param trainer Trained TCRTrainerV7 object
#' @param epitope_reference Reference from build_epitope_reference()
#' @param tra_vocab TRA V/J vocabulary
#' @param trb_vocab TRB V/J vocabulary
#' @param top_k Number of top predictions per query
#' @param batch_size Batch size for processing
#' @param calibration_temp Calibration temperature (default 1.0)
#' @return Tibble with predictions
#' @export
predict_epitopes <- function(query_data,
                             trainer,
                             epitope_reference,
                             tra_vocab,
                             trb_vocab,
                             top_k = 5,
                             batch_size = 256,
                             calibration_temp = 1.0) {

  # Validate input
  query_data <- validate_paired_chain_data(query_data, require_alpha = FALSE)

  valid_idx <- which(query_data$valid)
  if (length(valid_idx) == 0) {
    warning("No valid sequences to process")
    return(create_empty_predictions(nrow(query_data), top_k))
  }

  valid_data <- query_data[valid_idx, ]
  message("Processing ", nrow(valid_data), " valid TCRs...")

  # Encode sequences
  cdr3_alpha_idx <- encode_cdr3_for_inference(
    valid_data$cdr3_alpha_clean,
    max_length = 25L,
    fill_missing = TRUE
  )
  cdr3_beta_idx <- sequences_to_indices(valid_data$cdr3_beta_clean, 25L)

  # Encode V/J genes
  v_alpha_idx <- encode_vj_for_inference(valid_data$v_alpha, tra_vocab$v, "TRA-V")
  j_alpha_idx <- encode_vj_for_inference(valid_data$j_alpha, tra_vocab$j, "TRA-J")
  v_beta_idx <- encode_vj_for_inference(valid_data$v_beta, trb_vocab$v, "TRB-V")
  j_beta_idx <- encode_vj_for_inference(valid_data$j_beta, trb_vocab$j, "TRB-J")

  # Get TCR embeddings
  message("Computing TCR embeddings...")
  tcr_embeddings <- get_tcr_embeddings_batched(
    trainer, cdr3_alpha_idx, cdr3_beta_idx,
    v_alpha_idx, j_alpha_idx, v_beta_idx, j_beta_idx,
    batch_size
  )

  # Compute similarities
  message("Computing epitope similarities...")
  similarities <- compute_similarities(tcr_embeddings, epitope_reference$embeddings)

  # Apply calibration
  if (calibration_temp != 1.0) {
    similarities <- similarities / calibration_temp
  }

  # Convert to probabilities
  probs <- softmax_rows(similarities)

  # Rank predictions
  message("Ranking predictions...")
  predictions <- rank_predictions(
    probabilities = probs,
    epitope_names = epitope_reference$epitopes,
    original_indices = valid_idx,
    top_k = top_k
  )

  # Add metadata
  predictions <- predictions %>%
    left_join(
      query_data %>%
        mutate(.idx = row_number()) %>%
        select(.idx, cdr3_alpha, cdr3_beta, is_paired),
      by = c("query_index" = ".idx")
    )

  # Add invalid sequences
  if (nrow(query_data) > length(valid_idx)) {
    predictions <- merge_invalid_sequences(predictions, nrow(query_data), top_k)
  }

  message("Inference complete: ", n_distinct(predictions$query_index), " TCRs processed")
  predictions
}

#' Encode CDR3 sequences for inference (with missing chain handling)
#'
#' @param sequences Character vector (NA for missing chains)
#' @param max_length Maximum sequence length
#' @param fill_missing If TRUE, fill missing with placeholder
#' @return Integer matrix of indices
encode_cdr3_for_inference <- function(sequences, max_length = 25L, fill_missing = TRUE) {

  # Replace NA with placeholder sequence
  if (fill_missing) {
    placeholder <- paste(rep("X", 10), collapse = "")  # Will encode as UNK
    sequences[is.na(sequences)] <- placeholder
  }

  sequences_to_indices(sequences, max_length)
}

#' Encode V/J genes for inference
#'
#' @param genes Character vector of gene names
#' @param vocab V or J vocabulary list (e.g., tra_vocab$v)
#' @param gene_type Description for messages
#' @return Integer vector of indices
encode_vj_for_inference <- function(genes, vocab, gene_type = "gene") {

  # Handle NA/missing
  genes[is.na(genes)] <- "<MISSING>"

  # Normalize gene names
  genes_norm <- normalize_gene_names(genes)

  # Map to indices - use correct key 'to_idx'
  indices <- vapply(genes_norm, function(g) {
    if (g %in% names(vocab$to_idx)) {
      vocab$to_idx[[g]]
    } else {
      vocab$to_idx[["<UNK>"]]
    }
  }, integer(1), USE.NAMES = FALSE)

  n_unk <- sum(indices == vocab$to_idx[["<UNK>"]])
  if (n_unk > 0) {
    message(sprintf("  %s: %d/%d mapped to <UNK>", gene_type, n_unk, length(genes)))
  }

  as.integer(indices)
}

#' Get TCR embeddings in batches (V7)
get_tcr_embeddings_batched <- function(trainer, cdr3_alpha_idx, cdr3_beta_idx,
                                       v_alpha_idx, j_alpha_idx,
                                       v_beta_idx, j_beta_idx,
                                       batch_size = 256) {

  n <- nrow(cdr3_beta_idx)
  n_batches <- ceiling(n / batch_size)

  all_emb <- list()

  for (i in seq_len(n_batches)) {
    start <- (i - 1) * batch_size + 1
    end <- min(i * batch_size, n)
    idx <- start:end

    emb <- trainer$get_embeddings(
      cdr3_alpha_idx = cdr3_alpha_idx[idx, , drop = FALSE],
      cdr3_beta_idx = cdr3_beta_idx[idx, , drop = FALSE],
      v_alpha_idx = v_alpha_idx[idx],
      j_alpha_idx = j_alpha_idx[idx],
      v_beta_idx = v_beta_idx[idx],
      j_beta_idx = j_beta_idx[idx]
    )

    all_emb[[i]] <- emb
  }

  do.call(rbind, all_emb)
}

# ===== Similarity and Ranking =====

#' Compute cosine similarities (embeddings assumed L2-normalized)
compute_similarities <- function(query_emb, reference_emb) {
  # Matrix multiplication for cosine similarity
  query_emb %*% t(reference_emb)
}

#' Row-wise softmax
softmax_rows <- function(x) {
  exp_x <- exp(x - apply(x, 1, max))  # Subtract max for numerical stability
  exp_x / rowSums(exp_x)
}

#' Rank predictions and format output
rank_predictions <- function(probabilities, epitope_names, original_indices, top_k = 5) {

  n <- nrow(probabilities)
  top_k <- min(top_k, ncol(probabilities))

  results <- lapply(seq_len(n), function(i) {
    probs <- probabilities[i, ]
    top_idx <- order(probs, decreasing = TRUE)[1:top_k]

    tibble(
      query_index = original_indices[i],
      rank = 1:top_k,
      predicted_epitope = epitope_names[top_idx],
      probability = probs[top_idx],
      confidence = probs[top_idx]  # For calibrated probs, these are the same
    )
  })

  bind_rows(results)
}

#' Create empty predictions for invalid sequences
create_empty_predictions <- function(n_queries, top_k) {
  tibble(
    query_index = integer(),
    rank = integer(),
    predicted_epitope = character(),
    probability = numeric(),
    confidence = numeric()
  )
}

#' Merge predictions with invalid sequence placeholders
merge_invalid_sequences <- function(predictions, n_total, top_k) {

  valid_idx <- unique(predictions$query_index)
  invalid_idx <- setdiff(seq_len(n_total), valid_idx)

  if (length(invalid_idx) > 0) {
    invalid_entries <- tibble(
      query_index = rep(invalid_idx, each = top_k),
      rank = rep(1:top_k, length(invalid_idx)),
      predicted_epitope = NA_character_,
      probability = NA_real_,
      confidence = NA_real_
    )

    predictions <- bind_rows(predictions, invalid_entries) %>%
      arrange(query_index, rank)
  }

  predictions
}

# ===== Convenience Functions =====

#' Get top-1 prediction for each query
#'
#' @param predictions Full predictions from predict_epitopes()
#' @return Tibble with only top prediction per query
#' @export
get_top_predictions <- function(predictions) {
  predictions %>%
    filter(rank == 1) %>%
    select(-rank)
}

#' Summarize predictions by epitope
#'
#' @param predictions Predictions tibble
#' @param rank_cutoff Only consider predictions up to this rank
#' @return Summary by epitope
#' @export
summarize_by_epitope <- function(predictions, rank_cutoff = 1) {

  predictions %>%
    filter(rank <= rank_cutoff, !is.na(predicted_epitope)) %>%
    group_by(predicted_epitope) %>%
    summarise(
      n_predictions = n(),
      mean_confidence = mean(confidence, na.rm = TRUE),
      median_confidence = median(confidence, na.rm = TRUE),
      max_confidence = max(confidence, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(n_predictions))
}

#' Compare predictions across conditions
#'
#' @param predictions Predictions with condition column
#' @param condition_col Name of condition column
#' @param rank_cutoff Max rank to consider
#' @return Comparison tibble
#' @export
compare_by_condition <- function(predictions, condition_col = "condition", rank_cutoff = 1) {

  if (!condition_col %in% names(predictions)) {
    stop("Column '", condition_col, "' not found")
  }

  predictions %>%
    filter(rank <= rank_cutoff, !is.na(predicted_epitope)) %>%
    group_by(.data[[condition_col]], predicted_epitope) %>%
    summarise(n = n(), mean_conf = mean(confidence, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(
      id_cols = predicted_epitope,
      names_from = all_of(condition_col),
      values_from = c(n, mean_conf),
      values_fill = list(n = 0)
    )
}

#' Predict with expanded output (all metadata)
#'
#' @param query_data Input data frame
#' @param trainer Trained trainer
#' @param epitope_reference Epitope reference database
#' @param tra_vocab TRA vocabulary
#' @param trb_vocab TRB vocabulary
#' @param top_k Top predictions
#' @param calibration_temp Calibration temperature
#' @return Predictions merged with all input columns
#' @export
predict_with_metadata <- function(query_data, trainer, epitope_reference,
                                  tra_vocab, trb_vocab, top_k = 5,
                                  calibration_temp = 1.0) {

  # Store original data
  query_data <- query_data %>%
    mutate(.row_id = row_number())

  # Get predictions
  predictions <- predict_epitopes(
    query_data, trainer, epitope_reference,
    tra_vocab, trb_vocab, top_k, calibration_temp = calibration_temp
  )

  # Merge with original data
  predictions %>%
    left_join(
      query_data %>% select(-any_of(c("cdr3_alpha", "cdr3_beta", "is_paired"))),
      by = c("query_index" = ".row_id")
    )
}

# ===== Gene Name Normalization (helper) =====

#' Normalize V/J gene names to standard format
normalize_gene_names <- function(genes) {
  genes %>%
    str_to_upper() %>%
    str_replace_all("\\*.*$", "") %>%  # Remove allele info
    str_replace_all("TRAV", "TRAV") %>%  # Standardize prefixes
    str_replace_all("TRAJ", "TRAJ") %>%
    str_replace_all("TRBV", "TRBV") %>%
    str_replace_all("TRBJ", "TRBJ")
}
