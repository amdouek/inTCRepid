# Inference pipeline for V5/V6 models (single-chain) - DEPRECATED
# ============================================================================

library(dplyr)
library(stringr)
library(reticulate)

#' Predict epitopes for CDR3 sequences (V5/V6)
#'
#' @param query_cdr3 Character vector of CDR3 sequences
#' @param trainer Trained TCRTrainerV5/V6 object
#' @param epitope_reference Reference from build_epitope_reference_legacy()
#' @param top_k Number of top predictions
#' @param batch_size Batch size
#' @param v_gene_idx V gene indices (V6 only)
#' @param j_gene_idx J gene indices (V6 only)
#' @return Predictions tibble
#' @export
predict_epitopes_legacy <- function(query_cdr3, trainer, epitope_reference,
                                    top_k = 5, batch_size = 256,
                                    v_gene_idx = NULL, j_gene_idx = NULL) {

  .Deprecated("predict_epitopes",
              msg = "Use predict_epitopes() from inference.R for V7 models")

  # Clean sequences
  valid_aa <- "ACDEFGHIKLMNPQRSTVWY"
  cleaned <- query_cdr3 %>% str_to_upper() %>% str_trim()

  invalid <- str_detect(cleaned, paste0("[^", valid_aa, "]")) |
    nchar(cleaned) < 8 | nchar(cleaned) > 25 | is.na(cleaned)
  invalid[is.na(invalid)] <- TRUE
  cleaned[invalid] <- NA

  valid_idx <- which(!is.na(cleaned))
  if (length(valid_idx) == 0) {
    warning("No valid sequences")
    return(tibble())
  }

  valid_seqs <- cleaned[valid_idx]
  query_idx <- sequences_to_indices(valid_seqs, 25L)

  # Get embeddings
  if (!is.null(v_gene_idx) && !is.null(j_gene_idx)) {
    # V6 with V/J
    emb <- trainer$get_embeddings(query_idx, v_gene_idx[valid_idx], j_gene_idx[valid_idx])
  } else {
    # V5
    emb <- trainer$get_embeddings(query_idx)
  }

  # Compute similarities
  sims <- emb %*% t(epitope_reference$embeddings)

  # Rank
  top_k <- min(top_k, ncol(sims))

  results <- lapply(seq_len(nrow(sims)), function(i) {
    s <- sims[i, ]
    top_idx <- order(s, decreasing = TRUE)[1:top_k]
    tibble(
      query_index = valid_idx[i],
      query_cdr3 = valid_seqs[i],
      rank = 1:top_k,
      predicted_epitope = epitope_reference$epitopes[top_idx],
      similarity_score = s[top_idx],
      confidence = pmax(0, pmin(1, s[top_idx]))
    )
  })

  bind_rows(results)
}

#' Build epitope reference for V5/V6 models
#'
#' @param trainer TCRTrainerV5/V6 object
#' @param unique_epitopes Character vector of epitopes
#' @return Reference list
#' @export
build_epitope_reference_legacy <- function(trainer, unique_epitopes) {

  epitope_idx <- sequences_to_indices(unique_epitopes, 30L)
  emb <- trainer$get_epitope_embeddings(epitope_idx)

  list(
    epitopes = unique_epitopes,
    embeddings = emb,
    n_epitopes = length(unique_epitopes),
    embedding_dim = ncol(emb)
  )
}

#' Get top predictions (V5/V6)
#' @export
get_top_predictions_legacy <- function(predictions) {
  predictions %>% filter(rank == 1)
}

#' Summarize by epitope (V5/V6)
#' @export
summarize_predictions_legacy <- function(predictions, rank_cutoff = 1) {
  predictions %>%
    filter(rank <= rank_cutoff, !is.na(predicted_epitope)) %>%
    group_by(predicted_epitope) %>%
    summarise(
      n = n(),
      mean_sim = mean(similarity_score, na.rm = TRUE),
      mean_conf = mean(confidence, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(n))
}

#' Calibrated prediction (V5/V6)
#' @export
predict_epitopes_calibrated_legacy <- function(trainer, model, cdr3_sequences,
                                               unique_epitopes, calibration_temp = 1.0,
                                               top_k = 5, device = "cpu") {

  .Deprecated("predict_epitopes",
              msg = "Use predict_epitopes() with calibration_temp parameter")

  cdr3_idx <- sequences_to_indices(cdr3_sequences, 25L)
  epitope_idx <- sequences_to_indices(unique_epitopes, 30L)

  py_run_string(sprintf("
import torch
import torch.nn.functional as F
import numpy as np

def predict_calibrated_v5(model, cdr3_idx, epitope_idx, temp, top_k, device):
    model.eval()
    cdr3_t = torch.tensor(np.array(cdr3_idx, copy=True), dtype=torch.long).to(device)
    epi_t = torch.tensor(np.array(epitope_idx, copy=True), dtype=torch.long).to(device)

    with torch.no_grad():
        cdr3_emb = model.encode_cdr3(cdr3_t)
        epi_emb = model.encode_epitope(epi_t)
        logits = torch.mm(cdr3_emb, epi_emb.t()) / model.temperature.clamp(min=0.01)
        probs = F.softmax(logits / temp, dim=1)
        top_p, top_i = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)

    return {'top_indices': top_i.cpu().numpy(), 'top_probs': top_p.cpu().numpy()}

_res = predict_calibrated_v5(r.model, r.cdr3_idx, r.epitope_idx, %f, %d, '%s')
", calibration_temp, top_k, device))

  res <- py$`_res`

  tibble(
    query_index = seq_along(cdr3_sequences),
    cdr3 = cdr3_sequences,
    predicted_epitope = unique_epitopes[res$top_indices[, 1] + 1],
    confidence = res$top_probs[, 1]
  )
}
