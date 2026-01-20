# Confidence calibration via temperature scaling (V7)
# Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
# ============================================================================

library(reticulate)
library(ggplot2)
library(dplyr)
library(tidyr)

# ===== Python Calibration Infrastructure =====

py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# --- Calibration Metrics ---
def compute_ece(probs, labels, n_bins=15):
    '''Expected Calibration Error'''
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            acc = float(np.mean(accuracies[mask]))
            conf = float(np.mean(confidences[mask]))
            ece += (count / len(labels)) * np.abs(acc - conf)
            bin_accs.append(acc)
            bin_confs.append(conf)
        else:
            bin_accs.append(float('nan'))
            bin_confs.append(float('nan'))

    return float(ece), bin_accs, bin_confs, bin_counts, bins.tolist()

def compute_mce(bin_accs, bin_confs):
    '''Maximum Calibration Error'''
    mce = 0.0
    for acc, conf in zip(bin_accs, bin_confs):
        if not np.isnan(acc):
            mce = max(mce, np.abs(acc - conf))
    return float(mce)

def brier_score(probs, labels):
    '''Brier Score'''
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def learn_temperature(logits, labels, lr=0.01, max_iter=100):
    '''Learn optimal temperature via LBFGS'''
    log_temp = nn.Parameter(torch.zeros(1))
    optimizer = optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        temp = log_temp.exp().clamp(min=0.01, max=10.0)
        loss = criterion(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temp.exp().clamp(min=0.01, max=10.0).item())

# --- V7 Calibration (Dual Chain) ---
def calibrate_v7(model, cdr3_alpha, cdr3_beta, v_alpha, j_alpha,
                 v_beta, j_beta, labels, epitope_idx, device='cpu'):
    '''
    Calibrate V7 model using temperature scaling.

    Returns dict with temperature, metrics before/after, and bin data.
    '''
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Convert to tensors
    def to_tensor(x, dtype=torch.long):
        return torch.tensor(np.array(x, copy=True), dtype=dtype).to(device)

    cdr3_a_t = to_tensor(cdr3_alpha)
    cdr3_b_t = to_tensor(cdr3_beta)
    v_a_t = to_tensor(v_alpha)
    j_a_t = to_tensor(j_alpha)
    v_b_t = to_tensor(v_beta)
    j_b_t = to_tensor(j_beta)
    labels_t = to_tensor(labels)
    epi_t = to_tensor(epitope_idx)

    n_samples = cdr3_b_t.shape[0]
    print(f'Calibrating on {n_samples} samples...')

    # Collect logits
    all_logits = []
    batch_size = 256

    with torch.no_grad():
        epi_emb = model.encode_epitope(epi_t)

        for i in range(0, n_samples, batch_size):
            j = min(i + batch_size, n_samples)

            tcr_emb = model.encode_tcr(
                cdr3_a_t[i:j], cdr3_b_t[i:j],
                v_a_t[i:j], j_a_t[i:j],
                v_b_t[i:j], j_b_t[i:j]
            )
            logits = torch.mm(tcr_emb, epi_emb.t()) / model.temperature.clamp(min=0.01)
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels_np = labels_t.cpu().numpy()

    # Before calibration
    probs_before = F.softmax(torch.tensor(logits), dim=1).numpy()
    ece_before, ba_before, bc_before, cnt_before, bins = compute_ece(probs_before, labels_np)
    mce_before = compute_mce(ba_before, bc_before)
    brier_before = brier_score(probs_before, labels_np)

    print(f'  ECE before: {ece_before:.4f}')

    # Learn temperature
    opt_temp = learn_temperature(logits, labels_np)
    print(f'  Optimal temperature: {opt_temp:.4f}')

    # After calibration
    probs_after = F.softmax(torch.tensor(logits / opt_temp), dim=1).numpy()
    ece_after, ba_after, bc_after, cnt_after, _ = compute_ece(probs_after, labels_np)
    mce_after = compute_mce(ba_after, bc_after)
    brier_after = brier_score(probs_after, labels_np)

    print(f'  ECE after: {ece_after:.4f}')
    print(f'  Improvement: {100*(1 - ece_after/ece_before):.1f}%')

    return {
        'optimal_temperature': opt_temp,
        'ece_before': ece_before, 'ece_after': ece_after,
        'mce_before': mce_before, 'mce_after': mce_after,
        'brier_before': brier_before, 'brier_after': brier_after,
        'bin_data_before': {'bin_acc': ba_before, 'bin_conf': bc_before,
                           'bin_count': cnt_before, 'bin_edges': bins},
        'bin_data_after': {'bin_acc': ba_after, 'bin_conf': bc_after,
                          'bin_count': cnt_after, 'bin_edges': bins},
        'probs_before': probs_before,
        'probs_after': probs_after,
        'labels': labels_np,
        'n_samples': n_samples,
        'n_epitopes': int(epi_t.shape[0])
    }
")

# ===== R Calibration Function (V7) =====

#' Calibrate model confidence using temperature scaling (V7)
#'
#' Learns optimal temperature on validation data to improve calibration.
#'
#' @param model Trained V7 model
#' @param data_splits Data splits from prepare_paired_chain_data()
#' @param device Computation device
#' @return List with calibration results
#' @export
calibrate_model <- function(model, data_splits, device = "cpu") {

  message("\n", strrep("=", 60))
  message("CONFIDENCE CALIBRATION (Temperature Scaling)")
  message(strrep("=", 60))

  # Validate V7 data structure
  required <- c("cdr3_alpha_idx", "cdr3_beta_idx", "v_alpha_idx", "j_alpha_idx",
                "v_beta_idx", "j_beta_idx", "labels")

  missing <- setdiff(required, names(data_splits$validation))
  if (length(missing) > 0) {
    stop("Missing V7 columns in validation data: ", paste(missing, collapse = ", "))
  }

  val <- data_splits$validation
  message("\nValidation samples: ", nrow(val$cdr3_beta_idx))
  message("Epitope classes: ", nrow(data_splits$unique_epitope_idx))

  # Run calibration
  message("\nLearning optimal temperature...")

  results <- py$calibrate_v7(
    model = model,
    cdr3_alpha = val$cdr3_alpha_idx,
    cdr3_beta = val$cdr3_beta_idx,
    v_alpha = val$v_alpha_idx,
    j_alpha = val$j_alpha_idx,
    v_beta = val$v_beta_idx,
    j_beta = val$j_beta_idx,
    labels = val$labels,
    epitope_idx = data_splits$unique_epitope_idx,
    device = device
  )

  # Report results
  message("\n", strrep("-", 50))
  message("CALIBRATION RESULTS")
  message(strrep("-", 50))

  message(sprintf("\nOptimal Temperature: %.4f", results$optimal_temperature))

  if (results$optimal_temperature > 1) {
    message("  → Model was OVERCONFIDENT (T > 1 softens predictions)")
  } else {
    message("  → Model was UNDERCONFIDENT (T < 1 sharpens predictions)")
  }

  message(sprintf("\nExpected Calibration Error (ECE):"))
  message(sprintf("  Before: %.4f", results$ece_before))
  message(sprintf("  After:  %.4f", results$ece_after))
  message(sprintf("  Improvement: %.1f%%", 100 * (1 - results$ece_after / results$ece_before)))

  message(sprintf("\nMaximum Calibration Error (MCE):"))
  message(sprintf("  Before: %.4f → After: %.4f", results$mce_before, results$mce_after))

  message(sprintf("\nBrier Score:"))
  message(sprintf("  Before: %.4f → After: %.4f", results$brier_before, results$brier_after))

  message("\n", strrep("=", 60))

  results
}

# ===== Plotting Functions =====

#' Plot reliability diagram (calibration curve)
#'
#' @param calib_results Output from calibrate_model()
#' @param show_before_after Show both pre and post calibration
#' @return ggplot object
#' @export
plot_reliability_diagram <- function(calib_results, show_before_after = TRUE) {

  n_bins <- length(calib_results$bin_data_before$bin_acc)
  edges <- calib_results$bin_data_before$bin_edges
  centers <- (edges[-1] + edges[-(n_bins + 1)]) / 2

  if (show_before_after) {
    df <- tibble(
      bin_center = rep(centers, 2),
      accuracy = c(unlist(calib_results$bin_data_before$bin_acc),
                   unlist(calib_results$bin_data_after$bin_acc)),
      confidence = c(unlist(calib_results$bin_data_before$bin_conf),
                     unlist(calib_results$bin_data_after$bin_conf)),
      count = c(unlist(calib_results$bin_data_before$bin_count),
                unlist(calib_results$bin_data_after$bin_count)),
      stage = rep(c(sprintf("Before (ECE=%.3f)", calib_results$ece_before),
                    sprintf("After (ECE=%.3f)", calib_results$ece_after)),
                  each = n_bins)
    ) %>% filter(!is.na(accuracy))

    p <- ggplot(df, aes(x = confidence, y = accuracy, color = stage)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
      geom_line(linewidth = 1) +
      geom_point(aes(size = count), alpha = 0.7) +
      facet_wrap(~stage) +
      scale_color_manual(values = c("firebrick", "steelblue"), guide = "none")
  } else {
    df <- tibble(
      confidence = unlist(calib_results$bin_data_after$bin_conf),
      accuracy = unlist(calib_results$bin_data_after$bin_acc),
      count = unlist(calib_results$bin_data_after$bin_count)
    ) %>% filter(!is.na(accuracy))

    p <- ggplot(df, aes(x = confidence, y = accuracy)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
      geom_line(linewidth = 1, color = "steelblue") +
      geom_point(aes(size = count), alpha = 0.7, color = "steelblue")
  }

  p + scale_size_continuous(range = c(2, 8), name = "Samples") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "Reliability Diagram",
         subtitle = sprintf("Optimal T = %.3f", calib_results$optimal_temperature),
         x = "Mean Predicted Confidence", y = "Actual Accuracy") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold"))
}

#' Plot confidence distribution before/after calibration
#'
#' @param calib_results Output from calibrate_model()
#' @return ggplot object
#' @export
plot_confidence_distribution <- function(calib_results) {

  labels <- calib_results$labels

  conf_before <- apply(calib_results$probs_before, 1, max)
  conf_after <- apply(calib_results$probs_after, 1, max)

  pred_before <- apply(calib_results$probs_before, 1, which.max) - 1
  pred_after <- apply(calib_results$probs_after, 1, which.max) - 1

  df <- tibble(
    confidence = c(conf_before, conf_after),
    correct = c(pred_before == labels, pred_after == labels),
    stage = factor(rep(c("Before", "After"), each = length(labels)),
                   levels = c("Before", "After"))
  ) %>%
    mutate(outcome = ifelse(correct, "Correct", "Incorrect"))

  ggplot(df, aes(x = confidence, fill = outcome)) +
    geom_histogram(bins = 30, position = "stack", alpha = 0.8) +
    facet_wrap(~stage, ncol = 1) +
    scale_fill_manual(values = c("Correct" = "steelblue", "Incorrect" = "firebrick")) +
    labs(title = "Confidence Distribution",
         subtitle = "Before vs After Temperature Scaling",
         x = "Confidence", y = "Count", fill = "Prediction") +
    theme_minimal() +
    theme(legend.position = "bottom")
}

#' Plot ECE improvement summary
#'
#' @param calib_results Output from calibrate_model()
#' @return ggplot object
#' @export
plot_calibration_summary <- function(calib_results) {

  metrics <- tibble(
    metric = rep(c("ECE", "MCE", "Brier"), 2),
    stage = rep(c("Before", "After"), each = 3),
    value = c(calib_results$ece_before, calib_results$mce_before, calib_results$brier_before,
              calib_results$ece_after, calib_results$mce_after, calib_results$brier_after)
  ) %>%
    mutate(stage = factor(stage, levels = c("Before", "After")))

  ggplot(metrics, aes(x = metric, y = value, fill = stage)) +
    geom_col(position = "dodge", alpha = 0.8) +
    scale_fill_manual(values = c("Before" = "firebrick", "After" = "steelblue")) +
    labs(title = "Calibration Metrics",
         subtitle = sprintf("Temperature = %.3f", calib_results$optimal_temperature),
         x = NULL, y = "Value", fill = NULL) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

# ===== Save/Load Calibration =====

#' Save calibration results
#'
#' @param calib_results Output from calibrate_model()
#' @param output_path File path for saving
#' @export
save_calibration <- function(calib_results, output_path) {

  # Save key parameters (not full probability matrices)
  to_save <- list(
    optimal_temperature = calib_results$optimal_temperature,
    ece_before = calib_results$ece_before,
    ece_after = calib_results$ece_after,
    mce_before = calib_results$mce_before,
    mce_after = calib_results$mce_after,
    brier_before = calib_results$brier_before,
    brier_after = calib_results$brier_after,
    n_samples = calib_results$n_samples,
    n_epitopes = calib_results$n_epitopes
  )

  saveRDS(to_save, output_path)
  message("Saved calibration to: ", output_path)
}

#' Load calibration results
#'
#' @param input_path Path to calibration file
#' @return Calibration parameters
#' @export
load_calibration <- function(input_path) {

  if (!file.exists(input_path)) {
    warning("Calibration file not found: ", input_path)
    return(list(optimal_temperature = 1.0))
  }

  readRDS(input_path)
}

#' Apply calibration temperature to predictions
#'
#' @param logits Raw model logits (matrix)
#' @param temperature Calibration temperature
#' @return Calibrated probabilities
#' @export
apply_calibration <- function(logits, temperature) {
  scaled <- logits / temperature
  # Row-wise softmax
  exp_scaled <- exp(scaled - apply(scaled, 1, max))
  exp_scaled / rowSums(exp_scaled)
}
