# Post-calibration analysis and visualization (V7)
# Deep comparison of confidence before/after temperature scaling

library(tidyverse)
library(patchwork)
library(viridis)

# ===== Theme Constants =====

.calib_colors <- c(Before = "#E53935", After = "#1E88E5")
.outcome_colors <- c(Correct = "#43A047", Incorrect = "#E53935")

.calib_theme <- function(base_size = 12) {

  theme_minimal(base_size = base_size) +
    theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
}

# ===== Main Analysis Function =====

#' Comprehensive calibration comparison analysis
#'
#' Creates visualizations comparing model behavior before/after temperature scaling.
#'
#' @param calib_results Output from calibrate_model()
#' @param output_dir Directory for saving plots
#' @return List with plots, comparison data, and analysis tables
#' @export
analyze_calibration_deep <- function(calib_results, output_dir = "results_transfer") {

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  message("\n", strrep("=", 60))
  message("DEEP CALIBRATION ANALYSIS")
  message(strrep("=", 60))

  # Extract core data
  probs_before <- calib_results$probs_before
  probs_after <- calib_results$probs_after
  labels <- calib_results$labels
  n_samples <- length(labels)
  n_epitopes <- calib_results$n_epitopes
  opt_temp <- calib_results$optimal_temperature

  message(sprintf("\nSamples: %d | Epitopes: %d | T = %.3f", n_samples, n_epitopes, opt_temp))

  # ----- Compute Derived Metrics -----

  pred_before <- apply(probs_before, 1, which.max) - 1
  pred_after <- apply(probs_after, 1, which.max) - 1
  conf_before <- apply(probs_before, 1, max)
  conf_after <- apply(probs_after, 1, max)

  correct_before <- pred_before == labels
  correct_after <- pred_after == labels

  # Probability and rank of true label
  true_prob_before <- sapply(seq_len(n_samples), \(i) probs_before[i, labels[i] + 1])
  true_prob_after <- sapply(seq_len(n_samples), \(i) probs_after[i, labels[i] + 1])

  rank_before <- sapply(seq_len(n_samples), \(i)
                        sum(probs_before[i, ] > probs_before[i, labels[i] + 1]) + 1)
  rank_after <- sapply(seq_len(n_samples), \(i)
                       sum(probs_after[i, ] > probs_after[i, labels[i] + 1]) + 1)

  # Master comparison dataframe
  comparison_df <- tibble(
    sample_id = seq_len(n_samples), true_label = labels,
    pred_before, conf_before, correct_before, true_prob_before, rank_before,
    pred_after, conf_after, correct_after, true_prob_after, rank_after,
    conf_delta = conf_after - conf_before,
    true_prob_delta = true_prob_after - true_prob_before,
    rank_delta = rank_after - rank_before,
    prediction_changed = pred_before != pred_after,
    outcome_changed = correct_before != correct_after
  )

  n_changed <- sum(comparison_df$prediction_changed)
  message(sprintf("\nPrediction changes: %d (%.1f%%)", n_changed, 100 * n_changed / n_samples))
  message(sprintf("Accuracy: %.2f%% → %.2f%%",
                  100 * mean(correct_before), 100 * mean(correct_after)))

  plots <- list()

  # ----- Plot 1: Reliability Diagram -----

  n_bins <- length(calib_results$bin_data_before$bin_acc)
  bin_edges <- calib_results$bin_data_before$bin_edges
  bin_centers <- (bin_edges[-1] + bin_edges[-length(bin_edges)]) / 2

  reliability_df <- tibble(
    bin_center = rep(bin_centers, 2),
    accuracy = c(unlist(calib_results$bin_data_before$bin_acc),
                 unlist(calib_results$bin_data_after$bin_acc)),
    confidence = c(unlist(calib_results$bin_data_before$bin_conf),
                   unlist(calib_results$bin_data_after$bin_conf)),
    count = c(unlist(calib_results$bin_data_before$bin_count),
              unlist(calib_results$bin_data_after$bin_count)),
    stage = factor(rep(c("Before Calibration", "After Calibration"), each = n_bins),
                   levels = c("Before Calibration", "After Calibration"))
  ) %>% filter(!is.na(accuracy), count > 0)

  ece_pct <- 100 * (1 - calib_results$ece_after / calib_results$ece_before)

  plots$reliability <- ggplot(reliability_df, aes(x = confidence, y = accuracy)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40", linewidth = 0.8) +
    geom_ribbon(aes(ymin = pmin(accuracy, confidence), ymax = pmax(accuracy, confidence)),
                alpha = 0.2, fill = "firebrick") +
    geom_line(aes(color = stage), linewidth = 1.2) +
    geom_point(aes(color = stage, size = count), alpha = 0.8) +
    facet_wrap(~stage) +
    scale_color_manual(values = c("Before Calibration" = .calib_colors["Before"],
                                  "After Calibration" = .calib_colors["After"]), guide = "none") +
    scale_size_continuous(range = c(2, 10), labels = scales::comma, name = "Samples") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "Reliability Diagrams: Before vs After Temperature Scaling",
         subtitle = sprintf("ECE: %.3f → %.3f (%.0f%% reduction) | T = %.2f",
                            calib_results$ece_before, calib_results$ece_after, ece_pct, opt_temp),
         x = "Mean Predicted Confidence", y = "Observed Accuracy") +
    .calib_theme() + theme(strip.text = element_text(face = "bold", size = 11))

  # ----- Plot 2: Confidence Density -----

  conf_long <- comparison_df %>%
    select(sample_id, correct_before, conf_before, conf_after) %>%
    pivot_longer(c(conf_before, conf_after), names_to = "stage", values_to = "confidence") %>%
    mutate(stage = factor(ifelse(stage == "conf_before", "Before", "After"),
                          levels = c("Before", "After")),
           outcome = ifelse(correct_before, "Correct", "Incorrect"))

  plots$confidence_density <- ggplot(conf_long, aes(x = confidence, fill = outcome)) +
    geom_density(alpha = 0.6, adjust = 1.5) +
    facet_wrap(~stage, ncol = 1) +
    scale_fill_manual(values = .outcome_colors) +
    scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
    labs(title = "Confidence Distribution Density",
         subtitle = "How temperature scaling reshapes confidence",
         x = "Confidence", y = "Density", fill = "Prediction") +
    .calib_theme() + theme(strip.text = element_text(face = "bold"))

  # ----- Plot 3: Threshold Analysis -----

  thresholds <- seq(0, 0.95, by = 0.05)

  threshold_analysis <- map_dfr(thresholds, function(thresh) {
    before <- comparison_df %>% filter(conf_before >= thresh)
    after <- comparison_df %>% filter(conf_after >= thresh)
    tibble(
      threshold = thresh,
      n_before = nrow(before), n_after = nrow(after),
      acc_before = if (nrow(before) > 0) mean(before$correct_before) else NA_real_,
      acc_after = if (nrow(after) > 0) mean(after$correct_after) else NA_real_,
      coverage_before = nrow(before) / n_samples,
      coverage_after = nrow(after) / n_samples
    )
  })

  # Helper for stage factor conversion
  stage_factor <- function(x, prefix) {
    factor(gsub(paste0(prefix, "_"), "", x),
           levels = c("before", "after"), labels = c("Before", "After"))
  }

  acc_long <- threshold_analysis %>%
    select(threshold, acc_before, acc_after) %>%
    pivot_longer(-threshold, names_to = "stage", values_to = "accuracy") %>%
    mutate(stage = stage_factor(stage, "acc"))

  p_acc <- ggplot(acc_long, aes(x = threshold, y = accuracy, color = stage)) +
    geom_line(linewidth = 1.2) + geom_point(size = 2) +
    geom_hline(yintercept = mean(correct_before), linetype = "dashed", color = "gray50") +
    scale_x_continuous(labels = scales::percent) +
    scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
    scale_color_manual(values = .calib_colors) +
    labs(title = "Accuracy at Confidence Thresholds",
         x = "Min Confidence", y = "Accuracy", color = "Stage") +
    .calib_theme(11)

  cov_long <- threshold_analysis %>%
    select(threshold, coverage_before, coverage_after) %>%
    pivot_longer(-threshold, names_to = "stage", values_to = "coverage") %>%
    mutate(stage = stage_factor(stage, "coverage"))

  p_cov <- ggplot(cov_long, aes(x = threshold, y = coverage, color = stage)) +
    geom_line(linewidth = 1.2) + geom_point(size = 2) +
    scale_x_continuous(labels = scales::percent) +
    scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
    scale_color_manual(values = .calib_colors) +
    labs(title = "Sample Coverage at Thresholds",
         x = "Min Confidence", y = "Fraction of Samples", color = "Stage") +
    .calib_theme(11)

  plots$threshold_analysis <- p_acc + p_cov +
    plot_layout(ncol = 2, guides = "collect") & theme(legend.position = "bottom")

  # ----- Plot 4: Confidence Shift Scatter -----

  plots$conf_change_scatter <- ggplot(comparison_df,
                                      aes(x = conf_before, y = conf_after, color = correct_before)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
    geom_point(alpha = 0.3, size = 0.8) +
    geom_density2d(alpha = 0.5, linewidth = 0.3) +
    scale_color_manual(values = c("TRUE" = .outcome_colors["Correct"],
                                  "FALSE" = .outcome_colors["Incorrect"]),
                       labels = c("TRUE" = "Correct", "FALSE" = "Incorrect")) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "Confidence Shift: Before vs After",
         subtitle = sprintf("T = %.2f > 1 → points below diagonal", opt_temp),
         x = "Confidence Before", y = "Confidence After", color = "Prediction") +
    .calib_theme()

  # ----- Plot 5: True Label Probability -----

  true_prob_long <- comparison_df %>%
    select(sample_id, correct_before, true_prob_before, true_prob_after) %>%
    pivot_longer(c(true_prob_before, true_prob_after), names_to = "stage", values_to = "true_prob") %>%
    mutate(stage = factor(ifelse(stage == "true_prob_before", "Before", "After"),
                          levels = c("Before", "After")),
           outcome = ifelse(correct_before, "Correct", "Incorrect"))

  plots$true_prob_dist <- ggplot(true_prob_long, aes(x = true_prob, fill = stage)) +
    geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
    facet_wrap(~outcome, scales = "free_y", ncol = 1) +
    scale_fill_manual(values = .calib_colors) +
    scale_x_continuous(labels = scales::percent) +
    labs(title = "Probability Assigned to True Label",
         x = "P(True Epitope)", y = "Count", fill = "Stage") +
    .calib_theme() + theme(strip.text = element_text(face = "bold"))

  # ----- Plot 6: Rank Comparison -----

  rank_subset <- comparison_df %>% filter(rank_before <= 20, rank_after <= 20)

  plots$rank_comparison <- ggplot(rank_subset,
                                  aes(x = factor(rank_before), y = factor(rank_after))) +
    geom_bin2d() +
    geom_abline(slope = 1, intercept = 0, color = "white", linetype = "dashed", linewidth = 1) +
    scale_fill_viridis_c(option = "plasma", trans = "log10", name = "Count") +
    labs(title = "Rank of True Label: Before vs After",
         subtitle = "Rank 1 = correct; calibration should preserve ranking",
         x = "Rank Before", y = "Rank After") +
    .calib_theme() + theme(axis.text = element_text(size = 8))

  # ----- Plot 7: ECE Decomposition -----

  ece_decomp <- tibble(
    bin = seq_len(n_bins), bin_center = bin_centers,
    gap_before = abs(unlist(calib_results$bin_data_before$bin_acc) -
                       unlist(calib_results$bin_data_before$bin_conf)),
    gap_after = abs(unlist(calib_results$bin_data_after$bin_acc) -
                      unlist(calib_results$bin_data_after$bin_conf)),
    weight = unlist(calib_results$bin_data_before$bin_count) / n_samples,
    contribution_before = gap_before * weight,
    contribution_after = gap_after * weight
  ) %>% filter(!is.na(gap_before))

  ece_long <- ece_decomp %>%
    select(bin_center, contribution_before, contribution_after) %>%
    pivot_longer(-bin_center, names_to = "stage", values_to = "contribution") %>%
    mutate(stage = stage_factor(stage, "contribution"))

  plots$ece_decomposition <- ggplot(ece_long, aes(x = bin_center, y = contribution, fill = stage)) +
    geom_col(position = "dodge", alpha = 0.8) +
    scale_fill_manual(values = .calib_colors) +
    scale_x_continuous(labels = scales::percent) +
    labs(title = "ECE Decomposition by Confidence Bin",
         x = "Confidence Bin Center", y = "Contribution to ECE", fill = "Stage") +
    .calib_theme()

  # ----- Plot 8: Metrics Summary -----

  metrics_df <- tibble(
    Metric = rep(c("ECE", "MCE", "Brier"), 2),
    Stage = factor(rep(c("Before", "After"), each = 3), levels = c("Before", "After")),
    Value = c(calib_results$ece_before, calib_results$mce_before, calib_results$brier_before,
              calib_results$ece_after, calib_results$mce_after, calib_results$brier_after)
  )

  plots$metrics_comparison <- ggplot(metrics_df, aes(x = Metric, y = Value, fill = Stage)) +
    geom_col(position = "dodge", alpha = 0.85, width = 0.7) +
    geom_text(aes(label = sprintf("%.3f", Value)),
              position = position_dodge(width = 0.7), vjust = -0.5, size = 3.5) +
    scale_fill_manual(values = .calib_colors) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Calibration Metrics: Before vs After",
         subtitle = sprintf("T = %.3f", opt_temp),
         x = NULL, y = "Value (lower is better)", fill = "Stage") +
    .calib_theme()

  # ----- Save Plots -----

  message("\nSaving plots to: ", output_dir)

  ggsave(file.path(output_dir, "calibration_reliability.pdf"),
         plots$reliability, width = 12, height = 6)
  ggsave(file.path(output_dir, "calibration_confidence_density.pdf"),
         plots$confidence_density, width = 8, height = 8)
  ggsave(file.path(output_dir, "calibration_threshold_analysis.pdf"),
         plots$threshold_analysis, width = 14, height = 6)
  ggsave(file.path(output_dir, "calibration_conf_shift.pdf"),
         plots$conf_change_scatter, width = 8, height = 8)
  ggsave(file.path(output_dir, "calibration_true_prob.pdf"),
         plots$true_prob_dist, width = 10, height = 8)
  ggsave(file.path(output_dir, "calibration_rank_comparison.pdf"),
         plots$rank_comparison, width = 8, height = 8)
  ggsave(file.path(output_dir, "calibration_ece_decomposition.pdf"),
         plots$ece_decomposition, width = 10, height = 6)
  ggsave(file.path(output_dir, "calibration_metrics_summary.pdf"),
         plots$metrics_comparison, width = 8, height = 6)

  # Combined dashboard
  dashboard <- plots$reliability /
    plots$threshold_analysis /
    (plots$metrics_comparison + plots$ece_decomposition) +
    plot_layout(heights = c(1, 1, 1))

  ggsave(file.path(output_dir, "calibration_dashboard.pdf"), dashboard, width = 14, height = 16)

  message("Saved 9 plots including dashboard")

  list(plots = plots, comparison_df = comparison_df,
       threshold_analysis = threshold_analysis, ece_decomposition = ece_decomp)
}
