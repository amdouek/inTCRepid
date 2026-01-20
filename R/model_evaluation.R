# Model evaluation utilities: metrics, plotting, and report generation

library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(pROC)
library(patchwork)
library(readr)

# ===== Core Metrics Functions =====

#' Calculate top-k accuracy
#'
#' @param probabilities Matrix of predicted probabilities (samples x classes)
#' @param true_labels Vector of true class indices (0-indexed)
#' @param k Value of k for top-k accuracy
#' @return Top-k accuracy value
#' @export
calc_topk_accuracy <- function(probabilities, true_labels, k = 5) {
  n <- nrow(probabilities)
  correct <- vapply(seq_len(n), function(i) {
    top_k <- order(probabilities[i, ], decreasing = TRUE)[1:k] - 1  # 0-indexed
    true_labels[i] %in% top_k
  }, logical(1))
  mean(correct)
}

#' Calculate per-class metrics (precision, recall, F1)
#'
#' @param predictions Vector of predicted class indices
#' @param true_labels Vector of true class indices
#' @param class_names Optional vector of class names
#' @return Tibble with per-class metrics
#' @export
calc_per_class_metrics <- function(predictions, true_labels, class_names = NULL) {

  classes <- sort(unique(c(predictions, true_labels)))

  metrics <- lapply(classes, function(cls) {
    tp <- sum(predictions == cls & true_labels == cls)
    fp <- sum(predictions == cls & true_labels != cls)
    fn <- sum(predictions != cls & true_labels == cls)

    n_true <- sum(true_labels == cls)
    n_pred <- sum(predictions == cls)

    precision <- if (tp + fp > 0) tp / (tp + fp) else 0
    recall <- if (tp + fn > 0) tp / (tp + fn) else 0
    f1 <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0

    cls_name <- if (!is.null(class_names) && cls + 1 <= length(class_names)) {
      class_names[cls + 1]
    } else {
      as.character(cls)
    }

    tibble(class_idx = cls, class_name = cls_name, n_true = n_true,
           n_predicted = n_pred, true_positives = tp,
           precision = precision, recall = recall, f1 = f1)
  })

  bind_rows(metrics)
}

#' Calculate macro and weighted average metrics
#'
#' @param per_class Per-class metrics from calc_per_class_metrics()
#' @return List with macro and weighted averages
#' @export
calc_aggregate_metrics <- function(per_class) {
  total <- sum(per_class$n_true)

  list(
    macro_precision = mean(per_class$precision, na.rm = TRUE),
    macro_recall = mean(per_class$recall, na.rm = TRUE),
    macro_f1 = mean(per_class$f1, na.rm = TRUE),
    weighted_precision = sum(per_class$precision * per_class$n_true, na.rm = TRUE) / total,
    weighted_recall = sum(per_class$recall * per_class$n_true, na.rm = TRUE) / total,
    weighted_f1 = sum(per_class$f1 * per_class$n_true, na.rm = TRUE) / total
  )
}

# ===== Plotting Functions =====

#' Plot training history
#'
#' @param history Training history list with train_loss, val_loss, train_acc, val_acc
#' @return ggplot object
#' @export
plot_training_history <- function(history) {

  df <- tibble(
    epoch = seq_along(history$train_loss),
    train_loss = unlist(history$train_loss),
    val_loss = unlist(history$val_loss),
    train_acc = unlist(history$train_acc),
    val_acc = unlist(history$val_acc)
  )

  df_long <- df %>%
    pivot_longer(-epoch, names_to = "metric", values_to = "value") %>%
    mutate(
      type = ifelse(str_detect(metric, "train"), "Training", "Validation"),
      metric_type = ifelse(str_detect(metric, "loss"), "Loss", "Accuracy")
    )

  ggplot(df_long, aes(x = epoch, y = value, color = type)) +
    geom_line(linewidth = 1) +
    geom_point(size = 0.5, alpha = 0.5) +
    facet_wrap(~metric_type, scales = "free_y") +
    scale_color_manual(values = c("Training" = "#2E86AB", "Validation" = "#E94F37")) +
    labs(title = "Training History", x = "Epoch", y = "Value", color = "Dataset") +
    theme_minimal() +
    theme(legend.position = "bottom", panel.grid.minor = element_blank())
}

#' Plot confusion matrix for top epitopes
#'
#' @param predictions_df Predictions tibble with true_epitope, predicted_epitope columns
#' @param top_n Number of top epitopes to show
#' @return ggplot object
#' @export
plot_confusion_matrix <- function(predictions_df, top_n = 15) {

  # Get top epitopes by frequency
  top_epitopes <- predictions_df %>%
    count(true_epitope, sort = TRUE) %>%
    head(top_n) %>%
    pull(true_epitope)

  filtered <- predictions_df %>%
    filter(true_epitope %in% top_epitopes, predicted_epitope %in% top_epitopes)

  conf <- as.data.frame(table(
    True = filtered$true_epitope,
    Predicted = filtered$predicted_epitope
  )) %>%
    as_tibble() %>%
    rename(true_epitope = True, predicted_epitope = Predicted, count = Freq) %>%
    group_by(true_epitope) %>%
    mutate(proportion = count / sum(count)) %>%
    ungroup() %>%
    mutate(
      true_short = str_trunc(as.character(true_epitope), 15),
      pred_short = str_trunc(as.character(predicted_epitope), 15)
    )

  ggplot(conf, aes(x = pred_short, y = true_short, fill = proportion)) +
    geom_tile(color = "white") +
    geom_text(aes(label = count), size = 2.5) +
    scale_fill_gradient2(low = "white", mid = "#FED976", high = "#E31A1C",
                         midpoint = 0.5, limits = c(0, 1), name = "Proportion") +
    labs(title = paste("Confusion Matrix (Top", top_n, "Epitopes)"),
         x = "Predicted", y = "True") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
          axis.text.y = element_text(size = 7), panel.grid = element_blank())
}

#' Plot per-class performance metrics
#'
#' @param per_class Per-class metrics tibble
#' @param min_samples Minimum samples per class to include
#' @return ggplot object
#' @export
plot_per_class_performance <- function(per_class, min_samples = 3) {

  df <- per_class %>%
    filter(n_true >= min_samples) %>%
    arrange(desc(f1)) %>%
    select(class_name, precision, recall, f1) %>%
    pivot_longer(-class_name, names_to = "metric", values_to = "value") %>%
    mutate(
      class_short = str_trunc(class_name, 20),
      metric = factor(metric, levels = c("precision", "recall", "f1"),
                      labels = c("Precision", "Recall", "F1"))
    )

  # Order by F1
  class_order <- per_class %>%
    filter(n_true >= min_samples) %>%
    arrange(f1) %>%
    pull(class_name) %>%
    str_trunc(20)

  df$class_short <- factor(df$class_short, levels = class_order)

  ggplot(df, aes(x = class_short, y = value, fill = metric)) +
    geom_col(position = "dodge") +
    coord_flip() +
    scale_fill_manual(values = c("Precision" = "#2E86AB", "Recall" = "#E94F37", "F1" = "#52B788")) +
    labs(title = "Per-Epitope Performance",
         subtitle = paste("Epitopes with ≥", min_samples, "samples"),
         x = "Epitope", y = "Score", fill = "Metric") +
    theme_minimal() +
    theme(legend.position = "bottom", axis.text.y = element_text(size = 7))
}

#' Plot confidence analysis (distribution and calibration)
#'
#' @param predictions_df Predictions tibble with confidence, correct columns
#' @return ggplot object (combined)
#' @export
plot_confidence_analysis <- function(predictions_df) {

  # Confidence distribution
  p1 <- ggplot(predictions_df, aes(x = confidence, fill = correct)) +
    geom_density(alpha = 0.6) +
    scale_fill_manual(values = c("FALSE" = "#E94F37", "TRUE" = "#52B788"),
                      labels = c("Incorrect", "Correct")) +
    labs(title = "Confidence Distribution", x = "Confidence", y = "Density", fill = "Prediction") +
    theme_minimal()

  # Calibration plot
  bins <- predictions_df %>%
    mutate(bin = cut(confidence, breaks = seq(0, 1, 0.1), include.lowest = TRUE)) %>%
    group_by(bin) %>%
    summarise(n = n(), accuracy = mean(correct), mean_conf = mean(confidence), .groups = "drop") %>%
    filter(!is.na(bin))

  p2 <- ggplot(bins, aes(x = mean_conf, y = accuracy)) +
    geom_point(aes(size = n), color = "#2E86AB") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    geom_smooth(method = "loess", se = FALSE, color = "#E94F37") +
    scale_size_continuous(range = c(2, 10)) +
    labs(title = "Calibration Plot", x = "Mean Confidence", y = "Accuracy", size = "n") +
    theme_minimal() +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1))

  p1 + p2 + plot_annotation(title = "Model Confidence Analysis")
}

#' Plot ROC curves (one-vs-rest for multi-class)
#'
#' @param probabilities Probability matrix (samples x classes)
#' @param true_labels True label vector (0-indexed)
#' @param class_names Optional class name vector
#' @param top_n Number of top classes to plot
#' @return List with auc_values tibble and plot
#' @export
plot_roc_curves <- function(probabilities, true_labels, class_names = NULL, top_n = 10) {

  # Top classes by frequency
  top_classes <- sort(table(true_labels), decreasing = TRUE) %>%
    head(top_n) %>%
    names() %>%
    as.integer()

  roc_data <- list()
  auc_vals <- tibble(class_idx = integer(), class_name = character(), auc = numeric())

  for (cls in top_classes) {
    binary <- as.integer(true_labels == cls)
    probs <- probabilities[, cls + 1]

    if (sum(binary) >= 2 && sum(binary) < length(binary)) {
      roc_obj <- roc(binary, probs, quiet = TRUE)

      cls_name <- if (!is.null(class_names) && cls + 1 <= length(class_names)) {
        class_names[cls + 1]
      } else {
        as.character(cls)
      }
      cls_short <- str_trunc(cls_name, 25)

      roc_data[[as.character(cls)]] <- tibble(
        class = cls_short,
        specificity = roc_obj$specificities,
        sensitivity = roc_obj$sensitivities
      )

      auc_vals <- bind_rows(auc_vals, tibble(
        class_idx = cls, class_name = cls_short, auc = as.numeric(auc(roc_obj))
      ))
    }
  }

  roc_df <- bind_rows(roc_data) %>%
    left_join(auc_vals %>% select(class = class_name, auc), by = "class") %>%
    mutate(label = sprintf("%s (AUC=%.3f)", class, auc))

  p <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity, color = label)) +
    geom_line(linewidth = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    scale_color_viridis_d(option = "turbo") +
    labs(title = paste("ROC Curves (Top", top_n, "Epitopes)"),
         x = "False Positive Rate", y = "True Positive Rate", color = "Epitope") +
    theme_minimal() +
    theme(legend.position = "right", legend.text = element_text(size = 7)) +
    coord_fixed()

  list(auc_values = auc_vals, plot = p)
}

# ===== Report Generation =====

#' Generate comprehensive evaluation report
#'
#' @param history Training history
#' @param evaluation Evaluation results list with overall, per_class, predictions, probabilities
#' @param output_dir Directory to save outputs
#' @return List of plots
#' @export
generate_evaluation_report <- function(history, evaluation, output_dir = "results") {

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  message("Generating evaluation report to: ", output_dir)

  # Training history
  p_history <- plot_training_history(history)
  ggsave(file.path(output_dir, "training_history.png"), p_history, width = 10, height = 5, dpi = 150)

  # Confusion matrix
  p_conf <- plot_confusion_matrix(evaluation$predictions, top_n = 15)
  ggsave(file.path(output_dir, "confusion_matrix.png"), p_conf, width = 10, height = 8, dpi = 150)

  # Per-class performance
  p_class <- plot_per_class_performance(evaluation$per_class, min_samples = 2)
  ggsave(file.path(output_dir, "per_class_performance.png"), p_class, width = 10, height = 8, dpi = 150)

  # Confidence analysis
  p_conf_analysis <- plot_confidence_analysis(evaluation$predictions)
  ggsave(file.path(output_dir, "confidence_analysis.png"), p_conf_analysis, width = 12, height = 5, dpi = 150)

  # ROC curves
  roc_result <- plot_roc_curves(
    evaluation$probabilities,
    evaluation$predictions$true_label,
    evaluation$per_class$class_name,
    top_n = 10
  )
  ggsave(file.path(output_dir, "roc_curves.png"), roc_result$plot, width = 10, height = 8, dpi = 150)

  # Save CSVs
  metrics_df <- tibble(
    metric = names(evaluation$overall),
    value = unlist(evaluation$overall)
  )
  write_csv(metrics_df, file.path(output_dir, "metrics_summary.csv"))
  write_csv(evaluation$per_class, file.path(output_dir, "per_class_metrics.csv"))
  write_csv(roc_result$auc_values, file.path(output_dir, "auc_values.csv"))
  write_csv(evaluation$predictions, file.path(output_dir, "test_predictions.csv"))

  message("Report complete.")

  list(history = p_history, confusion = p_conf, per_class = p_class,
       confidence = p_conf_analysis, roc = roc_result$plot, auc_values = roc_result$auc_values)
}

#' Print evaluation summary to console
#'
#' @param evaluation Evaluation results list
#' @export
print_evaluation_summary <- function(evaluation) {

  o <- evaluation$overall

  cat("\n", strrep("=", 60), "\n")
  cat("EVALUATION SUMMARY\n")
  cat(strrep("=", 60), "\n\n")

  cat(sprintf("Test Loss:       %.4f\n", o$test_loss))
  cat(sprintf("Accuracy:        %.1f%%\n", o$accuracy * 100))
  cat(sprintf("Top-5 Accuracy:  %.1f%%\n", o$top5_accuracy * 100))
  cat(sprintf("Top-10 Accuracy: %.1f%%\n", o$top10_accuracy * 100))
  cat(sprintf("Macro F1:        %.4f\n", o$macro_f1))

  if (!is.null(evaluation$by_score)) {
    cat("\nAccuracy by VDJdb Score:\n")
    for (i in seq_len(nrow(evaluation$by_score))) {
      r <- evaluation$by_score[i, ]
      cat(sprintf("  Score %d: %.1f%% (n=%d)\n", r$score, r$accuracy * 100, r$n))
    }
  }

  cat(strrep("=", 60), "\n\n")
}

# ===== Stratified Evaluation Utilities =====

#' Evaluate by score strata
#'
#' @param predictions_df Predictions tibble with score and correct columns
#' @return Tibble with accuracy by score
#' @export
evaluate_by_score <- function(predictions_df) {
  if (!"score" %in% names(predictions_df)) return(NULL)

  predictions_df %>%
    group_by(score) %>%
    summarise(n = n(), accuracy = mean(correct), .groups = "drop")
}

#' Evaluate by chain pairing status (V7)
#'
#' @param predictions_df Predictions tibble with is_paired and correct columns
#' @return Tibble with accuracy by pairing status
#' @export
evaluate_by_pairing <- function(predictions_df) {
  if (!"is_paired" %in% names(predictions_df)) return(NULL)

  predictions_df %>%
    group_by(is_paired) %>%
    summarise(
      n = n(),
      accuracy = mean(correct),
      mean_confidence = mean(confidence, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(status = ifelse(is_paired, "Paired (TRA+TRB)", "TRB only"))
}

#' Evaluate by data source
#'
#' @param predictions_df Predictions tibble with source and correct columns
#' @return Tibble with accuracy by source
#' @export
evaluate_by_source <- function(predictions_df) {
  if (!"source" %in% names(predictions_df)) return(NULL)

  predictions_df %>%
    group_by(source) %>%
    summarise(
      n = n(),
      accuracy = mean(correct),
      mean_confidence = mean(confidence, na.rm = TRUE),
      .groups = "drop"
    )
}

#' Evaluate by species
#' @param predictions Predictions tibble
#' @param test_data Test data with source column
#' @return Tibble with per-species metrics
#' @export
evaluate_by_species <- function(predictions, test_data) {

  src_col <- if ("source" %in% names(test_data)) "source"
  else if ("tcr_species" %in% names(test_data)) "tcr_species"
  else NULL

  if (is.null(src_col)) {
    return(tibble(source = "unknown", n = nrow(predictions),
                  accuracy = mean(predictions$correct, na.rm = TRUE)))
  }

  if (!"source" %in% names(predictions) && nrow(predictions) == nrow(test_data)) {
    predictions$source <- test_data[[src_col]]
  }

  by_species <- predictions %>%
    group_by(source) %>%
    summarise(
      n = n(),
      accuracy = mean(correct, na.rm = TRUE),
      mean_confidence = mean(confidence, na.rm = TRUE),
      .groups = "drop"
    )

  message("\nPerformance by species:")
  for (i in seq_len(nrow(by_species))) {
    message("  ", by_species$source[i], ": ", round(by_species$accuracy[i] * 100, 1), "%")
  }
  by_species
}

#' Evaluate by pairing status
#' @param predictions Predictions tibble with is_paired column
#' @return Tibble with per-pairing metrics
#' @export
evaluate_by_pairing <- function(predictions) {

  if (!"is_paired" %in% names(predictions)) {
    return(NULL)
  }

  predictions %>%
    mutate(status = ifelse(is_paired, "Paired (TRA+TRB)", "Unpaired (TRB only)")) %>%
    group_by(status) %>%
    summarise(
      n = n(),
      accuracy = mean(correct, na.rm = TRUE),
      mean_confidence = mean(confidence, na.rm = TRUE),
      .groups = "drop"
    )
}

#' Evaluate by MHC class
#' @param predictions Predictions tibble with mhc_class column
#' @return Tibble with per-MHC-class metrics
#' @export
evaluate_by_mhc_class <- function(predictions) {

  if (!"mhc_class" %in% names(predictions)) {
    message("No mhc_class column in predictions")
    return(NULL)
  }

  by_mhc <- predictions %>%
    mutate(mhc_class = ifelse(is.na(mhc_class), "Unknown", mhc_class)) %>%
    group_by(mhc_class) %>%
    summarise(
      n = n(),
      accuracy = mean(correct, na.rm = TRUE),
      mean_confidence = mean(confidence, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(n))

  message("\nPerformance by MHC class:")
  for (i in seq_len(nrow(by_mhc))) {
    message(sprintf("  %s: %.1f%% (n=%s)",
                    by_mhc$mhc_class[i],
                    by_mhc$accuracy[i] * 100,
                    format(by_mhc$n[i], big.mark = ",")))
  }

  by_mhc
}
