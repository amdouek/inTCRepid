# VDJdb Data Preprocessing

library(dplyr)
library(tidyr)
library(stringr)
library(purrr)

# ===== Sequence Cleaning =====

#' Clean amino acid sequences
#'
#' @param sequences Character vector of sequences
#' @param min_len Minimum length (default 8)
#' @param max_len Maximum length (default 30)
#' @param allow_x Allow X character (for placeholders)
#' @return Cleaned sequences (invalid → NA)
clean_sequences <- function(sequences, min_len = 8, max_len = 30, allow_x = FALSE) {

  valid_aa <- if (allow_x) "ACDEFGHIKLMNPQRSTVWYX" else "ACDEFGHIKLMNPQRSTVWY"
  invalid_pattern <- paste0("[^", valid_aa, "]")


  cleaned <- str_to_upper(str_trim(sequences))

  # Invalid characters → NA
  has_invalid <- str_detect(cleaned, invalid_pattern)
  if (sum(has_invalid, na.rm = TRUE) > 0) {
    cleaned[has_invalid] <- NA
  }

  # Length filter
  len <- nchar(cleaned)
  cleaned[len < min_len | len > max_len] <- NA

  cleaned
}

#' Clean CDR3 sequences (convenience wrapper)
clean_cdr3_sequences <- function(sequences) {
  clean_sequences(sequences, min_len = 8, max_len = 25)
}

#' Clean epitope sequences (convenience wrapper)
clean_epitope_sequences <- function(sequences) {
  clean_sequences(sequences, min_len = 8, max_len = 30, allow_x = FALSE)
}

# ===== Sample Weighting =====

#' Calculate sample weights based on VDJdb score
#'
#' Score interpretation: 0=low confidence, 3=high confidence
#'
#' @param scores Numeric vector of scores (0-3)
#' @param method Weighting method (see details)
#' @param min_weight Minimum weight for score=0
#' @return Numeric weights
#'
#' @details Methods:
#'   - "aggressive": 16:1 ratio (recommended for training)
#'   - "exponential": Smooth exponential scaling
#'   - "linear": Linear 0.25-1.0 scaling
#'   - "quadratic": Quadratic scaling
calculate_score_weights <- function(scores, method = "aggressive", min_weight = 0.25) {

  scores[is.na(scores)] <- 0

  weights <- switch(method,
                    "aggressive" = c("0" = 0.0625, "1" = 0.125, "2" = 0.5, "3" = 1.0)[as.character(scores)],
                    "exponential" = min_weight + (1 - min_weight) * (1 - exp(-scores)) / (1 - exp(-3)),
                    "linear" = min_weight + (1 - min_weight) * (scores / 3),
                    "quadratic" = min_weight + (1 - min_weight) * (scores / 3)^2,
                    "score3_focused" = c("0" = 0.05, "1" = 0.1, "2" = 0.25, "3" = 1.0)[as.character(scores)],
                    stop("Unknown method: ", method)
  )

  pmax(pmin(as.numeric(weights), 1.0), 0.01)
}


#' Compare weight methods (utility)
#' @param scores Scores to compare (default 0:3)
#' @return Comparison tibble (invisible)
compare_weight_methods <- function(scores = 0:3) {

  methods <- c("aggressive", "exponential", "linear", "quadratic", "score3_focused")

  comparison <- map_dfc(methods, ~tibble(!!.x := calculate_score_weights(scores, .x))) %>%
    mutate(score = scores, .before = 1)

  cat("\nWeight comparison:\n")
  print(mutate(comparison, across(where(is.numeric), ~round(., 3))))

  # Show ratios
  cat("\nScore 3:0 ratios:\n")
  for (m in methods) {
    ratio <- comparison[[m]][4] / comparison[[m]][1]
    cat(" ", m, ": ", round(ratio, 1), ":1\n", sep = "")
  }

  invisible(comparison)
}

# ===== Preprocessing Functions =====

#' Get standard column names from VDJdb data
#' @param df VDJdb data frame
#' @return Named list of column names
get_vdjdb_cols <- function(df) {
  find_col <- function(pattern) {
    matches <- grep(pattern, names(df), ignore.case = TRUE, value = TRUE)
    if (length(matches) > 0) matches[1] else NA_character_
  }
  list(
    cdr3 = find_col("^cdr3$"),
    epitope = find_col("epitope"),
    score = find_col("score"),
    v_gene = find_col("^v\\.?segm"),
    j_gene = find_col("^j\\.?segm"),
    mhc_class = find_col("mhc.*class"),
    antigen_species = find_col("antigen.*species"),
    antigen_gene = find_col("antigen.*gene"),
    gene = find_col("^gene$"),
    species = find_col("^species$")
  )
}


#' Preprocess VDJdb data for model training
#'
#' Standardizes column names, cleans sequences, calculates weights.
#'
#' @param vdjdb_filtered Filtered VDJdb tibble
#' @param species_label Species label ("human" or "mouse")
#' @param weight_method Score weighting method
#' @return Preprocessed tibble
preprocess_vdjdb_for_training <- function(vdjdb_filtered,
                                          species_label,
                                          weight_method = "aggressive") {

  message("Preprocessing ", species_label, " data...")

  cols <- get_vdjdb_cols(vdjdb_filtered)

  # Build standardized dataframe
  processed <- tibble(
    cdr3 = clean_cdr3_sequences(vdjdb_filtered[[cols$cdr3]]),
    epitope = clean_epitope_sequences(vdjdb_filtered[[cols$epitope]]),
    score = as.numeric(vdjdb_filtered[[cols$score]]),
    v_gene = if (!is.na(cols$v_gene)) vdjdb_filtered[[cols$v_gene]] else NA_character_,
    j_gene = if (!is.na(cols$j_gene)) vdjdb_filtered[[cols$j_gene]] else NA_character_,
    mhc_class = if (!is.na(cols$mhc_class)) vdjdb_filtered[[cols$mhc_class]] else NA_character_,
    antigen_species = if (!is.na(cols$antigen_species)) vdjdb_filtered[[cols$antigen_species]] else NA_character_,
    antigen_gene = if (!is.na(cols$antigen_gene)) vdjdb_filtered[[cols$antigen_gene]] else NA_character_,
    tcr_species = species_label
  )

  # Remove missing sequences
  n_before <- nrow(processed)
  processed <- filter(processed, !is.na(cdr3), !is.na(epitope))

  if (nrow(processed) < n_before) {
    message("  Removed ", n_before - nrow(processed), " entries with missing sequences")
  }

  # Add weights and metadata
  processed <- processed %>%
    mutate(
      score = replace_na(score, 0),
      sample_weight = calculate_score_weights(score, method = weight_method),
      cdr3_length = nchar(cdr3),
      epitope_length = nchar(epitope)
    )

  message("  Entries: ", format(nrow(processed), big.mark = ","),
          ", CDR3: ", format(n_distinct(processed$cdr3), big.mark = ","),
          ", Epitopes: ", format(n_distinct(processed$epitope), big.mark = ","))

  processed
}


#' Legacy preprocessing function (calls preprocess_vdjdb_for_training)
#' @inheritParams preprocess_vdjdb_for_training
preprocess_vdjdb <- function(vdjdb_filtered, weight_method = "aggressive") {
  preprocess_vdjdb_for_training(vdjdb_filtered, species_label = "unknown", weight_method)
}

# ===== Dataset Combination =====

#' Combine datasets for training
#'
#' Merges preprocessed datasets, creates unified epitope encoding.
#' Works with VDJdb-only or VDJdb+IEDB data.
#'
#' @param human_data Preprocessed human data
#' @param mouse_data Preprocessed mouse data
#' @param iedb_data Optional curated IEDB data
#' @param min_epitope_samples Minimum samples per epitope
#' @return List with combined data and metadata
combine_datasets <- function(human_data,
                             mouse_data,
                             iedb_data = NULL,
                             min_epitope_samples = 2) {

  message("\n", strrep("=", 60))
  message("COMBINING DATASETS")
  message(strrep("=", 60))

  # Combine VDJdb data
  combined <- bind_rows(
    human_data %>% mutate(source = "VDJdb"),
    mouse_data %>% mutate(source = "VDJdb")
  )

  message("\nVDJdb: ", format(nrow(combined), big.mark = ","),
          " (human: ", format(sum(combined$tcr_species == "human"), big.mark = ","),
          ", mouse: ", format(sum(combined$tcr_species == "mouse"), big.mark = ","), ")")

  # Add IEDB if provided
  if (!is.null(iedb_data) && nrow(iedb_data) > 0) {
    message("IEDB: ", format(nrow(iedb_data), big.mark = ","))

    combined_raw <- bind_rows(combined, iedb_data)

    # Deduplicate (prefer higher score, then VDJdb)
    combined <- combined_raw %>%
      group_by(cdr3, epitope) %>%
      arrange(desc(score), desc(source == "VDJdb")) %>%
      slice(1) %>%
      ungroup()

    message("After deduplication: ", format(nrow(combined), big.mark = ","))
  }

  # Filter by minimum epitope samples
  epitope_counts <- count(combined, epitope) %>% filter(n >= min_epitope_samples)
  n_before <- nrow(combined)
  combined <- filter(combined, epitope %in% epitope_counts$epitope)

  message("After epitope filter (>=", min_epitope_samples, "): ",
          format(nrow(combined), big.mark = ","),
          " (", n_distinct(combined$epitope), " epitopes)")

  # Create epitope index mapping
  unique_epitopes <- unique(combined$epitope)
  epitope_to_idx <- setNames(seq_along(unique_epitopes) - 1L, unique_epitopes)

  combined <- combined %>%
    mutate(
      epitope_idx = epitope_to_idx[epitope],
      sample_weight = calculate_score_weights(score, "aggressive"),
      cdr3_length = nchar(cdr3),
      epitope_length = nchar(epitope),
      cdr3_epitope_pair = paste(cdr3, epitope, sep = "_"),
      entry_id = row_number()
    )

  # Species epitope overlap
  human_ep <- unique(combined$epitope[combined$tcr_species == "human"])
  mouse_ep <- unique(combined$epitope[combined$tcr_species == "mouse"])
  overlap_ep <- intersect(human_ep, mouse_ep)

  message("\nEpitope overlap: ", length(overlap_ep), " shared, ",
          length(setdiff(human_ep, mouse_ep)), " human-only, ",
          length(setdiff(mouse_ep, human_ep)), " mouse-only")

  # Summary
  message("\n", strrep("-", 40))
  message("Final: ", format(nrow(combined), big.mark = ","), " entries")
  message("  Human: ", format(sum(combined$tcr_species == "human"), big.mark = ","))
  message("  Mouse: ", format(sum(combined$tcr_species == "mouse"), big.mark = ","))
  message("  Epitopes: ", length(unique_epitopes))
  message(strrep("-", 40))

  list(
    data = combined,
    epitope_to_idx = epitope_to_idx,
    idx_to_epitope = setNames(names(epitope_to_idx), epitope_to_idx),
    unique_epitopes = unique_epitopes,
    n_human = sum(combined$tcr_species == "human"),
    n_mouse = sum(combined$tcr_species == "mouse"),
    overlap_epitopes = overlap_ep
  )
}


#' Combine all sources (VDJdb + IEDB) - alias for combine_datasets
#'
#' @param vdjdb_human Preprocessed human VDJdb data
#' @param vdjdb_mouse Preprocessed mouse VDJdb data
#' @param iedb_curated Curated IEDB data (optional)
#' @param min_epitope_samples Minimum samples per epitope
#' @return List with combined data and metadata
combine_all_sources <- function(vdjdb_human,
                                vdjdb_mouse,
                                iedb_curated = NULL,
                                min_epitope_samples = 2) {
  combine_datasets(
    human_data = vdjdb_human,
    mouse_data = vdjdb_mouse,
    iedb_data = iedb_curated,
    min_epitope_samples = min_epitope_samples
  )
}

# ===== Example Usage =====
#
# # Load and filter VDJdb (see vdjdb_download.R)
# vdjdb <- load_vdjdb("data/vdjdb.txt")
#
# # Preprocess human and mouse data
# human_processed <- vdjdb %>%
#   filter_vdjdb(tcr_species = "HomoSapiens", chain = "TRB") %>%
#   preprocess_vdjdb_for_training(species_label = "human")
#
# mouse_processed <- vdjdb %>%
#   filter_vdjdb(tcr_species = "MusMusculus", chain = "TRB") %>%
#   preprocess_vdjdb_for_training(species_label = "mouse")
#
# # Combine datasets
# combined <- combine_datasets(human_processed, mouse_processed)
#
# # Or with IEDB
# iedb <- curate_iedb_for_tcr_model(load_iedb_tcr("data/iedb_tcr.csv"))
# combined <- combine_datasets(human_processed, mouse_processed, iedb_data = iedb)
