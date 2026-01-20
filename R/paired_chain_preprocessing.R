# ============================================================================
# Paired Chain (TRA+TRB) Data Preprocessing (V9.1)
# ============================================================================
#
# Handles paired (complex.id != 0) and unpaired entries, with MHC encoding.
#
# V9.1 Changes:
#   - Integrated MHC standardization, vocabulary building, and encoding
#   - Added species-MHC consistency filtering
#   - Updated sample weights to include MHC quality factor
#   - Added mhc_class_idx and mhc_allele_idx to encoded outputs
#
# ============================================================================

library(dplyr)
library(tidyr)
library(data.table)

# Source dependencies
if (!exists("sequences_to_indices")) source("R/sequence_encoding.R")
if (!exists("encode_vj_for_dataset")) source("R/vj_gene_encoding.R")
if (!exists("standardize_mhc_allele")) source("R/mhc_encoding.R")

# ===== Constants =====

MISSING_CDR3_PLACEHOLDER <- "XXXXXXXXXX"  # Encodes to UNK tokens

# ===== Extract Paired Chain Data =====

#' Extract paired TRA+TRB chain data from VDJdb
#'
#' Groups entries by complex.id to identify paired chains, extracting
#' CDR3, V gene, and J gene for both alpha and beta chains.
#'
#' @param vdjdb_raw Raw VDJdb data
#' @param require_both_chains If TRUE, only return complexes with both chains
#' @return Tibble with one row per unique TCR (paired or unpaired)
#' @export
extract_paired_chains <- function(vdjdb_raw, require_both_chains = FALSE) {

  message("\nExtracting paired chain data...")

  # Separate by complex.id presence
  has_complex <- !is.na(vdjdb_raw$complex.id) & vdjdb_raw$complex.id != 0 & vdjdb_raw$complex.id != ""
  paired_entries <- vdjdb_raw[has_complex, ]
  unpaired_entries <- vdjdb_raw[!has_complex, ]

  message(sprintf("  With complex.id: %s | Without: %s",
                  format(nrow(paired_entries), big.mark = ","),
                  format(nrow(unpaired_entries), big.mark = ",")))

  # ----- Process paired entries -----
  # Pivot to wide format: one row per complex with TRA/TRB columns

  paired_wide <- paired_entries %>%
    select(complex.id, gene, cdr3, v.segm, j.segm, species, antigen.epitope,
           antigen.species, antigen.gene, mhc.class, mhc.a, mhc.b, vdjdb.score) %>%
    group_by(complex.id, gene) %>%
    slice(1) %>%  # Deduplicate
    ungroup() %>%
    pivot_wider(
      id_cols = c(complex.id, species, antigen.epitope, antigen.species,
                  antigen.gene, mhc.class, mhc.a, mhc.b, vdjdb.score),
      names_from = gene,
      values_from = c(cdr3, v.segm, j.segm),
      names_glue = "{.value}_{gene}"
    ) %>%
    rename(
      cdr3_alpha = cdr3_TRA, v_alpha = v.segm_TRA, j_alpha = j.segm_TRA,
      cdr3_beta = cdr3_TRB, v_beta = v.segm_TRB, j_beta = j.segm_TRB,
      epitope = antigen.epitope, score = vdjdb.score
    ) %>%
    mutate(
      has_alpha = !is.na(cdr3_alpha) & cdr3_alpha != "",
      has_beta = !is.na(cdr3_beta) & cdr3_beta != "",
      is_paired = has_alpha & has_beta,
      source_type = "paired"
    )

  n_both <- sum(paired_wide$is_paired)
  n_alpha_only <- sum(paired_wide$has_alpha & !paired_wide$has_beta)
  n_beta_only <- sum(!paired_wide$has_alpha & paired_wide$has_beta)

  message(sprintf("  Paired breakdown: TRA+TRB=%s, TRA-only=%s, TRB-only=%s",
                  format(n_both, big.mark = ","),
                  format(n_alpha_only, big.mark = ","),
                  format(n_beta_only, big.mark = ",")))

  # ----- Process unpaired TRB entries -----

  unpaired_trb <- unpaired_entries %>%
    filter(gene == "TRB") %>%
    select(cdr3, v.segm, j.segm, species, antigen.epitope, antigen.species,
           antigen.gene, mhc.class, mhc.a, mhc.b, vdjdb.score) %>%
    rename(cdr3_beta = cdr3, v_beta = v.segm, j_beta = j.segm,
           epitope = antigen.epitope, score = vdjdb.score) %>%
    mutate(
      complex.id = NA_integer_,
      cdr3_alpha = NA_character_, v_alpha = NA_character_, j_alpha = NA_character_,
      has_alpha = FALSE, has_beta = TRUE, is_paired = FALSE, source_type = "unpaired"
    )

  message(sprintf("  Unpaired TRB: %s", format(nrow(unpaired_trb), big.mark = ",")))

  # ----- Combine -----

  paired_cols <- c("complex.id", "cdr3_alpha", "v_alpha", "j_alpha",
                   "cdr3_beta", "v_beta", "j_beta", "epitope", "species",
                   "antigen.species", "antigen.gene", "mhc.class", "mhc.a", "mhc.b",
                   "score", "has_alpha", "has_beta", "is_paired", "source_type")

  if (require_both_chains) {
    combined <- paired_wide %>% filter(is_paired) %>% select(all_of(paired_cols))
    message(sprintf("\n  Returning paired-only: %s entries", format(nrow(combined), big.mark = ",")))
  } else {
    combined <- bind_rows(
      paired_wide %>% select(all_of(paired_cols)),
      unpaired_trb %>% select(all_of(paired_cols))
    )
    message(sprintf("\n  Combined: %s (paired: %s, TRB-only: %s)",
                    format(nrow(combined), big.mark = ","),
                    format(sum(combined$is_paired), big.mark = ","),
                    format(sum(!combined$is_paired), big.mark = ",")))
  }

  combined
}

#' Fill missing chain values with placeholder tokens
#'
#' @param paired_data Output from extract_paired_chains()
#' @param tra_vocab TRA vocabulary (for reference)
#' @return Data with missing values filled
#' @export
fill_missing_chains <- function(paired_data, tra_vocab) {

  message("\nFilling missing chain values...")
  n_missing <- sum(!paired_data$has_alpha)

  # Helper for NA/empty replacement
  fill_na <- function(x, replacement) ifelse(is.na(x) | x == "", replacement, x)

  filled <- paired_data %>%
    mutate(
      cdr3_alpha = fill_na(cdr3_alpha, MISSING_CDR3_PLACEHOLDER),
      cdr3_beta = fill_na(cdr3_beta, MISSING_CDR3_PLACEHOLDER),
      v_alpha = fill_na(v_alpha, ""), j_alpha = fill_na(j_alpha, ""),
      v_beta = fill_na(v_beta, ""), j_beta = fill_na(j_beta, "")
    )

  message(sprintf("  Filled %s missing TRA chains", format(n_missing, big.mark = ",")))
  filled
}

# ===== Quality Filtering =====

#' Apply quality filters to paired chain data
#'
#' @param paired_data Output from extract/fill functions
#' @param min_cdr3_len Minimum CDR3 length
#' @param max_cdr3_len Maximum CDR3 length
#' @param min_epitope_len Minimum epitope length
#' @param max_epitope_len Maximum epitope length
#' @param valid_aa_pattern Regex for valid amino acids
#' @return Filtered data
#' @export
filter_paired_chains <- function(paired_data,
                                 min_cdr3_len = 8, max_cdr3_len = 25,
                                 min_epitope_len = 8, max_epitope_len = 30,
                                 valid_aa_pattern = "^[ACDEFGHIKLMNPQRSTVWYX]+$") {

  message("\nApplying quality filters...")
  n_before <- nrow(paired_data)

  # Length filter helper
  len_ok <- function(x, min_l, max_l) nchar(x) >= min_l & nchar(x) <= max_l

  filtered <- paired_data %>%
    filter(
      len_ok(cdr3_beta, min_cdr3_len, max_cdr3_len),
      grepl(valid_aa_pattern, cdr3_beta),
      len_ok(cdr3_alpha, min_cdr3_len, max_cdr3_len),
      grepl(valid_aa_pattern, cdr3_alpha),
      !is.na(epitope), epitope != "",
      len_ok(epitope, min_epitope_len, max_epitope_len),
      grepl("^[ACDEFGHIKLMNPQRSTVWY]+$", epitope)  # No X in epitopes
    )

  n_removed <- n_before - nrow(filtered)
  message(sprintf("  %s → %s (removed %s, %.1f%%)",
                  format(n_before, big.mark = ","), format(nrow(filtered), big.mark = ","),
                  format(n_removed, big.mark = ","), 100 * n_removed / n_before))
  filtered
}

# ===== Combine Multiple Data Sources =====

#' Combine paired chain data from all sources (VDJdb, IEDB, McPAS-TCR)
#'
#' @param vdjdb_paired Output from extract_paired_chains()
#' @param iedb_paired Output from curate_iedb_for_paired_chains() (optional)
#' @param mcpas_paired Output from curate_mcpas_for_paired_chains() (optional)
#' @param deduplicate If TRUE, remove duplicates
#' @return Combined tibble
#' @export
combine_all_sources <- function(vdjdb_paired,
                                iedb_paired = NULL,
                                mcpas_paired = NULL,
                                deduplicate = TRUE) {

  message("\n", strrep("=", 60))
  message("COMBINING ALL DATA SOURCES")
  message(strrep("=", 60))

  required_cols <- c("complex.id", "cdr3_alpha", "v_alpha", "j_alpha",
                     "cdr3_beta", "v_beta", "j_beta", "epitope", "species",
                     "antigen.species", "antigen.gene", "mhc.class", "mhc.a", "mhc.b",
                     "score", "has_alpha", "has_beta", "is_paired", "source_type")

  # ----- VDJdb -----
  message("\nVDJdb: ", format(nrow(vdjdb_paired), big.mark = ","), " entries")
  message("  Paired: ", format(sum(vdjdb_paired$is_paired), big.mark = ","))
  message("  TRB-only: ", format(sum(!vdjdb_paired$is_paired), big.mark = ","))

  all_sources <- list(vdjdb_paired %>% select(any_of(required_cols)))

  # ----- IEDB -----
  if (!is.null(iedb_paired) && nrow(iedb_paired) > 0) {
    message("\nIEDB: ", format(nrow(iedb_paired), big.mark = ","), " entries")
    message("  Paired: ", format(sum(iedb_paired$is_paired), big.mark = ","))
    message("  TRB-only: ", format(sum(!iedb_paired$is_paired), big.mark = ","))
    all_sources <- c(all_sources, list(iedb_paired %>% select(any_of(required_cols))))
  }

  # ----- McPAS-TCR -----
  if (!is.null(mcpas_paired) && nrow(mcpas_paired) > 0) {
    message("\nMcPAS-TCR: ", format(nrow(mcpas_paired), big.mark = ","), " entries")
    message("  Paired: ", format(sum(mcpas_paired$is_paired), big.mark = ","))
    message("  TRB-only: ", format(sum(!mcpas_paired$is_paired), big.mark = ","))
    all_sources <- c(all_sources, list(mcpas_paired %>% select(any_of(required_cols))))
  }

  # ----- Combine -----
  combined <- bind_rows(all_sources)
  message("\nCombined (raw): ", format(nrow(combined), big.mark = ","))

  # ----- Deduplicate -----
  if (deduplicate) {
    n_before <- nrow(combined)

    combined <- combined %>%
      mutate(
        dedup_key = paste(
          ifelse(is.na(cdr3_alpha), "", cdr3_alpha),
          cdr3_beta,
          epitope,
          sep = "_"
        )
      ) %>%
      # Priority: VDJdb > McPAS > IEDB, then by score
      group_by(dedup_key) %>%
      arrange(
        desc(source_type %in% c("paired", "unpaired")),  # VDJdb first
        desc(source_type == "McPAS"),                     # Then McPAS
        desc(score)                                       # Then by score
      ) %>%
      slice(1) %>%
      ungroup() %>%
      select(-dedup_key)

    n_removed <- n_before - nrow(combined)
    message("After deduplication: ", format(nrow(combined), big.mark = ","),
            " (removed ", format(n_removed, big.mark = ","), " duplicates)")
  }

  # ----- Summary -----
  message("\n", strrep("-", 40))
  message("Combined data summary:")
  message("  Total: ", format(nrow(combined), big.mark = ","))
  message("  Paired (TRA+TRB): ", format(sum(combined$is_paired), big.mark = ","))
  message("  TRB-only: ", format(sum(!combined$is_paired), big.mark = ","))
  message("  Unique epitopes: ", format(n_distinct(combined$epitope), big.mark = ","))

  message("\nBy source:")
  print(combined %>% count(source_type) %>% arrange(desc(n)))

  message("\nBy species:")
  print(combined %>% count(species) %>% arrange(desc(n)))

  message(strrep("-", 40))

  combined
}

#' Prepare combined data for V9.1 model training
#'
#' Complete pipeline: load sources, extract pairs, combine, filter, encode, split.
#' V9.1: Includes MHC standardization, vocabulary building, and encoding.
#'
#' @param vdjdb_raw Raw VDJdb data
#' @param iedb_raw Raw IEDB data (optional)
#' @param mcpas_raw Raw McPAS-TCR data (optional)
#' @param trb_vocab TRB V/J vocabulary
#' @param tra_vocab TRA V/J vocabulary
#' @param include_unpaired Include unpaired TRB entries
#' @param iedb_filter Filter for IEDB entries: "all", "paired_only", "high_quality"
#' @param mhc_min_allele_freq Minimum frequency for MHC allele vocabulary (default: 10)
#' @param test_fraction Test set fraction
#' @param validation_fraction Validation set fraction
#' @param seed Random seed
#' @return List with train/val/test splits, vocabularies, and metadata
#' @export
prepare_combined_data <- function(vdjdb_raw,
                                  iedb_raw = NULL,
                                  mcpas_raw = NULL,
                                  trb_vocab,
                                  tra_vocab,
                                  include_unpaired = TRUE,
                                  iedb_filter = "all",
                                  mhc_min_allele_freq = 10,
                                  test_fraction = 0.15,
                                  validation_fraction = 0.15,
                                  seed = 42) {

  set.seed(seed)

  message("\n", strrep("=", 70))
  message("PREPARING COMBINED DATA FOR V9.1")
  message(strrep("=", 70))

  # ----- Step 1: Extract paired chains from VDJdb -----

  message("\n--- VDJdb Processing ---")
  vdjdb_paired <- extract_paired_chains(vdjdb_raw, require_both_chains = FALSE)

  # ----- Step 2: Extract paired chains from IEDB (if provided) -----

  iedb_paired <- NULL
  if (!is.null(iedb_raw) && nrow(iedb_raw) > 0) {
    message("\n--- IEDB Processing ---")
    iedb_paired <- curate_iedb_for_paired_chains(iedb_raw)

    # Apply IEDB filtering if requested
    if (iedb_filter != "all" && !is.null(iedb_paired) && nrow(iedb_paired) > 0) {
      n_before <- nrow(iedb_paired)

      iedb_paired <- switch(iedb_filter,
                            "paired_only" = {
                              message("\n  Filtering IEDB to paired chains only...")
                              iedb_paired %>% filter(is_paired == TRUE)
                            },
                            "high_quality" = {
                              message("\n  Filtering IEDB to high-quality entries...")
                              iedb_paired %>%
                                filter(is_paired == TRUE |
                                         (!is.na(v_beta) & v_beta != "" &
                                            !is.na(j_beta) & j_beta != ""))
                            },
                            iedb_paired
      )

      n_after <- nrow(iedb_paired)
      message(sprintf("  IEDB filtered: %s → %s (removed %s, %.1f%%)",
                      format(n_before, big.mark = ","),
                      format(n_after, big.mark = ","),
                      format(n_before - n_after, big.mark = ","),
                      100 * (n_before - n_after) / n_before))
    }
  }

  # ----- Step 3: Extract paired chains from McPAS-TCR (if provided) -----

  mcpas_paired <- NULL
  if (!is.null(mcpas_raw) && nrow(mcpas_raw) > 0) {
    message("\n--- McPAS-TCR Processing ---")
    mcpas_paired <- curate_mcpas_for_paired_chains(mcpas_raw)
  }

  # ----- Step 4: Combine all sources -----

  combined <- combine_all_sources(
    vdjdb_paired = vdjdb_paired,
    iedb_paired = iedb_paired,
    mcpas_paired = mcpas_paired,
    deduplicate = TRUE
  )

  # ----- Step 5: Filter unpaired if requested -----

  if (!include_unpaired) {
    combined <- combined %>% filter(is_paired)
    message("\nFiltered to paired-only: ", format(nrow(combined), big.mark = ","))
  }

  # ----- Step 6: Fill missing chains and apply quality filters -----

  paired_data <- combined %>%
    fill_missing_chains(tra_vocab) %>%
    filter_paired_chains()

  # ----- Step 7: Species standardization -----

  paired_data <- paired_data %>%
    mutate(
      tcr_species = case_when(
        species == "HomoSapiens" ~ "human",
        species == "MusMusculus" ~ "mouse",
        TRUE ~ "other"
      )
    ) %>%
    filter(tcr_species %in% c("human", "mouse"))

  message(sprintf("\n  Species: Human=%s, Mouse=%s",
                  format(sum(paired_data$tcr_species == "human"), big.mark = ","),
                  format(sum(paired_data$tcr_species == "mouse"), big.mark = ",")))

  # =========================================================================
  # V9.1: MHC PROCESSING
  # =========================================================================

  message("\n", strrep("-", 60))
  message("V9.1: MHC PROCESSING")
  message(strrep("-", 60))

  # ----- Step 8: Standardize MHC alleles -----

  message("\nStandardizing MHC alleles...")
  n_before_mhc <- nrow(paired_data)

  paired_data <- paired_data %>%
    mutate(
      mhc_allele_std = standardize_mhc_allele(mhc.a)
    )

  # Summary of standardization
  n_class_only <- sum(paired_data$mhc_allele_std == "<CLASS_ONLY>", na.rm = TRUE)
  n_missing <- sum(is.na(paired_data$mhc_allele_std) | paired_data$mhc_allele_std == "")
  n_specific <- n_before_mhc - n_class_only - n_missing

  message(sprintf("  Specific allele: %s (%.1f%%)",
                  format(n_specific, big.mark = ","),
                  100 * n_specific / n_before_mhc))
  message(sprintf("  Class-only: %s (%.1f%%)",
                  format(n_class_only, big.mark = ","),
                  100 * n_class_only / n_before_mhc))
  message(sprintf("  Missing: %s (%.1f%%)",
                  format(n_missing, big.mark = ","),
                  100 * n_missing / n_before_mhc))

  # ----- Step 9: Filter species-MHC mismatches -----

  message("\nChecking species-MHC consistency...")

  paired_data <- paired_data %>%
    mutate(
      mhc_species_consistent = check_mhc_species_consistency(mhc_allele_std, tcr_species)
    )

  n_inconsistent <- sum(!paired_data$mhc_species_consistent)

  if (n_inconsistent > 0) {
    # Report mismatches before filtering
    mismatch_summary <- paired_data %>%
      filter(!mhc_species_consistent) %>%
      count(tcr_species, mhc_allele_std, name = "n") %>%
      arrange(desc(n)) %>%
      head(10)

    message(sprintf("  Found %s species-MHC mismatches:", format(n_inconsistent, big.mark = ",")))
    for (i in seq_len(nrow(mismatch_summary))) {
      message(sprintf("    %s + %s: %s",
                      mismatch_summary$tcr_species[i],
                      mismatch_summary$mhc_allele_std[i],
                      format(mismatch_summary$n[i], big.mark = ",")))
    }

    # Filter
    paired_data <- paired_data %>%
      filter(mhc_species_consistent) %>%
      select(-mhc_species_consistent)

    message(sprintf("  After filtering: %s entries", format(nrow(paired_data), big.mark = ",")))
  } else {
    paired_data <- paired_data %>% select(-mhc_species_consistent)
    message("  No species-MHC mismatches found.")
  }

  # ----- Step 10: Build MHC vocabulary -----

  message("\nBuilding MHC vocabulary...")
  mhc_vocab <- build_mhc_vocabulary(paired_data, min_allele_freq = mhc_min_allele_freq)

  # ----- Step 11: Infer MHC class -----

  paired_data <- paired_data %>%
    mutate(
      mhc_class_inferred = infer_mhc_class(mhc_allele_std, mhc.class)
    )

  # ----- Step 12: Encode MHC class -----

  class_token_to_idx <- mhc_vocab$class$token_to_idx

  paired_data <- paired_data %>%
    mutate(
      mhc_class_idx = case_when(
        is.na(mhc_class_inferred) ~ class_token_to_idx[["UNK"]],
        mhc_class_inferred == "MHCI" ~ class_token_to_idx[["MHCI"]],
        mhc_class_inferred == "MHCII" ~ class_token_to_idx[["MHCII"]],
        TRUE ~ class_token_to_idx[["UNK"]]
      )
    )

  # ----- Step 13: Encode MHC allele -----

  allele_token_to_idx <- mhc_vocab$allele$token_to_idx

  paired_data <- paired_data %>%
    mutate(
      mhc_allele_idx = case_when(
        # Missing allele
        is.na(mhc_allele_std) | mhc_allele_std == "" ~
          as.integer(allele_token_to_idx[["MISSING"]]),

        # Class-only placeholder
        mhc_allele_std == "<CLASS_ONLY>" ~
          as.integer(allele_token_to_idx[["CLASS_ONLY"]]),

        # Known allele in vocabulary
        mhc_allele_std %in% names(allele_token_to_idx) ~
          as.integer(allele_token_to_idx[mhc_allele_std]),

        # Unknown/rare allele
        TRUE ~ as.integer(allele_token_to_idx[["UNK"]])
      )
    )

  # ----- Step 14: Calculate MHC quality weight -----

  paired_data <- paired_data %>%
    mutate(
      mhc_weight = 1.0
      #case_when(
        # Class-only gets reduced weight
        #mhc_allele_idx == allele_token_to_idx[["CLASS_ONLY"]] ~ 0.5,
        # Missing gets reduced weight
        #mhc_allele_idx == allele_token_to_idx[["MISSING"]] ~ 0.5,
        # Everything else gets full weight
        #TRUE ~ 1.0
      )
    #)

  # MHC encoding summary
  message("\n  MHC encoding summary:")
  mhc_class_dist <- paired_data %>%
    count(mhc_class_inferred, name = "n") %>%
    mutate(pct = round(100 * n / sum(n), 1))
  message("    Class: MHCI=", sum(paired_data$mhc_class_inferred == "MHCI", na.rm = TRUE),
          ", MHCII=", sum(paired_data$mhc_class_inferred == "MHCII", na.rm = TRUE),
          ", UNK=", sum(is.na(paired_data$mhc_class_inferred)))

  n_specific_allele <- sum(!paired_data$mhc_allele_idx %in%
                             c(allele_token_to_idx[["UNK"]],
                               allele_token_to_idx[["MISSING"]],
                               allele_token_to_idx[["CLASS_ONLY"]]))
  message(sprintf("    Specific alleles: %s (%.1f%%)",
                  format(n_specific_allele, big.mark = ","),
                  100 * n_specific_allele / nrow(paired_data)))

  message(strrep("-", 60))

  # =========================================================================
  # END V9.1 MHC PROCESSING
  # =========================================================================

  # ----- Step 15: Epitope indexing -----

  unique_epitopes <- sort(unique(paired_data$epitope))
  epitope_to_idx <- setNames(seq_along(unique_epitopes) - 1L, unique_epitopes)
  idx_to_epitope <- setNames(unique_epitopes, as.character(seq_along(unique_epitopes) - 1L))

  paired_data <- paired_data %>%
    mutate(epitope_idx = epitope_to_idx[epitope])

  message("\n  Unique epitopes: ", length(unique_epitopes))

  # ----- Step 15b: Compute representative MHC per unique epitope -----

  message("\n  Computing representative MHC per epitope...")

  epitope_mhc_rep <- paired_data %>%
    group_by(epitope) %>%
    summarise(
      # Most common MHC class for this epitope
      rep_mhc_class_idx = {
        counts <- table(mhc_class_idx)
        as.integer(names(counts)[which.max(counts)])
      },
      # Most common MHC allele for this epitope
      rep_mhc_allele_idx = {
        counts <- table(mhc_allele_idx)
        as.integer(names(counts)[which.max(counts)])
      },
      .groups = "drop"
    )

  # Align with unique_epitopes ordering
  unique_epitope_mhc_class <- as.integer(
    epitope_mhc_rep$rep_mhc_class_idx[match(unique_epitopes, epitope_mhc_rep$epitope)]
  )
  unique_epitope_mhc_allele <- as.integer(
    epitope_mhc_rep$rep_mhc_allele_idx[match(unique_epitopes, epitope_mhc_rep$epitope)]
  )

  message(sprintf("    Epitopes with specific MHC class: %s/%s",
                  sum(unique_epitope_mhc_class != class_token_to_idx[["UNK"]]),
                  length(unique_epitopes)))

  # ----- Step 16: Sample weights (score × MHC quality) -----

  paired_data <- paired_data %>%
    mutate(
      score_weight = 2^(2 * score),
      sample_weight = score_weight * mhc_weight,  # Combined weight
      entry_id = paste0(source_type, "_", row_number())
    )

  # Weight distribution summary
  message("\n  Sample weight distribution:")
  message(sprintf("    Score-only range: %.2f - %.2f",
                  min(paired_data$score_weight), max(paired_data$score_weight)))
  message(sprintf("    Combined range: %.2f - %.2f",
                  min(paired_data$sample_weight), max(paired_data$sample_weight)))
  message(sprintf("    Entries with MHC penalty (0.5×): %s",
                  format(sum(paired_data$mhc_weight < 1), big.mark = ",")))

  # ----- Step 17: Add source column for evaluation -----

  paired_data <- paired_data %>%
    mutate(source = tcr_species)  # 'source' used by evaluate_by_species()

  message(sprintf("\n  Pairing: TRA+TRB=%s, TRB-only=%s",
                  format(sum(paired_data$is_paired), big.mark = ","),
                  format(sum(!paired_data$is_paired), big.mark = ",")))
  message(sprintf("  Data source: VDJdb=%s, IEDB=%s, McPAS=%s",
                  format(sum(paired_data$source_type %in% c("paired", "unpaired")), big.mark = ","),
                  format(sum(paired_data$source_type == "IEDB"), big.mark = ","),
                  format(sum(paired_data$source_type == "McPAS"), big.mark = ",")))

  # ----- Step 18: Stratified splits -----

  message("\n  Creating stratified splits...")

  paired_data <- paired_data %>%
    mutate(strat_var = paste(source,
                             ifelse(is_paired, "paired", "unpaired"),
                             ifelse(score >= 2, "high", "low"),
                             source_type,
                             sep = "_"))

  # Separate splittable from rare epitopes
  epitope_counts <- count(paired_data, epitope)
  splittable <- epitope_counts$epitope[epitope_counts$n >= 3]

  data_split <- paired_data %>% filter(epitope %in% splittable)
  data_rare <- paired_data %>% filter(!epitope %in% splittable)

  test_idx <- caret::createDataPartition(data_split$strat_var, p = test_fraction, list = FALSE)
  test_data <- data_split[test_idx, ]
  train_val <- data_split[-test_idx, ]

  val_idx <- caret::createDataPartition(train_val$strat_var,
                                        p = validation_fraction / (1 - test_fraction),
                                        list = FALSE)
  val_data <- train_val[val_idx, ]
  train_data <- bind_rows(train_val[-val_idx, ], data_rare)

  message(sprintf("    Train: %s | Val: %s | Test: %s",
                  format(nrow(train_data), big.mark = ","),
                  format(nrow(val_data), big.mark = ","),
                  format(nrow(test_data), big.mark = ",")))

  # ----- Step 19: Encode sequences -----

  message("\n  Encoding sequences...")

  cdr3_max_len <- 25L
  epitope_max_len <- 30L

  if (!exists("sequences_to_indices")) source("R/sequence_encoding.R")

  encode_split <- function(data, name) {
    message("    ", name, "...")

    cdr3_alpha_idx <- sequences_to_indices(data$cdr3_alpha, cdr3_max_len)
    cdr3_beta_idx <- sequences_to_indices(data$cdr3_beta, cdr3_max_len)
    epitope_idx <- sequences_to_indices(data$epitope, epitope_max_len)

    # V/J genes - TRB
    trb_data <- data %>% rename(v.segm = v_beta, j.segm = j_beta)
    trb_data <- encode_vj_for_dataset(trb_data, trb_vocab)
    v_beta_idx <- trb_data$v_idx
    j_beta_idx <- trb_data$j_idx

    # V/J genes - TRA
    tra_data <- data %>% rename(v.segm = v_alpha, j.segm = j_alpha)
    tra_data <- encode_vj_for_dataset(tra_data, tra_vocab)
    v_alpha_idx <- tra_data$v_idx
    j_alpha_idx <- tra_data$j_idx

    list(
      data = data,
      cdr3_alpha_idx = cdr3_alpha_idx,
      cdr3_beta_idx = cdr3_beta_idx,
      epitope_idx = epitope_idx,
      v_alpha_idx = as.integer(v_alpha_idx),
      j_alpha_idx = as.integer(j_alpha_idx),
      v_beta_idx = as.integer(v_beta_idx),
      j_beta_idx = as.integer(j_beta_idx),
      labels = as.integer(data$epitope_idx),
      weights = as.numeric(data$sample_weight),
      # V9.1: Add MHC indices
      mhc_class_idx = as.integer(data$mhc_class_idx),
      mhc_allele_idx = as.integer(data$mhc_allele_idx)
    )
  }

  train_enc <- encode_split(train_data, "Training")
  val_enc <- encode_split(val_data, "Validation")
  test_enc <- encode_split(test_data, "Test")

  unique_epitope_idx <- sequences_to_indices(unique_epitopes, epitope_max_len)

  message("\n", strrep("=", 70))
  message("COMBINED DATA PREPARATION COMPLETE (V9.1)")
  message(strrep("=", 70))

  list(
    train = train_enc,
    validation = val_enc,
    test = test_enc,
    epitope_to_idx = epitope_to_idx,
    idx_to_epitope = idx_to_epitope,
    unique_epitopes = unique_epitopes,
    unique_epitope_idx = unique_epitope_idx,
    unique_epitope_mhc_class = unique_epitope_mhc_class,
    unique_epitope_mhc_allele = unique_epitope_mhc_allele,
    trb_vocab = trb_vocab,
    tra_vocab = tra_vocab,
    mhc_vocab = mhc_vocab,  # V9.1: Add MHC vocabulary
    metadata = list(
      n_train = nrow(train_data),
      n_val = nrow(val_data),
      n_test = nrow(test_data),
      n_epitopes = length(unique_epitopes),
      n_paired = sum(paired_data$is_paired),
      n_unpaired = sum(!paired_data$is_paired),
      n_vdjdb = sum(paired_data$source_type %in% c("paired", "unpaired")),
      n_iedb = sum(paired_data$source_type == "IEDB"),
      n_mcpas = sum(paired_data$source_type == "McPAS"),
      include_unpaired = include_unpaired,
      # V9.1: Add MHC metadata
      mhc_class_vocab_size = mhc_vocab$class$size,
      mhc_allele_vocab_size = mhc_vocab$allele$size,
      mhc_min_allele_freq = mhc_min_allele_freq,
      n_with_specific_allele = n_specific_allele,
      n_with_mhc_penalty = sum(paired_data$mhc_weight < 1)
    )
  )
}


# ===== Phase 2 Splits =====

#' Create Phase 2 splits for mouse fine-tuning with experience replay
#'
#' @param data_splits Output from prepare_combined_data()
#' @param replay_fraction Fraction of human data to include (default: 0.03)
#' @param replay_stratified Use stratified sampling for replay
#' @param seed Random seed
#' @return List with Phase 2 train/validation/test splits
#' @export
create_phase2_splits <- function(data_splits,
                                 replay_fraction = 0.03,
                                 replay_stratified = TRUE,
                                 seed = 42) {

  set.seed(seed)

  message("\n", strrep("=", 60))
  message("CREATING PHASE 2 SPLITS (MOUSE FINE-TUNING)")
  message(strrep("=", 60))

  train_data <- data_splits$train$data
  val_data <- data_splits$validation$data
  test_data <- data_splits$test$data

  # Split by species
  train_mouse <- train_data %>% filter(source == "mouse")
  train_human <- train_data %>% filter(source == "human")
  val_mouse <- val_data %>% filter(source == "mouse")
  test_mouse <- test_data %>% filter(source == "mouse")

  message(sprintf("\nMouse data: Train=%s, Val=%s, Test=%s",
                  format(nrow(train_mouse), big.mark = ","),
                  format(nrow(val_mouse), big.mark = ","),
                  format(nrow(test_mouse), big.mark = ",")))
  message(sprintf("Human training data available: %s",
                  format(nrow(train_human), big.mark = ",")))

  # Calculate replay samples
  n_replay <- ceiling(nrow(train_mouse) * replay_fraction / (1 - replay_fraction))
  n_replay <- min(n_replay, nrow(train_human))

  message(sprintf("\nExperience replay: %.1f%% → %s human samples",
                  100 * replay_fraction, format(n_replay, big.mark = ",")))

  # Sample human data for replay
  if (replay_stratified && n_replay > 0) {
    # Stratified by epitope
    replay_idx <- caret::createDataPartition(
      train_human$epitope_idx,
      p = n_replay / nrow(train_human),
      list = FALSE
    )
    replay_data <- train_human[replay_idx, ]
  } else if (n_replay > 0) {
    replay_data <- train_human %>% sample_n(n_replay)
  } else {
    replay_data <- train_human[0, ]
  }

  message(sprintf("  Replay samples selected: %s (%.1f%% of combined)",
                  format(nrow(replay_data), big.mark = ","),
                  100 * nrow(replay_data) / (nrow(train_mouse) + nrow(replay_data))))

  # Combine mouse + replay for training
  phase2_train <- bind_rows(train_mouse, replay_data)

  # Re-encode for Phase 2
  cdr3_max_len <- 25L
  epitope_max_len <- 30L

  trb_vocab <- data_splits$trb_vocab
  tra_vocab <- data_splits$tra_vocab

  encode_phase2_split <- function(data, name) {
    message("  Encoding ", name, "...")

    cdr3_alpha_idx <- sequences_to_indices(data$cdr3_alpha, cdr3_max_len)
    cdr3_beta_idx <- sequences_to_indices(data$cdr3_beta, cdr3_max_len)
    epitope_idx <- sequences_to_indices(data$epitope, epitope_max_len)

    trb_data <- data %>% rename(v.segm = v_beta, j.segm = j_beta)
    trb_data <- encode_vj_for_dataset(trb_data, trb_vocab)

    tra_data <- data %>% rename(v.segm = v_alpha, j.segm = j_alpha)
    tra_data <- encode_vj_for_dataset(tra_data, tra_vocab)

    list(
      data = data,
      cdr3_alpha_idx = cdr3_alpha_idx,
      cdr3_beta_idx = cdr3_beta_idx,
      epitope_idx = epitope_idx,
      v_alpha_idx = as.integer(tra_data$v_idx),
      j_alpha_idx = as.integer(tra_data$j_idx),
      v_beta_idx = as.integer(trb_data$v_idx),
      j_beta_idx = as.integer(trb_data$j_idx),
      labels = as.integer(data$epitope_idx),
      weights = as.numeric(data$sample_weight),
      # V9.1: Include MHC indices
      mhc_class_idx = as.integer(data$mhc_class_idx),
      mhc_allele_idx = as.integer(data$mhc_allele_idx)
    )
  }

  train_enc <- encode_phase2_split(phase2_train, "training")
  val_enc <- encode_phase2_split(val_mouse, "validation")
  test_enc <- encode_phase2_split(test_mouse, "test")

  unique_epitope_idx <- data_splits$unique_epitope_idx

  message("\n", strrep("=", 60))

  list(
    train = train_enc,
    validation = val_enc,
    test = test_enc,
    unique_epitope_idx = unique_epitope_idx,
    replay_config = list(
      fraction = replay_fraction,
      n_replay = nrow(replay_data),
      n_mouse = nrow(train_mouse),
      stratified = replay_stratified
    )
  )
}


# ===== Diagnostics =====

#' Print summary of paired chain data splits
#' @param data_splits Output from prepare_combined_data()
#' @export
print_paired_data_summary <- function(data_splits) {

  m <- data_splits$metadata
  trb <- data_splits$trb_vocab
  tra <- data_splits$tra_vocab
  tr <- data_splits$train

  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("PAIRED CHAIN DATA SUMMARY (V9.1)\n")
  cat(strrep("=", 60), "\n")

  cat(sprintf("\nSplits: Train=%s, Val=%s, Test=%s\n",
              format(m$n_train, big.mark = ","),
              format(m$n_val, big.mark = ","),
              format(m$n_test, big.mark = ",")))

  cat("Epitope classes:", m$n_epitopes, "\n")

  cat(sprintf("\nChain pairing: TRA+TRB=%s, TRB-only=%s\n",
              format(m$n_paired, big.mark = ","),
              format(m$n_unpaired, big.mark = ",")))

  cat(sprintf("\nData sources: VDJdb=%s, IEDB=%s, McPAS=%s\n",
              format(m$n_vdjdb, big.mark = ","),
              format(m$n_iedb, big.mark = ","),
              format(m$n_mcpas, big.mark = ",")))

  cat(sprintf("\nV/J vocab sizes: TRB V=%d J=%d | TRA V=%d J=%d\n",
              trb$v$size, trb$j$size, tra$v$size, tra$j$size))

  # V9.1: MHC summary
  if (!is.null(data_splits$mhc_vocab)) {
    mhc <- data_splits$mhc_vocab
    cat(sprintf("\nMHC vocab sizes: Class=%d, Allele=%d (min_freq=%d)\n",
                m$mhc_class_vocab_size, m$mhc_allele_vocab_size, m$mhc_min_allele_freq))
    cat(sprintf("  With specific allele: %s (%.1f%%)\n",
                format(m$n_with_specific_allele, big.mark = ","),
                100 * m$n_with_specific_allele / (m$n_train + m$n_val + m$n_test)))
    cat(sprintf("  With MHC weight penalty: %s\n",
                format(m$n_with_mhc_penalty, big.mark = ",")))
  }

  cat(sprintf("\nTensor shapes (train):\n"))
  cat(sprintf("  CDR3α: %dx%d, CDR3β: %dx%d\n",
              nrow(tr$cdr3_alpha_idx), ncol(tr$cdr3_alpha_idx),
              nrow(tr$cdr3_beta_idx), ncol(tr$cdr3_beta_idx)))
  cat(sprintf("  MHC class: %d, MHC allele: %d\n",
              length(tr$mhc_class_idx), length(tr$mhc_allele_idx)))

  cat(strrep("=", 60), "\n\n")
}


#' Validate data splits structure from prepare_combined_data()
#'
#' @param data_splits Output from prepare_combined_data()
#' @return TRUE if valid, stops with error otherwise
#' @export
validate_data_splits <- function(data_splits) {

  message("Validating data splits structure...")

  # Check required top-level components
  required <- c("train", "validation", "test", "epitope_to_idx", "idx_to_epitope",
                "unique_epitopes", "unique_epitope_idx", "trb_vocab", "tra_vocab",
                "mhc_vocab", "metadata")

  missing <- setdiff(required, names(data_splits))
  if (length(missing) > 0) {
    stop("Missing required components: ", paste(missing, collapse = ", "))
  }

  # Check split components
  split_required <- c("data", "cdr3_alpha_idx", "cdr3_beta_idx", "epitope_idx",
                      "v_alpha_idx", "j_alpha_idx", "v_beta_idx", "j_beta_idx",
                      "labels", "weights", "mhc_class_idx", "mhc_allele_idx")

  for (split_name in c("train", "validation", "test")) {
    split <- data_splits[[split_name]]
    missing_split <- setdiff(split_required, names(split))
    if (length(missing_split) > 0) {
      stop(split_name, " split missing: ", paste(missing_split, collapse = ", "))
    }

    # Check dimensions match
    n <- nrow(split$cdr3_alpha_idx)
    if (nrow(split$cdr3_beta_idx) != n) stop(split_name, ": CDR3β dim mismatch")
    if (nrow(split$epitope_idx) != n) stop(split_name, ": epitope dim mismatch")
    if (length(split$labels) != n) stop(split_name, ": labels dim mismatch")
    if (length(split$weights) != n) stop(split_name, ": weights dim mismatch")
    if (length(split$mhc_class_idx) != n) stop(split_name, ": mhc_class dim mismatch")
    if (length(split$mhc_allele_idx) != n) stop(split_name, ": mhc_allele dim mismatch")
  }

  # Check MHC vocabulary
  if (is.null(data_splits$mhc_vocab$class) || is.null(data_splits$mhc_vocab$allele)) {
    stop("MHC vocabulary incomplete")
  }

  message("  ✓ All validations passed")
  invisible(TRUE)
}
