# MHC Encoding for TCR-Epitope Model (V9.1)
#
# Functions for MHC allele standardization, vocabulary building, and encoding.
# Implements hierarchical MHC representation: MHC class + MHC allele embeddings.
#
# Key design decisions:
#   - Serological names (HLA-A2) kept distinct from allelic names (HLA-A*02)
#   - Allele resolution truncated to 2-field (HLA-A*02:01)
#   - Class-only entries ("HLA class I") use <CLASS_ONLY> token
#   - Species-MHC mismatches filtered (e.g., Mamu-A*01 in mouse data)
#   - Poorly annotated samples receive reduced weight (0.5×)
#

library(dplyr)
library(stringr)
library(tidyr)

# ===== CONSTANTS =====

# MHC Class vocabulary (fixed)
MHC_CLASS_VOCAB <- list(

  PAD = 0L,
  MHCI = 1L,
  MHCII = 2L,
  UNK = 3L
)

# MHC Allele special tokens (indices 0-3 reserved)
MHC_ALLELE_SPECIAL_TOKENS <- list(


  PAD = 0L,
  UNK = 1L,
  MISSING = 2L,
  CLASS_ONLY = 3L
)

# Weight multiplier for class-only annotations
MHC_CLASS_ONLY_WEIGHT <- 0.5

# ===== MHC CLASS INFERENCE =====

#' Infer MHC class from allele name
#'
#' Determines MHC class (I or II) from the allele string. Uses pattern matching
#' based on standard HLA and H-2 nomenclature.
#'
#' @param allele Character vector of MHC allele names
#' @param existing_class Optional character vector of existing class annotations
#' @return Character vector: "MHCI", "MHCII", or NA
#' @export
#'
#' @examples
#' infer_mhc_class("HLA-A*02:01")        # "MHCI"
#' infer_mhc_class("HLA-DRB1*04:01")     # "MHCII"
#' infer_mhc_class("H-2Kb")              # "MHCI"
#' infer_mhc_class("I-Ab")               # "MHCII"
#' infer_mhc_class("HLA class I")        # "MHCI"
infer_mhc_class <- function(allele, existing_class = NULL) {

  # Use existing class if valid
  if (!is.null(existing_class)) {
    valid_existing <- !is.na(existing_class) & existing_class %in% c("MHCI", "MHCII")
    if (all(valid_existing)) {
      return(existing_class)
    }
  }

  # Vectorized inference
  result <- case_when(
    # ----- Class II patterns (check first - more specific) -----

    # Human Class II: DR, DQ, DP genes
    str_detect(allele, "HLA-D[RPQAB]|DRB|DQB|DPB|DRA|DQA|DPA") ~ "MHCII",

    # Mouse Class II: I-A, I-E
    str_detect(allele, "^I-[AE]|H-?2-?I[AE]|H2-I[AE]") ~ "MHCII",

    # Explicit class II annotations
    str_detect(allele, regex("class\\s*II", ignore_case = TRUE)) ~ "MHCII",

    # ----- Class I patterns -----

    # Human Class I: A, B, C, E, F, G genes
    str_detect(allele, "HLA-[ABCEFG]") ~ "MHCI",

    # Mouse Class I: K, D, L, Q (Qa), T (Tla)
    str_detect(allele, "H-?2-?[KDLQT]|H2-[KDLQT]") ~ "MHCI",

    # Explicit class I annotations (negative lookahead for "II")
    str_detect(allele, regex("class\\s*I(?!I)", ignore_case = TRUE)) ~ "MHCI",

    # Beta-2-microglobulin (Class I component)
    str_detect(allele, "^B2M$") ~ "MHCI",

    # Macaque Class I (for detection before filtering)
    str_detect(allele, "Mamu-[AB]") ~ "MHCI",

    # Human MR1 and CD1 (non-classical Class I)
    str_detect(allele, "MR1|CD1") ~ "MHCI",

    # ----- Default -----
    TRUE ~ NA_character_
  )

  # If existing_class provided, use it for NA results
  if (!is.null(existing_class)) {
    use_existing <- is.na(result) & !is.na(existing_class) & existing_class %in% c("MHCI", "MHCII")
    result[use_existing] <- existing_class[use_existing]
  }

  result
}

# ===== MHC ALLELE STANDARDIZATION =====

#' Standardize MHC allele nomenclature
#'
#' Normalizes MHC allele names to a consistent format:
#'
#' 1. Trims whitespace and converts to uppercase (except gene names)
#' 2. Handles multi-allele entries (takes first allele)
#' 3. Removes mutant/variant annotations
#' 4. Normalizes mouse H-2 format
#' 5. Truncates to 2-field resolution for HLA alleles
#' 6. Preserves serological names (HLA-A2) as distinct from allelic (HLA-A*02)
#'
#' @param allele Character vector of MHC allele names
#' @return Character vector of standardized allele names
#' @export
#'
#' @examples
#' standardize_mhc_allele("HLA-A*02:01:01:02")  # "HLA-A*02:01"
#' standardize_mhc_allele("HLA-A2")             # "HLA-A2" (serological, kept)
#' standardize_mhc_allele("H-2Kb")              # "H2-Kb"
#' standardize_mhc_allele("HLA-A*02:01 mutant") # "HLA-A*02:01"
standardize_mhc_allele <- function(allele) {

  if (length(allele) == 0) return(character(0))

  standardized <- vapply(allele, function(a) {

    if (is.na(a) || a == "") return(NA_character_)

    a <- str_trim(a)

    # ----- Handle class-only placeholders -----
    if (str_detect(a, regex("^HLA\\s+class\\s+(I|II)$", ignore_case = TRUE))) {
      return("<CLASS_ONLY>")
    }
    if (str_detect(a, regex("^class\\s+(I|II)$", ignore_case = TRUE))) {
      return("<CLASS_ONLY>")
    }
    if (str_detect(a, regex("^(I|II)$"))) {
      return("<CLASS_ONLY>")
    }

    # ----- Handle multi-allele entries (take first) -----
    # "HLA-A*02:01, HLA-B*07:02" → "HLA-A*02:01"
    if (str_detect(a, ",")) {
      a <- str_trim(str_split(a, ",")[[1]][1])
    }

    # ----- Remove mutant/variant annotations -----
    # "HLA-A*02:01 K66A mutant" → "HLA-A*02:01"
    a <- str_remove(a, "\\s+[A-Z]\\d+[A-Z].*$")  # "K66A mutant" pattern
    a <- str_remove(a, "\\s+mutant.*$")
    a <- str_remove(a, "\\s+variant.*$")
    a <- str_trim(a)

    # ----- Normalize mouse H-2 format -----
    # Various formats → "H2-Xx"
    if (str_detect(a, "^H-?2")) {
      # Remove hyphen inconsistency: "H-2Kb", "H2-Kb", "H-2-Kb" → "H2-Kb"
      a <- str_replace(a, "^H-2-?", "H2-")
      a <- str_replace(a, "^H2([A-Z])", "H2-\\1")  # "H2Kb" → "H2-Kb"

      # Normalize case for allele: "H2-KB" → "H2-Kb"
      if (str_detect(a, "^H2-[A-Z][A-Za-z]?$")) {
        parts <- str_match(a, "^(H2-)([A-Z])([a-zA-Z]?)$")
        if (!is.na(parts[1, 1])) {
          gene <- parts[1, 3]
          allele_char <- tolower(parts[1, 4])
          a <- paste0("H2-", gene, allele_char)
        }
      }

      return(a)
    }

    # ----- Normalize mouse I-A/I-E format -----
    if (str_detect(a, "^I-[AE]")) {
      # "I-Ab", "I-Ag7" → keep as-is (already standard)
      return(a)
    }

    # ----- Handle HLA alleles -----
    if (str_detect(a, "^HLA-")) {

      # Check if serological (no asterisk, number directly after gene)
      # "HLA-A2", "HLA-B7", "HLA-DR5" → keep as serological
      if (str_detect(a, "^HLA-[A-Z]+\\d+$") && !str_detect(a, "\\*")) {
        return(a)
      }

      # Check if allelic (has asterisk)
      if (str_detect(a, "\\*")) {
        # Truncate to 2-field resolution
        # "HLA-A*02:01:01:02" → "HLA-A*02:01"
        match <- str_match(a, "^(HLA-[A-Z]+\\d*\\*\\d+:\\d+)")
        if (!is.na(match[1, 2])) {
          return(match[1, 2])
        }

        # Handle 1-field only: "HLA-A*02" → keep as-is
        match <- str_match(a, "^(HLA-[A-Z]+\\d*\\*\\d+)$")
        if (!is.na(match[1, 2])) {
          return(match[1, 2])
        }
      }

      # Other HLA formats - return as-is
      return(a)
    }

    # ----- Handle non-standard prefixes -----

    # DRB1*04:01 → HLA-DRB1*04:01
    if (str_detect(a, "^D[RPQAB]")) {
      a <- paste0("HLA-", a)
      # Apply truncation
      match <- str_match(a, "^(HLA-[A-Z]+\\d*\\*\\d+:\\d+)")
      if (!is.na(match[1, 2])) {
        return(match[1, 2])
      }
      return(a)
    }

    # ----- Special molecules -----
    # "human CD1d" → "CD1d"
    if (str_detect(a, "CD1|MR1")) {
      a <- str_extract(a, "CD1[a-d]?|MR1")
      return(a)
    }

    # ----- Non-human, non-mouse (e.g., Mamu) -----
    # Return as-is for detection/filtering
    if (str_detect(a, "Mamu")) {
      return(a)
    }

    # ----- Fallback: return trimmed original -----
    a

  }, character(1), USE.NAMES = FALSE)

  standardized
}

# ===== SPECIES-MHC CONSISTENCY CHECK =====

#' Check MHC-species consistency
#'
#' Identifies entries where TCR species doesn't match MHC type.
#' Human TCRs should have HLA; mouse TCRs should have H-2/I-A/I-E.
#'
#' @param mhc_allele Character vector of standardized MHC alleles
#' @param tcr_species Character vector of TCR species ("human" or "mouse")
#' @return Logical vector: TRUE if consistent, FALSE if mismatch
#' @export
#'
#' @examples
#' check_mhc_species_consistency("HLA-A*02:01", "human")  # TRUE
#' check_mhc_species_consistency("H2-Kb", "mouse")        # TRUE
#' check_mhc_species_consistency("Mamu-A*01", "mouse")    # FALSE (macaque MHC)
#' check_mhc_species_consistency("HLA-A*02:01", "mouse")  # FALSE (human MHC)
check_mhc_species_consistency <- function(mhc_allele, tcr_species) {

  # Define valid patterns
  human_mhc_pattern <- "^HLA-|^CD1|^MR1|^B2M$|^<"
  mouse_mhc_pattern <- "^H2-|^I-[AE]|^<"

  consistent <- case_when(
    # Missing/special tokens always consistent
    is.na(mhc_allele) | mhc_allele == "" ~ TRUE,
    mhc_allele %in% c("<PAD>", "<UNK>", "<MISSING>", "<CLASS_ONLY>") ~ TRUE,

    # Human TCR should have human MHC
    tcr_species == "human" & str_detect(mhc_allele, human_mhc_pattern) ~ TRUE,
    tcr_species == "human" & !str_detect(mhc_allele, human_mhc_pattern) ~ FALSE,

    # Mouse TCR should have mouse MHC
    tcr_species == "mouse" & str_detect(mhc_allele, mouse_mhc_pattern) ~ TRUE,
    tcr_species == "mouse" & !str_detect(mhc_allele, mouse_mhc_pattern) ~ FALSE,

    # Unknown species - allow anything
    TRUE ~ TRUE
  )

  consistent
}

#' Identify MHC-species mismatches for reporting
#'
#' @param data Data frame with mhc_allele_std and tcr_species columns
#' @return Data frame summarizing mismatches
#' @export
summarize_mhc_species_mismatches <- function(data) {

  if (!"mhc_allele_std" %in% names(data)) {
    stop("Data must have 'mhc_allele_std' column. Run standardize_mhc_allele() first.")
  }

  mismatches <- data %>%
    mutate(
      is_consistent = check_mhc_species_consistency(mhc_allele_std, tcr_species)
    ) %>%
    filter(!is_consistent) %>%
    count(tcr_species, mhc_allele_std, name = "n_mismatched") %>%
    arrange(desc(n_mismatched))

  if (nrow(mismatches) > 0) {
    message("\nMHC-Species Mismatches Found:")
    message("  Total mismatched entries: ", sum(mismatches$n_mismatched))
    message("\n  Top mismatches:")
    print(head(mismatches, 20))
  } else {
    message("\nNo MHC-species mismatches found.")
  }

  invisible(mismatches)
}

# ===== MHC VOCABULARY BUILDING =====

#' Build MHC vocabulary from combined data
#'
#' Creates vocabularies for MHC class (fixed) and MHC allele (data-driven).
#' Allele vocabulary uses frequency threshold to manage long-tail distribution.
#'
#' @param data Data frame with mhc_allele_std column (output from standardization)
#' @param min_allele_freq Minimum frequency for allele to get its own token (default: 10)
#' @param verbose Print vocabulary summary
#' @return List with class_vocab, allele_vocab, and metadata
#' @export
build_mhc_vocabulary <- function(data,
                                 min_allele_freq = 10,
                                 verbose = TRUE) {

  if (!"mhc_allele_std" %in% names(data)) {
    stop("Data must have 'mhc_allele_std' column. Run standardize_mhc_allele() first.")
  }

  if (verbose) {
    message("\n", strrep("=", 60))
    message("BUILDING MHC VOCABULARY")
    message(strrep("=", 60))
  }

  # ----- Class vocabulary (fixed) -----
  class_vocab <- list(
    token_to_idx = MHC_CLASS_VOCAB,
    idx_to_token = setNames(
      names(MHC_CLASS_VOCAB),
      as.character(unlist(MHC_CLASS_VOCAB))
    ),
    size = length(MHC_CLASS_VOCAB)
  )

  if (verbose) {
    message("\nMHC Class vocabulary (fixed):")
    message("  Size: ", class_vocab$size)
    message("  Tokens: ", paste(names(MHC_CLASS_VOCAB), collapse = ", "))
  }

  # ----- Allele vocabulary (data-driven) -----

  # Count allele frequencies
  allele_counts <- data %>%
    filter(!is.na(mhc_allele_std) & mhc_allele_std != "") %>%
    filter(!mhc_allele_std %in% c("<CLASS_ONLY>", "<MISSING>", "<UNK>", "<PAD>")) %>%
    count(mhc_allele_std, name = "freq") %>%
    arrange(desc(freq))

  # Filter by frequency threshold
  alleles_above_thresh <- allele_counts %>%
    filter(freq >= min_allele_freq) %>%
    pull(mhc_allele_std)

  # Build vocabulary: special tokens + frequent alleles
  special_tokens <- names(MHC_ALLELE_SPECIAL_TOKENS)
  all_allele_tokens <- c(special_tokens, alleles_above_thresh)

  allele_token_to_idx <- setNames(
    seq_along(all_allele_tokens) - 1L,
    all_allele_tokens
  )

  allele_idx_to_token <- setNames(
    all_allele_tokens,
    as.character(seq_along(all_allele_tokens) - 1L)
  )

  allele_vocab <- list(
    token_to_idx = allele_token_to_idx,
    idx_to_token = allele_idx_to_token,
    size = length(all_allele_tokens),
    min_freq = min_allele_freq,
    n_alleles = length(alleles_above_thresh),
    n_special = length(special_tokens)
  )

  if (verbose) {
    # Calculate coverage statistics
    n_total_with_allele <- sum(allele_counts$freq)
    n_covered <- allele_counts %>%
      filter(mhc_allele_std %in% alleles_above_thresh) %>%
      pull(freq) %>%
      sum()
    coverage_pct <- round(100 * n_covered / n_total_with_allele, 1)

    n_rare <- nrow(allele_counts) - length(alleles_above_thresh)

    message("\nMHC Allele vocabulary:")
    message("  Total size: ", allele_vocab$size, " tokens")
    message("  Special tokens: ", allele_vocab$n_special)
    message("  Allele tokens: ", allele_vocab$n_alleles)
    message("  Frequency threshold: >= ", min_allele_freq)
    message("  Coverage: ", coverage_pct, "% of entries with allele")
    message("  Rare alleles (→ <UNK>): ", n_rare)
  }

  # ----- Compile result -----

  result <- list(
    class = class_vocab,
    allele = allele_vocab,
    allele_frequencies = allele_counts,
    metadata = list(
      min_allele_freq = min_allele_freq,
      n_entries_processed = nrow(data),
      n_unique_raw_alleles = nrow(allele_counts),
      timestamp = Sys.time()
    )
  )

  if (verbose) {
    message("\n", strrep("-", 40))
    message("Top 20 alleles in vocabulary:")
    top_alleles <- head(allele_counts, 20)
    for (i in seq_len(nrow(top_alleles))) {
      message(sprintf("  %s: %s",
                      top_alleles$mhc_allele_std[i],
                      format(top_alleles$freq[i], big.mark = ",")))
    }
    message(strrep("=", 60))
  }

  result
}

# ===== MHC ENCODING =====

#' Encode MHC information for dataset
#'
#' Adds MHC class and allele indices to data frame, plus MHC quality weight.
#'
#' @param data Data frame with mhc.a, mhc.class columns (raw MHC data)
#' @param mhc_vocab MHC vocabulary from build_mhc_vocabulary()
#' @param verbose Print encoding summary
#' @return Data frame with added columns: mhc_allele_std, mhc_class_inferred,
#'         mhc_class_idx, mhc_allele_idx, mhc_weight
#' @export
encode_mhc_for_dataset <- function(data, mhc_vocab, verbose = TRUE) {

  if (verbose) {
    message("\nEncoding MHC information...")
  }

  n_before <- nrow(data)

  # ----- Step 1: Standardize allele names -----
  data <- data %>%
    mutate(
      mhc_allele_std = standardize_mhc_allele(mhc.a)
    )

  # ----- Step 2: Infer MHC class -----
  data <- data %>%
    mutate(
      mhc_class_inferred = infer_mhc_class(mhc_allele_std, mhc.class)
    )

  # ----- Step 3: Determine tcr_species for consistency check -----
  # (May already exist from earlier processing)
  if (!"tcr_species" %in% names(data)) {
    data <- data %>%
      mutate(
        tcr_species = case_when(
          species == "HomoSapiens" ~ "human",
          species == "MusMusculus" ~ "mouse",
          TRUE ~ "other"
        )
      )
  }

  # ----- Step 4: Check species-MHC consistency -----
  data <- data %>%
    mutate(
      mhc_species_consistent = check_mhc_species_consistency(mhc_allele_std, tcr_species)
    )

  n_inconsistent <- sum(!data$mhc_species_consistent)
  if (verbose && n_inconsistent > 0) {
    message("  Species-MHC mismatches: ", format(n_inconsistent, big.mark = ","),
            " (will be filtered)")
  }

  # Filter out inconsistent entries
  data <- data %>%
    filter(mhc_species_consistent) %>%
    select(-mhc_species_consistent)

  n_after <- nrow(data)
  if (verbose) {
    message("  Entries after MHC consistency filter: ", format(n_after, big.mark = ","),
            " (removed ", format(n_before - n_after, big.mark = ","), ")")
  }

  # ----- Step 5: Encode MHC class -----
  class_token_to_idx <- mhc_vocab$class$token_to_idx

  data <- data %>%
    mutate(
      mhc_class_idx = case_when(
        is.na(mhc_class_inferred) ~ class_token_to_idx[["UNK"]],
        mhc_class_inferred == "MHCI" ~ class_token_to_idx[["MHCI"]],
        mhc_class_inferred == "MHCII" ~ class_token_to_idx[["MHCII"]],
        TRUE ~ class_token_to_idx[["UNK"]]
      )
    )

  # ----- Step 6: Encode MHC allele -----
  allele_token_to_idx <- mhc_vocab$allele$token_to_idx

  data <- data %>%
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

  # ----- Step 7: Calculate MHC quality weight -----
  data <- data %>%
    mutate(
      mhc_weight = case_when(
        # Class-only gets reduced weight
        mhc_allele_idx == allele_token_to_idx[["CLASS_ONLY"]] ~ MHC_CLASS_ONLY_WEIGHT,
        # Missing gets reduced weight
        mhc_allele_idx == allele_token_to_idx[["MISSING"]] ~ MHC_CLASS_ONLY_WEIGHT,
        # Everything else gets full weight
        TRUE ~ 1.0
      )
    )

  # ----- Summary -----
  if (verbose) {
    message("\n  MHC encoding summary:")

    # Class distribution
    class_dist <- data %>%
      mutate(class_name = mhc_vocab$class$idx_to_token[as.character(mhc_class_idx)]) %>%
      count(class_name) %>%
      arrange(desc(n))
    message("    Class distribution:")
    for (i in seq_len(nrow(class_dist))) {
      message(sprintf("      %s: %s (%.1f%%)",
                      class_dist$class_name[i],
                      format(class_dist$n[i], big.mark = ","),
                      100 * class_dist$n[i] / nrow(data)))
    }

    # Allele token type distribution
    allele_type_dist <- data %>%
      mutate(
        allele_type = case_when(
          mhc_allele_idx == allele_token_to_idx[["PAD"]] ~ "PAD",
          mhc_allele_idx == allele_token_to_idx[["UNK"]] ~ "UNK (rare)",
          mhc_allele_idx == allele_token_to_idx[["MISSING"]] ~ "MISSING",
          mhc_allele_idx == allele_token_to_idx[["CLASS_ONLY"]] ~ "CLASS_ONLY",
          TRUE ~ "Specific allele"
        )
      ) %>%
      count(allele_type) %>%
      arrange(desc(n))

    message("    Allele token types:")
    for (i in seq_len(nrow(allele_type_dist))) {
      message(sprintf("      %s: %s (%.1f%%)",
                      allele_type_dist$allele_type[i],
                      format(allele_type_dist$n[i], big.mark = ","),
                      100 * allele_type_dist$n[i] / nrow(data)))
    }

    # Weight distribution
    weight_dist <- data %>%
      count(mhc_weight) %>%
      arrange(desc(n))
    message("    MHC weight distribution:")
    for (i in seq_len(nrow(weight_dist))) {
      message(sprintf("      %.1f×: %s entries",
                      weight_dist$mhc_weight[i],
                      format(weight_dist$n[i], big.mark = ",")))
    }
  }

  data
}

# ===== PERSISTENCE =====

#' Save MHC vocabulary to file
#'
#' @param mhc_vocab MHC vocabulary from build_mhc_vocabulary()
#' @param filepath Path to save RDS file
#' @export
save_mhc_vocabulary <- function(mhc_vocab, filepath) {
  saveRDS(mhc_vocab, filepath)
  message("MHC vocabulary saved to: ", filepath)
  invisible(filepath)
}

#' Load MHC vocabulary from file
#'
#' @param filepath Path to RDS file
#' @return MHC vocabulary list
#' @export
load_mhc_vocabulary <- function(filepath) {
  mhc_vocab <- readRDS(filepath)
  message("MHC vocabulary loaded from: ", filepath)
  message("  Class vocab size: ", mhc_vocab$class$size)
  message("  Allele vocab size: ", mhc_vocab$allele$size)
  mhc_vocab
}

#' Print MHC vocabulary summary
#'
#' @param mhc_vocab MHC vocabulary list
#' @export
print_mhc_vocab_summary <- function(mhc_vocab) {

  cat("\n", strrep("=", 50), "\n", sep = "")
  cat("MHC VOCABULARY SUMMARY\n")
  cat(strrep("=", 50), "\n")

  cat("\nClass vocabulary:\n")
  cat("  Size: ", mhc_vocab$class$size, " tokens\n", sep = "")
  cat("  Tokens: ", paste(names(mhc_vocab$class$token_to_idx), collapse = ", "), "\n", sep = "")

  cat("\nAllele vocabulary:\n")
  cat("  Total size: ", mhc_vocab$allele$size, " tokens\n", sep = "")
  cat("  Special tokens: ", mhc_vocab$allele$n_special, "\n", sep = "")
  cat("  Allele tokens: ", mhc_vocab$allele$n_alleles, "\n", sep = "")
  cat("  Frequency threshold: >= ", mhc_vocab$allele$min_freq, "\n", sep = "")

  if (!is.null(mhc_vocab$allele_frequencies)) {
    top_10 <- head(mhc_vocab$allele_frequencies, 10)
    cat("\n  Top 10 alleles:\n")
    for (i in seq_len(nrow(top_10))) {
      cat(sprintf("    %2d. %s (%s)\n",
                  i, top_10$mhc_allele_std[i],
                  format(top_10$freq[i], big.mark = ",")))
    }
  }

  cat(strrep("=", 50), "\n\n")

  invisible(mhc_vocab)
}

# ===== INTEGRATION WITH SAMPLE WEIGHTS =====

#' Calculate combined sample weight including MHC quality
#'
#' Combines VDJdb score-based weight with MHC annotation quality weight.
#'
#' @param score_weight Numeric vector of score-based weights
#' @param mhc_weight Numeric vector of MHC quality weights
#' @return Combined weight (product of both weights)
#' @export
combine_sample_weights <- function(score_weight, mhc_weight) {
  score_weight * mhc_weight
}

# ===== DIAGNOSTIC FUNCTIONS =====

#' Analyze MHC encoding results
#'
#' Provides detailed analysis of MHC encoding for quality control.
#'
#' @param data Data frame after encode_mhc_for_dataset()
#' @param mhc_vocab MHC vocabulary
#' @return List of analysis results (invisible)
#' @export
analyze_mhc_encoding <- function(data, mhc_vocab) {

  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("MHC ENCODING ANALYSIS\n")
  cat(strrep("=", 60), "\n")

  # ----- Overall coverage -----
  cat("\n--- Overall Coverage ---\n")
  cat("Total entries: ", format(nrow(data), big.mark = ","), "\n")

  n_with_class <- sum(data$mhc_class_idx != mhc_vocab$class$token_to_idx[["UNK"]])
  cat("With MHC class: ", format(n_with_class, big.mark = ","),
      " (", round(100 * n_with_class / nrow(data), 1), "%)\n", sep = "")

  n_with_specific_allele <- sum(!data$mhc_allele_idx %in%
                                  c(mhc_vocab$allele$token_to_idx[["UNK"]],
                                    mhc_vocab$allele$token_to_idx[["MISSING"]],
                                    mhc_vocab$allele$token_to_idx[["CLASS_ONLY"]]))
  cat("With specific allele: ", format(n_with_specific_allele, big.mark = ","),
      " (", round(100 * n_with_specific_allele / nrow(data), 1), "%)\n", sep = "")

  # ----- By source -----
  if ("source_type" %in% names(data)) {
    cat("\n--- Coverage by Source ---\n")
    source_summary <- data %>%
      group_by(source_type) %>%
      summarise(
        n = n(),
        pct_class = round(100 * mean(mhc_class_idx != mhc_vocab$class$token_to_idx[["UNK"]]), 1),
        pct_specific = round(100 * mean(!mhc_allele_idx %in%
                                          c(mhc_vocab$allele$token_to_idx[["UNK"]],
                                            mhc_vocab$allele$token_to_idx[["MISSING"]],
                                            mhc_vocab$allele$token_to_idx[["CLASS_ONLY"]])), 1),
        pct_class_only = round(100 * mean(mhc_allele_idx ==
                                            mhc_vocab$allele$token_to_idx[["CLASS_ONLY"]]), 1),
        .groups = "drop"
      )
    print(source_summary)
  }

  # ----- By species -----
  if ("tcr_species" %in% names(data)) {
    cat("\n--- Coverage by Species ---\n")
    species_summary <- data %>%
      group_by(tcr_species) %>%
      summarise(
        n = n(),
        n_unique_alleles = n_distinct(mhc_allele_std[!mhc_allele_std %in%
                                                       c("<CLASS_ONLY>", NA)]),
        pct_class = round(100 * mean(mhc_class_idx != mhc_vocab$class$token_to_idx[["UNK"]]), 1),
        pct_specific = round(100 * mean(!mhc_allele_idx %in%
                                          c(mhc_vocab$allele$token_to_idx[["UNK"]],
                                            mhc_vocab$allele$token_to_idx[["MISSING"]],
                                            mhc_vocab$allele$token_to_idx[["CLASS_ONLY"]])), 1),
        .groups = "drop"
      )
    print(species_summary)
  }

  # ----- Top alleles used -----
  cat("\n--- Top 20 Encoded Alleles ---\n")
  top_alleles <- data %>%
    filter(!mhc_allele_idx %in% c(mhc_vocab$allele$token_to_idx[["UNK"]],
                                  mhc_vocab$allele$token_to_idx[["MISSING"]],
                                  mhc_vocab$allele$token_to_idx[["CLASS_ONLY"]],
                                  mhc_vocab$allele$token_to_idx[["PAD"]])) %>%
    count(mhc_allele_std, mhc_allele_idx) %>%
    arrange(desc(n)) %>%
    head(20)

  for (i in seq_len(nrow(top_alleles))) {
    cat(sprintf("  %2d. [%3d] %s: %s\n",
                i,
                top_alleles$mhc_allele_idx[i],
                top_alleles$mhc_allele_std[i],
                format(top_alleles$n[i], big.mark = ",")))
  }

  cat(strrep("=", 60), "\n\n")

  invisible(list(
    n_total = nrow(data),
    n_with_class = n_with_class,
    n_with_specific = n_with_specific_allele,
    top_alleles = top_alleles
  ))
}
