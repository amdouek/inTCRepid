# IEDB Data Preprocessing for TCR-Epitope Model (V9.1)
#
# Functions to load and curate IEDB TCR data for integration with VDJdb.
# V7 update: Extract paired chain (TRA+TRB) information with V/J genes.
# V9.1 update: Improved MHC class inference from allele strings.
# Data source: https://www.iedb.org/ → More IEDB → tcr_full_v3.zip
#

library(data.table)
library(dplyr)
library(tidyr)
library(stringr)

# ===== Load IEDB Data =====

#' Load IEDB TCR data with two-row header handling
#'
#' IEDB tcr_full_v3.csv has a two-row header:
#'   Row 1: Category names (Receptor, Epitope, Chain 1, Chain 2, etc.)
#'   Row 2: Field names within each category
#' This function combines them into unique column names.
#'
#' @param filepath Path to tcr_full_v3.csv
#' @return data.table with combined column names
#' @export
load_iedb_tcr <- function(filepath) {

  message("Loading IEDB: ", filepath)

  # Read header rows
  header <- fread(filepath, nrows = 2, header = FALSE, fill = TRUE)
  categories <- as.character(header[1, ])
  fields <- as.character(header[2, ])

  # Forward-fill categories (they span multiple columns)
  current <- ""
  for (i in seq_along(categories)) {
    if (!is.na(categories[i]) && categories[i] != "") current <- categories[i]
    categories[i] <- current
  }

  # Create clean column names
  col_names <- paste(categories, fields, sep = "_") %>%
    str_replace_all("\\s+", "_") %>%
    str_replace_all("[^A-Za-z0-9_]", "") %>%
    make.unique()

  # Read data (skip header rows)
  iedb <- fread(filepath, skip = 2, header = FALSE, fill = TRUE,
                na.strings = c("", "NA", "null"))
  names(iedb) <- col_names

  message("  Loaded ", format(nrow(iedb), big.mark = ","), " rows x ", ncol(iedb), " cols")

  # Print available columns for debugging
  chain1_cols <- grep("Chain_1", names(iedb), value = TRUE)
  chain2_cols <- grep("Chain_2", names(iedb), value = TRUE)
  message("  Chain 1 columns: ", length(chain1_cols))
  message("  Chain 2 columns: ", length(chain2_cols))

  iedb
}

# ===== Audit IEDB Schema =====

#' Audit IEDB data for V/J gene and chain coverage
#'
#' @param iedb_data Raw IEDB data from load_iedb_tcr()
#' @return Summary tibble
#' @export
audit_iedb_schema <- function(iedb_data) {

  message("\n", strrep("=", 60))
  message("IEDB SCHEMA AUDIT")
  message(strrep("=", 60))

  # Find relevant columns
  cols <- names(iedb_data)

  # Chain 1 (typically alpha)
  c1_cdr3_cur <- grep("Chain_1.*CDR3.*Curated$", cols, value = TRUE)[1]
  c1_cdr3_calc <- grep("Chain_1.*CDR3.*Calculated$", cols, value = TRUE)[1]
  c1_v_cur <- grep("Chain_1.*Curated.*V.*Gene", cols, value = TRUE)[1]
  c1_v_calc <- grep("Chain_1.*Calculated.*V.*Gene", cols, value = TRUE)[1]
  c1_j_cur <- grep("Chain_1.*Curated.*J.*Gene", cols, value = TRUE)[1]
  c1_j_calc <- grep("Chain_1.*Calculated.*J.*Gene", cols, value = TRUE)[1]
  c1_type <- grep("Chain_1.*Type$", cols, value = TRUE)[1]

  # Chain 2 (typically beta)
  c2_cdr3_cur <- grep("Chain_2.*CDR3.*Curated$", cols, value = TRUE)[1]
  c2_cdr3_calc <- grep("Chain_2.*CDR3.*Calculated$", cols, value = TRUE)[1]
  c2_v_cur <- grep("Chain_2.*Curated.*V.*Gene", cols, value = TRUE)[1]
  c2_v_calc <- grep("Chain_2.*Calculated.*V.*Gene", cols, value = TRUE)[1]
  c2_j_cur <- grep("Chain_2.*Curated.*J.*Gene", cols, value = TRUE)[1]
  c2_j_calc <- grep("Chain_2.*Calculated.*J.*Gene", cols, value = TRUE)[1]
  c2_type <- grep("Chain_2.*Type$", cols, value = TRUE)[1]

  message("\nColumn mapping:")
  message("  Chain 1 CDR3 (curated): ", c1_cdr3_cur %||% "NOT FOUND")
  message("  Chain 1 CDR3 (calc): ", c1_cdr3_calc %||% "NOT FOUND")
  message("  Chain 1 V gene (curated): ", c1_v_cur %||% "NOT FOUND")
  message("  Chain 1 V gene (calc): ", c1_v_calc %||% "NOT FOUND")
  message("  Chain 1 J gene (curated): ", c1_j_cur %||% "NOT FOUND")
  message("  Chain 1 J gene (calc): ", c1_j_calc %||% "NOT FOUND")
  message("  Chain 1 Type: ", c1_type %||% "NOT FOUND")

  message("\n  Chain 2 CDR3 (curated): ", c2_cdr3_cur %||% "NOT FOUND")
  message("  Chain 2 CDR3 (calc): ", c2_cdr3_calc %||% "NOT FOUND")
  message("  Chain 2 V gene (curated): ", c2_v_cur %||% "NOT FOUND")
  message("  Chain 2 V gene (calc): ", c2_v_calc %||% "NOT FOUND")
  message("  Chain 2 J gene (curated): ", c2_j_cur %||% "NOT FOUND")
  message("  Chain 2 J gene (calc): ", c2_j_calc %||% "NOT FOUND")
  message("  Chain 2 Type: ", c2_type %||% "NOT FOUND")

  # Coverage statistics
  n_total <- nrow(iedb_data)

  count_non_empty <- function(col) {
    if (is.null(col) || is.na(col)) return(0)
    sum(!is.na(iedb_data[[col]]) & iedb_data[[col]] != "")
  }

  coverage <- tibble(
    field = c("Chain1_CDR3_curated", "Chain1_CDR3_calc",
              "Chain1_V_curated", "Chain1_V_calc",
              "Chain1_J_curated", "Chain1_J_calc",
              "Chain2_CDR3_curated", "Chain2_CDR3_calc",
              "Chain2_V_curated", "Chain2_V_calc",
              "Chain2_J_curated", "Chain2_J_calc"),
    column = c(c1_cdr3_cur, c1_cdr3_calc, c1_v_cur, c1_v_calc, c1_j_cur, c1_j_calc,
               c2_cdr3_cur, c2_cdr3_calc, c2_v_cur, c2_v_calc, c2_j_cur, c2_j_calc),
    count = c(count_non_empty(c1_cdr3_cur), count_non_empty(c1_cdr3_calc),
              count_non_empty(c1_v_cur), count_non_empty(c1_v_calc),
              count_non_empty(c1_j_cur), count_non_empty(c1_j_calc),
              count_non_empty(c2_cdr3_cur), count_non_empty(c2_cdr3_calc),
              count_non_empty(c2_v_cur), count_non_empty(c2_v_calc),
              count_non_empty(c2_j_cur), count_non_empty(c2_j_calc)),
    pct = round(100 * count / n_total, 1)
  )

  message("\nCoverage (n = ", format(n_total, big.mark = ","), "):")
  print(coverage)

  # Chain type distribution
  if (!is.null(c1_type) && !is.na(c1_type)) {
    message("\nChain 1 types:")
    print(table(iedb_data[[c1_type]], useNA = "ifany"))
  }

  if (!is.null(c2_type) && !is.na(c2_type)) {
    message("\nChain 2 types:")
    print(table(iedb_data[[c2_type]], useNA = "ifany"))
  }

  message(strrep("=", 60))

  invisible(coverage)
}

# ===== MHC Class Inference (V9.1) =====

#' Infer MHC class from IEDB allele string
#'
#' Handles various IEDB MHC allele formats including:
#'   - Placeholders: "HLA class I", "HLA class II"
#'   - Specific alleles: "HLA-A*02:01", "HLA-DRB1*04:01"
#'   - Mouse alleles: "H2-Kb", "H2-IAb"
#'
#' @param mhc_allele Character vector of MHC allele strings
#' @return Character vector: "MHCI", "MHCII", or NA
#' @keywords internal
infer_iedb_mhc_class <- function(mhc_allele) {

  case_when(
    is.na(mhc_allele) | mhc_allele == "" ~ NA_character_,

    # ----- Explicit class placeholders (anywhere in string) -----
    # Must check Class II before Class I (avoid "Class I" matching "Class II")
    str_detect(mhc_allele, regex("class\\s*II", ignore_case = TRUE)) ~ "MHCII",
    str_detect(mhc_allele, regex("class\\s*I(?!I)", ignore_case = TRUE)) ~ "MHCI",

    # ----- Human Class II: DR, DQ, DP genes -----
    str_detect(mhc_allele, "HLA-D[RPQAB]|DRB|DQB|DPB|DRA|DQA|DPA") ~ "MHCII",
    str_detect(mhc_allele, regex("^DR|^DQ|^DP", ignore_case = TRUE)) ~ "MHCII",

    # ----- Human Class I: A, B, C, E, F, G genes -----
    str_detect(mhc_allele, "HLA-[ABCEFG]") ~ "MHCI",

    # ----- Mouse Class II: I-A, I-E, H2-IA, H2-IE -----
    str_detect(mhc_allele, "H2-I[AE]|H-2-I[AE]|H-2I[AE]|^I-[AE]") ~ "MHCII",
    str_detect(mhc_allele, regex("H2.*class\\s*II|H-2.*class\\s*II", ignore_case = TRUE)) ~ "MHCII",

    # ----- Mouse Class I: K, D, L, Q (Qa), T (Tla) -----
    str_detect(mhc_allele, "H2-[KDLQT]|H-2[KDLQT]|H-2-[KDLQT]") ~ "MHCI",
    str_detect(mhc_allele, regex("H2.*class\\s*I(?!I)|H-2.*class\\s*I(?!I)", ignore_case = TRUE)) ~ "MHCI",

    # ----- Non-classical molecules -----
    str_detect(mhc_allele, "CD1|MR1|HLA-E|HLA-F|HLA-G") ~ "MHCI",
    str_detect(mhc_allele, "Qa-|Tla") ~ "MHCI",

    # ----- Macaque (for completeness, will be filtered later) -----
    str_detect(mhc_allele, "Mamu-[AB]") ~ "MHCI",
    str_detect(mhc_allele, "Mamu-D") ~ "MHCII",

    # ----- Default -----
    TRUE ~ NA_character_
  )
}

# ===== Curate IEDB for V7+ Paired Chains =====

#' Curate IEDB TCR data for V7+ paired-chain model
#'
#' Extracts both TRA (Chain 1) and TRB (Chain 2) information including
#' CDR3 sequences and V/J genes. Outputs format compatible with
#' extract_paired_chains() from VDJdb.
#'
#' V9.1 update: Improved MHC class inference using infer_iedb_mhc_class().
#'
#' @param iedb_data Raw IEDB data from load_iedb_tcr()
#' @param species_filter Species to include (default: human, mouse)
#' @return Tibble in same format as extract_paired_chains() output
#' @export
curate_iedb_for_paired_chains <- function(iedb_data,
                                          species_filter = c("human", "mouse")) {

  message("\n", strrep("=", 60))
  message("CURATING IEDB FOR V7+ PAIRED CHAINS")
  message(strrep("=", 60))
  message("\nStarting: ", format(nrow(iedb_data), big.mark = ","), " rows")

  valid_aa <- "ACDEFGHIKLMNPQRSTVWY"
  invalid_pattern <- paste0("[^", valid_aa, "]")

  # Find column names dynamically
  cols <- names(iedb_data)

  find_col <- function(pattern) {
    matches <- grep(pattern, cols, value = TRUE, ignore.case = TRUE)
    if (length(matches) > 0) matches[1] else NA_character_
  }

  # Map columns
  col_map <- list(
    epitope = find_col("^Epitope_Name$"),
    antigen_source = find_col("Epitope.*Source.*Organism"),
    assay_type = find_col("^Assay_Type$"),
    mhc_allele = find_col("Assay.*MHC.*Allele"),
    # Chain 1 (alpha)
    c1_cdr3_cur = find_col("Chain_1.*CDR3.*Curated$"),
    c1_cdr3_calc = find_col("Chain_1.*CDR3.*Calculated$"),
    c1_v_cur = find_col("Chain_1.*Curated.*V.*Gene"),
    c1_v_calc = find_col("Chain_1.*Calculated.*V.*Gene"),
    c1_j_cur = find_col("Chain_1.*Curated.*J.*Gene"),
    c1_j_calc = find_col("Chain_1.*Calculated.*J.*Gene"),
    c1_type = find_col("Chain_1.*Type$"),
    c1_organism = find_col("Chain_1.*Organism.*IRI"),
    # Chain 2 (beta)
    c2_cdr3_cur = find_col("Chain_2.*CDR3.*Curated$"),
    c2_cdr3_calc = find_col("Chain_2.*CDR3.*Calculated$"),
    c2_v_cur = find_col("Chain_2.*Curated.*V.*Gene"),
    c2_v_calc = find_col("Chain_2.*Calculated.*V.*Gene"),
    c2_j_cur = find_col("Chain_2.*Curated.*J.*Gene"),
    c2_j_calc = find_col("Chain_2.*Calculated.*J.*Gene"),
    c2_type = find_col("Chain_2.*Type$"),
    c2_organism = find_col("Chain_2.*Organism.*IRI")
  )

  # Check critical columns
  missing_critical <- c()
  if (is.na(col_map$epitope)) missing_critical <- c(missing_critical, "epitope")
  if (is.na(col_map$c2_cdr3_cur) && is.na(col_map$c2_cdr3_calc)) {
    missing_critical <- c(missing_critical, "Chain2_CDR3")
  }

  if (length(missing_critical) > 0) {
    stop("Missing critical columns: ", paste(missing_critical, collapse = ", "))
  }

  # Helper to coalesce curated/calculated values
  coalesce_cols <- function(data, cur_col, calc_col) {
    cur_vals <- if (!is.na(cur_col)) data[[cur_col]] else NA_character_
    calc_vals <- if (!is.na(calc_col)) data[[calc_col]] else NA_character_

    ifelse(!is.na(cur_vals) & cur_vals != "", cur_vals,
           ifelse(!is.na(calc_vals) & calc_vals != "", calc_vals, NA_character_))
  }

  # Helper to determine curation source
  get_source <- function(data, cur_col, calc_col) {
    cur_vals <- if (!is.na(cur_col)) data[[cur_col]] else NA_character_
    calc_vals <- if (!is.na(calc_col)) data[[calc_col]] else NA_character_

    case_when(
      !is.na(cur_vals) & cur_vals != "" ~ "curated",
      !is.na(calc_vals) & calc_vals != "" ~ "calculated",
      TRUE ~ NA_character_
    )
  }

  # ----- Extract and consolidate fields -----

  curated <- iedb_data %>%
    transmute(
      # Epitope info
      epitope_raw = .data[[col_map$epitope]],
      antigen_source = if (!is.na(col_map$antigen_source)) .data[[col_map$antigen_source]] else NA_character_,
      mhc_allele = if (!is.na(col_map$mhc_allele)) .data[[col_map$mhc_allele]] else NA_character_,

      # Chain 1 (alpha) - CDR3
      cdr3_alpha = coalesce_cols(iedb_data, col_map$c1_cdr3_cur, col_map$c1_cdr3_calc),
      c1_source = get_source(iedb_data, col_map$c1_cdr3_cur, col_map$c1_cdr3_calc),

      # Chain 1 (alpha) - V/J genes
      v_alpha = coalesce_cols(iedb_data, col_map$c1_v_cur, col_map$c1_v_calc),
      j_alpha = coalesce_cols(iedb_data, col_map$c1_j_cur, col_map$c1_j_calc),

      # Chain 2 (beta) - CDR3
      cdr3_beta = coalesce_cols(iedb_data, col_map$c2_cdr3_cur, col_map$c2_cdr3_calc),
      c2_source = get_source(iedb_data, col_map$c2_cdr3_cur, col_map$c2_cdr3_calc),

      # Chain 2 (beta) - V/J genes
      v_beta = coalesce_cols(iedb_data, col_map$c2_v_cur, col_map$c2_v_calc),
      j_beta = coalesce_cols(iedb_data, col_map$c2_j_cur, col_map$c2_j_calc),

      # Organism (prefer chain 2, fallback to chain 1)
      organism_iri = if (!is.na(col_map$c2_organism)) {
        ifelse(!is.na(.data[[col_map$c2_organism]]) & .data[[col_map$c2_organism]] != "",
               .data[[col_map$c2_organism]],
               if (!is.na(col_map$c1_organism)) .data[[col_map$c1_organism]] else NA_character_)
      } else if (!is.na(col_map$c1_organism)) {
        .data[[col_map$c1_organism]]
      } else {
        NA_character_
      }
    )

  # ----- Filter: require epitope + at least CDR3β -----

  curated <- curated %>%
    filter(
      !is.na(epitope_raw) & epitope_raw != "",
      !is.na(cdr3_beta) & cdr3_beta != ""
    )

  message("After requiring epitope + CDR3β: ", format(nrow(curated), big.mark = ","))

  # ----- Clean sequences -----

  curated <- curated %>%
    mutate(
      epitope = str_to_upper(str_trim(epitope_raw)),
      cdr3_alpha = str_to_upper(str_trim(cdr3_alpha)),
      cdr3_beta = str_to_upper(str_trim(cdr3_beta)),
      v_alpha = str_trim(v_alpha),
      j_alpha = str_trim(j_alpha),
      v_beta = str_trim(v_beta),
      j_beta = str_trim(j_beta)
    ) %>%
    # Filter valid sequences
    filter(
      !str_detect(epitope, invalid_pattern),
      !str_detect(cdr3_beta, invalid_pattern),
      between(nchar(cdr3_beta), 8, 25),
      between(nchar(epitope), 8, 30)
    ) %>%
    # Filter CDR3α if present
    filter(
      is.na(cdr3_alpha) | cdr3_alpha == "" |
        (!str_detect(cdr3_alpha, invalid_pattern) & between(nchar(cdr3_alpha), 8, 25))
    )

  message("After sequence cleaning: ", format(nrow(curated), big.mark = ","))

  # ----- Extract species from NCBI taxonomy IRI -----

  curated <- curated %>%
    mutate(
      species = case_when(
        str_detect(organism_iri, "9606") ~ "HomoSapiens",
        str_detect(organism_iri, "10090") ~ "MusMusculus",
        str_detect(organism_iri, "9544") ~ "MacacaMulatta",
        TRUE ~ "other"
      ),
      tcr_species = case_when(
        species == "HomoSapiens" ~ "human",
        species == "MusMusculus" ~ "mouse",
        TRUE ~ "other"
      )
    ) %>%
    filter(tcr_species %in% species_filter)

  message("After species filter: ", format(nrow(curated), big.mark = ","),
          " (", paste(species_filter, collapse = ", "), ")")

  # ----- Standardize V/J gene names -----

  curated <- curated %>%
    mutate(
      v_alpha = standardize_iedb_gene(v_alpha),
      j_alpha = standardize_iedb_gene(j_alpha),
      v_beta = standardize_iedb_gene(v_beta),
      j_beta = standardize_iedb_gene(j_beta)
    )

  # ----- Infer MHC class (V9.1: improved logic) -----

  curated <- curated %>%
    mutate(
      mhc_class_inferred = infer_iedb_mhc_class(mhc_allele)
    )

  # ----- Create quality score (0-3 scale, VDJdb-compatible) -----

  curated <- curated %>%
    mutate(
      has_alpha = !is.na(cdr3_alpha) & cdr3_alpha != "",
      has_beta = TRUE,  # Required by filter
      has_v_alpha = !is.na(v_alpha) & v_alpha != "",
      has_j_alpha = !is.na(j_alpha) & j_alpha != "",
      has_v_beta = !is.na(v_beta) & v_beta != "",
      has_j_beta = !is.na(j_beta) & j_beta != "",

      # Score: base on curation quality and completeness
      # 3 = curated CDR3 + complete V/J
      # 2 = curated CDR3 OR complete V/J
      # 1 = calculated with partial V/J
      # 0 = calculated, no V/J
      vj_count = has_v_alpha + has_j_alpha + has_v_beta + has_j_beta,
      score = case_when(
        c2_source == "curated" & vj_count >= 3 ~ 3L,
        c2_source == "curated" & vj_count >= 2 ~ 2L,
        c2_source == "curated" ~ 2L,
        c2_source == "calculated" & vj_count >= 2 ~ 2L,
        c2_source == "calculated" & vj_count >= 1 ~ 1L,
        TRUE ~ 0L
      ),

      is_paired = has_alpha & has_beta
    )

  # ----- Format output (compatible with extract_paired_chains) -----

  result <- curated %>%
    transmute(
      complex.id = NA_integer_,  # IEDB doesn't have complex IDs
      cdr3_alpha = ifelse(has_alpha, cdr3_alpha, NA_character_),
      v_alpha = ifelse(has_v_alpha, v_alpha, NA_character_),
      j_alpha = ifelse(has_j_alpha, j_alpha, NA_character_),
      cdr3_beta,
      v_beta = ifelse(has_v_beta, v_beta, NA_character_),
      j_beta = ifelse(has_j_beta, j_beta, NA_character_),
      epitope,
      species,
      antigen.species = antigen_source,
      antigen.gene = NA_character_,
      mhc.class = mhc_class_inferred,  # V9.1: Use inferred class
      mhc.a = mhc_allele,
      mhc.b = NA_character_,
      score,
      has_alpha,
      has_beta,
      is_paired,
      source_type = "IEDB"
    )

  # ----- Summary -----

  message("\n", strrep("-", 40))
  message("IEDB V9.1 curation complete:")
  message("  Total entries: ", format(nrow(result), big.mark = ","))
  message("  Paired (TRA+TRB): ", format(sum(result$is_paired), big.mark = ","),
          " (", round(100 * mean(result$is_paired), 1), "%)")
  message("  TRB-only: ", format(sum(!result$is_paired), big.mark = ","))
  message("  Unique epitopes: ", format(n_distinct(result$epitope), big.mark = ","))

  # V/J coverage
  message("\nV/J gene coverage:")
  message("  V_alpha: ", format(sum(!is.na(result$v_alpha)), big.mark = ","),
          " (", round(100 * mean(!is.na(result$v_alpha)), 1), "%)")
  message("  J_alpha: ", format(sum(!is.na(result$j_alpha)), big.mark = ","),
          " (", round(100 * mean(!is.na(result$j_alpha)), 1), "%)")
  message("  V_beta: ", format(sum(!is.na(result$v_beta)), big.mark = ","),
          " (", round(100 * mean(!is.na(result$v_beta)), 1), "%)")
  message("  J_beta: ", format(sum(!is.na(result$j_beta)), big.mark = ","),
          " (", round(100 * mean(!is.na(result$j_beta)), 1), "%)")

  # V9.1: MHC class coverage
  message("\nMHC class coverage:")
  mhc_class_counts <- table(result$mhc.class, useNA = "ifany")
  n_with_class <- sum(!is.na(result$mhc.class))
  message("  With MHC class: ", format(n_with_class, big.mark = ","),
          " (", round(100 * n_with_class / nrow(result), 1), "%)")
  message("  MHCI: ", format(sum(result$mhc.class == "MHCI", na.rm = TRUE), big.mark = ","))
  message("  MHCII: ", format(sum(result$mhc.class == "MHCII", na.rm = TRUE), big.mark = ","))
  message("  Unknown: ", format(sum(is.na(result$mhc.class)), big.mark = ","))

  message("\nBy species × score:")
  print(result %>%
          count(tcr_species = ifelse(species == "HomoSapiens", "human", "mouse"),
                score) %>%
          pivot_wider(names_from = score, values_from = n, values_fill = 0))

  message(strrep("-", 40))

  result
}

# ===== V/J Gene Standardization =====

#' Standardize IEDB V/J gene names to VDJdb format
#'
#' IEDB uses various formats: "TRBV6-5", "TRBV6-5*01", "V6-5", etc.
#' Standardizes to VDJdb format: "TRBV6-5*01"
#'
#' @param gene_name Gene name string
#' @return Standardized gene name
standardize_iedb_gene <- function(gene_name) {

  if (length(gene_name) == 0) return(character(0))

  result <- vapply(gene_name, function(g) {
    if (is.na(g) || g == "") return(NA_character_)

    g <- str_trim(g)

    # Remove common prefixes
    g <- str_remove(g, "^(Homo sapiens|Mus musculus|human|mouse)\\s*")

    # Add TR prefix if missing (e.g., "BV6-5" -> "TRBV6-5")
    if (!str_detect(g, "^TR[AB]")) {
      if (str_detect(g, "^[AB]?V")) {
        # Determine chain from context (V = variable)
        chain <- ifelse(str_detect(g, "^AV|^A.*V"), "TRA", "TRB")
        g <- str_replace(g, "^[AB]?V", paste0(chain, "V"))
      } else if (str_detect(g, "^[AB]?J")) {
        chain <- ifelse(str_detect(g, "^AJ|^A.*J"), "TRA", "TRB")
        g <- str_replace(g, "^[AB]?J", paste0(chain, "J"))
      }
    }

    # Add default allele if missing
    if (!str_detect(g, "\\*")) {
      g <- paste0(g, "*01")
    }

    g
  }, character(1), USE.NAMES = FALSE)

  result
}

# ===== Legacy Function (for backward compatibility) =====

#' Curate IEDB TCR data for model training (legacy, TRB-only)
#'
#' @param iedb_data Raw IEDB data from load_iedb_tcr()
#' @param species_filter Species to include
#' @return Curated tibble (TRB chain only)
#' @export
curate_iedb_for_tcr_model <- function(iedb_data,
                                      species_filter = c("human", "mouse")) {

  message("Note: Using legacy TRB-only curation. For V7+ paired chains, use curate_iedb_for_paired_chains()")

  # Use V7+ function and extract TRB-only format for backward compatibility
  paired <- curate_iedb_for_paired_chains(iedb_data, species_filter)

  paired %>%
    transmute(
      cdr3 = cdr3_beta,
      epitope,
      score,
      v_gene = v_beta,
      j_gene = j_beta,
      mhc_class = mhc.class,
      antigen_species = antigen.species,
      tcr_species = ifelse(species == "HomoSapiens", "human", "mouse"),
      source = "IEDB"
    )
}

# Helper for null coalescing
`%||%` <- function(x, y) if (is.null(x) || is.na(x)) y else x
