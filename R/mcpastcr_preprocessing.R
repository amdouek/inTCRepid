# McPAS-TCR Data Preprocessing for TCR-Epitope Model
#
# Functions to load and curate McPAS-TCR data for integration with VDJdb/IEDB.
# Data source: http://friedmanlab.weizmann.ac.il/McPAS-TCR/
#

library(data.table)
library(dplyr)
library(tidyr)
library(stringr)

# ===== Load McPAS-TCR Data =====

#' Load McPAS-TCR database
#'
#' @param filepath Path to McPAS-TCR.csv
#' @return data.table with raw McPAS-TCR data
#' @export
load_mcpas_tcr <- function(filepath = "data/mcpastcr/McPAS-TCR.csv") {

  message("Loading McPAS-TCR: ", filepath)

  mcpas <- fread(filepath, na.strings = c("", "NA", "null"))

  message("  Loaded ", format(nrow(mcpas), big.mark = ","), " rows x ", ncol(mcpas), " cols")

  # Quick summary
  message("  Species: ", paste(names(table(mcpas$Species)), collapse = ", "))
  message("  Categories: ", paste(names(table(mcpas$Category)), collapse = ", "))

  mcpas
}

# ===== V/J Gene Standardisation =====

#' Standardise McPAS-TCR V/J gene names to VDJdb format
#'
#' McPAS uses formats like "TRBV20-1", "TRBV1-01", "TRBJ2-3:01"
#' Standardises to VDJdb format: "TRBV20-1*01"
#'
#' @param gene_name Gene name string (vectorized)
#' @param chain Expected chain prefix ("TRB" or "TRA")
#' @return Standardised gene name
#' @export
standardize_mcpas_gene <- function(gene_name, chain = NULL) {

  if (length(gene_name) == 0) return(character(0))

  result <- vapply(gene_name, function(g) {
    if (is.na(g) || g == "") return(NA_character_)

    g <- str_trim(g)

    # Handle mouse-specific genes (e.g., "mTRDV2-2" -> keep as-is but standardise)
    is_mouse <- str_detect(g, "^m[A-Z]")

    # Remove leading 'm' for mouse genes temporarily
    if (is_mouse) {
      g <- str_sub(g, 2)
    }

    # Replace colon notation with asterisk (TRBJ2-3:01 -> TRBJ2-3*01)
    g <- str_replace(g, ":(\\d+)$", "*\\1")

    # Add *01 allele if no allele specified
    if (!str_detect(g, "\\*")) {
      g <- paste0(g, "*01")
    }

    # Ensure allele is two digits (*1 -> *01)
    g <- str_replace(g, "\\*(\\d)$", "*0\\1")

    # Re-add 'm' prefix for mouse genes
    if (is_mouse) {
      g <- paste0("m", g)
    }

    g
  }, character(1), USE.NAMES = FALSE)

  result
}

# ===== Quality Filtering =====

#' Identify entries to exclude based on Remarks column
#'
#' @param remarks Character vector of remarks
#' @return Logical vector (TRUE = exclude)
#' @export
should_exclude_by_remark <- function(remarks) {

  exclude_patterns <- c(
    "Stop codon",
    "No final F",
    "No first Cysteine",
    "Short Sequence"
  )

  # Create pattern
  pattern <- paste(exclude_patterns, collapse = "|")

  # Return TRUE for entries to exclude
  !is.na(remarks) & str_detect(remarks, pattern)
}

# ===== Main Curation Function =====

#' Curate McPAS-TCR data for V7+ paired-chain model
#'
#' Extracts and standardizes both TRA and TRB chain information.
#' Output format is compatible with combine_paired_sources().
#'
#' @param mcpas_data Raw McPAS-TCR data from load_mcpas_tcr()
#' @param species_filter Species to include (default: human, mouse)
#' @return Tibble in same format as curate_iedb_for_paired_chains() output
#' @export
curate_mcpas_for_paired_chains <- function(mcpas_data,
                                           species_filter = c("Human", "Mouse")) {

  message("\n", strrep("=", 60))
  message("CURATING McPAS-TCR FOR PAIRED CHAINS")
  message(strrep("=", 60))
  message("\nStarting: ", format(nrow(mcpas_data), big.mark = ","), " rows")

  valid_aa <- "ACDEFGHIKLMNPQRSTVWY"
  invalid_pattern <- paste0("[^", valid_aa, "]")

  # ----- Step 1: Basic field extraction -----

  curated <- mcpas_data %>%
    transmute(
      # CDR3 sequences
      cdr3_alpha = CDR3.alpha.aa,
      cdr3_beta = CDR3.beta.aa,

      # V/J genes (will standardise later)
      v_alpha_raw = TRAV,
      j_alpha_raw = TRAJ,
      v_beta_raw = TRBV,
      j_beta_raw = TRBJ,

      # Epitope
      epitope_raw = Epitope.peptide,

      # Metadata
      species_raw = Species,
      category = Category,
      pathology = Pathology,
      antigen_protein = Antigen.protein,
      mhc_raw = MHC,
      t_cell_type = `T.Cell.Type`,

      # Quality flags
      remarks = Remarks
    )

  # ----- Step 2: Filter by species -----

  curated <- curated %>%
    filter(species_raw %in% species_filter)

  message("After species filter: ", format(nrow(curated), big.mark = ","),
          " (", paste(species_filter, collapse = ", "), ")")

  # ----- Step 3: Require epitope + CDR3β -----

  curated <- curated %>%
    filter(
      !is.na(epitope_raw) & epitope_raw != "",
      !is.na(cdr3_beta) & cdr3_beta != ""
    )

  message("After requiring epitope + CDR3β: ", format(nrow(curated), big.mark = ","))

  # ----- Step 4: Exclude by quality remarks -----

  exclude_mask <- should_exclude_by_remark(curated$remarks)
  n_excluded_remarks <- sum(exclude_mask)

  curated <- curated %>% filter(!should_exclude_by_remark(remarks))

  message("After excluding quality issues: ", format(nrow(curated), big.mark = ","),
          " (removed ", n_excluded_remarks, " by remarks)")

  # ----- Step 5: Clean sequences -----

  curated <- curated %>%
    mutate(
      # Uppercase and trim
      epitope = str_to_upper(str_trim(epitope_raw)),
      cdr3_alpha = str_to_upper(str_trim(cdr3_alpha)),
      cdr3_beta = str_to_upper(str_trim(cdr3_beta))
    ) %>%
    # Filter valid epitope sequences
    filter(
      !str_detect(epitope, invalid_pattern),
      between(nchar(epitope), 8, 25)
    ) %>%
    # Filter valid CDR3β sequences
    filter(
      !str_detect(cdr3_beta, invalid_pattern),
      between(nchar(cdr3_beta), 8, 25)
    ) %>%
    # Filter valid CDR3α sequences (if present)
    filter(
      is.na(cdr3_alpha) | cdr3_alpha == "" |
        (!str_detect(cdr3_alpha, invalid_pattern) & between(nchar(cdr3_alpha), 8, 25))
    )

  message("After sequence cleaning: ", format(nrow(curated), big.mark = ","))

  # ----- Step 6: Standardise V/J gene names -----

  curated <- curated %>%
    mutate(
      v_alpha = standardize_mcpas_gene(v_alpha_raw),
      j_alpha = standardize_mcpas_gene(j_alpha_raw),
      v_beta = standardize_mcpas_gene(v_beta_raw),
      j_beta = standardize_mcpas_gene(j_beta_raw)
    )

  # ----- Step 7: Standardise species names -----

  curated <- curated %>%
    mutate(
      species = case_when(
        species_raw == "Human" ~ "HomoSapiens",
        species_raw == "Mouse" ~ "MusMusculus",
        TRUE ~ "other"
      ),
      tcr_species = case_when(
        species == "HomoSapiens" ~ "human",
        species == "MusMusculus" ~ "mouse",
        TRUE ~ "other"
      )
    )

  # ----- Step 8: Infer MHC class from T cell type -----

  curated <- curated %>%
    mutate(
      mhc_class = case_when(
        t_cell_type == "CD8" ~ "MHCI",
        t_cell_type == "CD4" ~ "MHCII",
        str_detect(t_cell_type, "CD4.*CD8|CD8.*CD4") ~ NA_character_,
        TRUE ~ NA_character_
      )
    )

  # ----- Step 9: Create quality score -----

  # McPAS doesn't have explicit scores, so we create one based on completeness
  curated <- curated %>%
    mutate(
      has_alpha = !is.na(cdr3_alpha) & cdr3_alpha != "",
      has_beta = TRUE,  # Required by earlier filter
      has_v_alpha = !is.na(v_alpha) & v_alpha != "",
      has_j_alpha = !is.na(j_alpha) & j_alpha != "",
      has_v_beta = !is.na(v_beta) & v_beta != "",
      has_j_beta = !is.na(j_beta) & j_beta != "",
      has_mhc = !is.na(mhc_raw) & mhc_raw != "",

      # Score: based on completeness (0-3 scale like VDJdb)
      vj_count = has_v_alpha + has_j_alpha + has_v_beta + has_j_beta,
      score = case_when(
        has_alpha & vj_count >= 3 & has_mhc ~ 3L,
        has_alpha & vj_count >= 2 ~ 2L,
        vj_count >= 2 ~ 2L,
        vj_count >= 1 ~ 1L,
        TRUE ~ 0L
      ),

      is_paired = has_alpha & has_beta
    )

  # ----- Step 10: Format output (compatible with combine_paired_sources) -----

  result <- curated %>%
    transmute(
      complex.id = NA_integer_,  # McPAS doesn't have complex IDs
      cdr3_alpha = ifelse(has_alpha, cdr3_alpha, NA_character_),
      v_alpha = ifelse(has_v_alpha, v_alpha, NA_character_),
      j_alpha = ifelse(has_j_alpha, j_alpha, NA_character_),
      cdr3_beta,
      v_beta = ifelse(has_v_beta, v_beta, NA_character_),
      j_beta = ifelse(has_j_beta, j_beta, NA_character_),
      epitope,
      species,
      antigen.species = category,  # Use category as approximate antigen source
      antigen.gene = antigen_protein,
      mhc.class = mhc_class,
      mhc.a = mhc_raw,
      mhc.b = NA_character_,
      score,
      has_alpha,
      has_beta,
      is_paired,
      source_type = "McPAS"
    )

  # ----- Summary -----

  message("\n", strrep("-", 40))
  message("McPAS-TCR curation complete:")
  message("  Total entries: ", format(nrow(result), big.mark = ","))
  message("  Paired (TRA+TRB): ", format(sum(result$is_paired), big.mark = ","),
          " (", round(100 * mean(result$is_paired), 1), "%)")
  message("  TRB-only: ", format(sum(!result$is_paired), big.mark = ","))
  message("  Unique epitopes: ", format(n_distinct(result$epitope), big.mark = ","))

  message("\nV/J gene coverage:")
  message("  V_alpha: ", format(sum(!is.na(result$v_alpha)), big.mark = ","),
          " (", round(100 * mean(!is.na(result$v_alpha)), 1), "%)")
  message("  J_alpha: ", format(sum(!is.na(result$j_alpha)), big.mark = ","),
          " (", round(100 * mean(!is.na(result$j_alpha)), 1), "%)")
  message("  V_beta: ", format(sum(!is.na(result$v_beta)), big.mark = ","),
          " (", round(100 * mean(!is.na(result$v_beta)), 1), "%)")
  message("  J_beta: ", format(sum(!is.na(result$j_beta)), big.mark = ","),
          " (", round(100 * mean(!is.na(result$j_beta)), 1), "%)")

  message("\nBy species:")
  print(result %>% count(species))

  message("\nBy score:")
  print(result %>% count(score) %>% arrange(desc(score)))

  message(strrep("-", 40))

  result
}

# ===== Overlap Analysis =====

#' Analyze epitope overlap between McPAS-TCR and other sources
#'
#' @param mcpas_curated Curated McPAS-TCR data
#' @param vdjdb_paired VDJdb paired chain data
#' @param iedb_paired IEDB paired chain data (optional)
#' @return Summary tibble
#' @export
analyze_mcpas_overlap <- function(mcpas_curated,
                                  vdjdb_paired,
                                  iedb_paired = NULL) {

  message("\n", strrep("=", 60))
  message("McPAS-TCR OVERLAP ANALYSIS")
  message(strrep("=", 60))

  mcpas_epitopes <- unique(mcpas_curated$epitope)
  vdjdb_epitopes <- unique(vdjdb_paired$epitope)

  message("\nMcPAS epitopes: ", length(mcpas_epitopes))
  message("VDJdb epitopes: ", length(vdjdb_epitopes))

  overlap_vdjdb <- intersect(mcpas_epitopes, vdjdb_epitopes)
  mcpas_only <- setdiff(mcpas_epitopes, vdjdb_epitopes)

  message("\nOverlap with VDJdb: ", length(overlap_vdjdb),
          " (", round(100 * length(overlap_vdjdb) / length(mcpas_epitopes), 1), "% of McPAS)")
  message("McPAS-only epitopes: ", length(mcpas_only))

  if (!is.null(iedb_paired)) {
    iedb_epitopes <- unique(iedb_paired$epitope)
    message("\nIEDB epitopes: ", length(iedb_epitopes))

    overlap_iedb <- intersect(mcpas_epitopes, iedb_epitopes)
    message("Overlap with IEDB: ", length(overlap_iedb),
            " (", round(100 * length(overlap_iedb) / length(mcpas_epitopes), 1), "% of McPAS)")

    all_epitopes <- unique(c(vdjdb_epitopes, iedb_epitopes))
    novel_mcpas <- setdiff(mcpas_epitopes, all_epitopes)
    message("\nMcPAS epitopes not in VDJdb or IEDB: ", length(novel_mcpas))

    if (length(novel_mcpas) > 0 && length(novel_mcpas) <= 20) {
      message("\nNovel McPAS epitopes:")
      print(novel_mcpas)
    }
  }

  # TCR-level overlap (CDR3β + epitope pairs)
  message("\n", strrep("-", 40))
  message("TCR-EPITOPE PAIR OVERLAP")
  message(strrep("-", 40))

  mcpas_pairs <- mcpas_curated %>%
    mutate(pair_key = paste(cdr3_beta, epitope, sep = "_")) %>%
    pull(pair_key) %>% unique()

  vdjdb_pairs <- vdjdb_paired %>%
    filter(!is.na(cdr3_beta)) %>%
    mutate(pair_key = paste(cdr3_beta, epitope, sep = "_")) %>%
    pull(pair_key) %>% unique()

  pair_overlap <- length(intersect(mcpas_pairs, vdjdb_pairs))
  message("\nMcPAS unique TCR-epitope pairs: ", format(length(mcpas_pairs), big.mark = ","))
  message("VDJdb unique TCR-epitope pairs: ", format(length(vdjdb_pairs), big.mark = ","))
  message("Overlapping pairs: ", format(pair_overlap, big.mark = ","),
          " (", round(100 * pair_overlap / length(mcpas_pairs), 1), "% of McPAS)")
  message("Novel McPAS pairs: ", format(length(mcpas_pairs) - pair_overlap, big.mark = ","))

  message(strrep("=", 60))

  invisible(list(
    mcpas_epitopes = mcpas_epitopes,
    vdjdb_overlap = overlap_vdjdb,
    mcpas_only = mcpas_only,
    pair_overlap = pair_overlap
  ))
}
