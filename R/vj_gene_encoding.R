# V/J gene encoding utilities (V7)
# Categorical encoding of germline gene segments for model integration

library(dplyr)
library(stringr)
library(tidyr)

# ===== Special Token Indices =====
# Used consistently across vocabulary building and encoding
.VJ_SPECIAL <- list(PAD = 0L, UNK = 1L, MISSING = 2L)

# ===== Vocabulary Construction =====

#' Build V/J gene vocabulary from VDJdb data
#'
#' Creates gene-to-index mappings for embedding layers.
#' Special tokens: <PAD>=0, <UNK>=1, <MISSING>=2, genes start at 3.
#'
#' @param vdjdb_data VDJdb data frame
#' @param chain Chain filter ("TRB", "TRA", or "both")
#' @param min_count Minimum occurrences to include gene
#' @return List with v/j vocabularies and metadata
#' @export
build_vj_vocabulary <- function(vdjdb_data, chain = "TRB", min_count = 1) {

  message("Building V/J gene vocabulary...")

  # Filter by chain
  data <- if (chain != "both") filter(vdjdb_data, gene == chain) else vdjdb_data
  message(sprintf("  Chain: %s (%s entries)", chain, format(nrow(data), big.mark = ",")))

  # Helper to build single vocabulary
  build_gene_vocab <- function(data, col) {
    genes <- data %>%
      filter(!is.na(.data[[col]]) & .data[[col]] != "") %>%
      count(.data[[col]], name = "count") %>%
      filter(count >= min_count) %>%
      arrange(desc(count))

    vocab <- c("<PAD>", "<UNK>", "<MISSING>", genes[[col]])
    to_idx <- setNames(seq_along(vocab) - 1L, vocab)
    idx_to <- setNames(vocab, as.character(seq_along(vocab) - 1L))

    list(vocab = vocab, to_idx = to_idx, idx_to = idx_to,
         size = length(vocab), counts = genes)
  }

  v_vocab <- build_gene_vocab(data, "v.segm")
  j_vocab <- build_gene_vocab(data, "j.segm")

  message(sprintf("  V genes: %d (vocab: %d) | J genes: %d (vocab: %d)",
                  nrow(v_vocab$counts), v_vocab$size,
                  nrow(j_vocab$counts), j_vocab$size))

  list(chain = chain, v = v_vocab, j = j_vocab, special_tokens = .VJ_SPECIAL)
}

# ===== Encoding Functions =====

#' Encode V/J genes to integer indices
#'
#' @param genes Character vector of gene names
#' @param vocab Vocabulary from build_vj_vocabulary()
#' @param gene_type "v" or "j"
#' @return Integer vector of indices
#' @export
encode_vj_genes <- function(genes, vocab, gene_type = "v") {

  to_idx <- vocab[[gene_type]]$to_idx
  unk_idx <- .VJ_SPECIAL$UNK
  missing_idx <- .VJ_SPECIAL$MISSING

  vapply(genes, function(g) {
    if (is.na(g) || g == "") missing_idx
    else if (g %in% names(to_idx)) to_idx[[g]]
    else unk_idx
  }, integer(1), USE.NAMES = FALSE)
}

#' Encode V and J genes for a dataset
#'
#' @param data Data frame with v.segm/j.segm or v_gene/j_gene columns
#' @param vocab V/J vocabulary from build_vj_vocabulary()
#' @return Data frame with v_idx and j_idx columns added
#' @export
encode_vj_for_dataset <- function(data, vocab) {

  # Handle naming conventions
  v_col <- intersect(c("v.segm", "v_gene"), names(data))[1]
  j_col <- intersect(c("j.segm", "j_gene"), names(data))[1]

  if (is.na(v_col) || is.na(j_col)) {
    stop("V/J gene columns not found. Expected 'v.segm'/'j.segm' or 'v_gene'/'j_gene'")
  }

  data <- data %>%
    mutate(v_idx = encode_vj_genes(.data[[v_col]], vocab, "v"),
           j_idx = encode_vj_genes(.data[[j_col]], vocab, "j"))

  # Report unknown/missing
  n_v_unk <- sum(data$v_idx == .VJ_SPECIAL$UNK)
  n_j_unk <- sum(data$j_idx == .VJ_SPECIAL$UNK)
  n_v_miss <- sum(data$v_idx == .VJ_SPECIAL$MISSING)
  n_j_miss <- sum(data$j_idx == .VJ_SPECIAL$MISSING)

  if (n_v_unk + n_j_unk > 0)
    message(sprintf("  V/J unknown: %d V, %d J", n_v_unk, n_j_unk))
  if (n_v_miss + n_j_miss > 0)
    message(sprintf("  V/J missing: %d V, %d J", n_v_miss, n_j_miss))

  data
}

# ===== Gene Name Standardization =====

#' Standardize V/J gene names across databases
#'
#' Handles naming variations between VDJdb, IEDB, and McPAS.
#'
#' @param gene_name Gene name string
#' @param target_format Target format ("vdjdb", "imgt")
#' @return Standardized gene name
#' @export
standardize_gene_name <- function(gene_name, target_format = "vdjdb") {

  if (is.na(gene_name) || gene_name == "") return(NA_character_)

  gene_name %>%
    str_trim() %>%
    str_replace_all("/", "-") %>%                              # Standardize separator
    str_replace("(?<!\\*)([0-9]{2})$", "*\\1") %>%             # Ensure allele format
    str_remove("^(human|mouse|Homo sapiens|Mus musculus)\\s*") # Remove species prefix
}

#' Batch standardize gene names in a data frame
#'
#' @param data Data frame
#' @param v_col V gene column name
#' @param j_col J gene column name
#' @return Data frame with standardized names
#' @export
standardize_vj_genes <- function(data, v_col = "v.segm", j_col = "j.segm") {

  if (v_col %in% names(data))
    data[[v_col]] <- vapply(data[[v_col]], standardize_gene_name,
                            character(1), USE.NAMES = FALSE)
  if (j_col %in% names(data))
    data[[j_col]] <- vapply(data[[j_col]], standardize_gene_name,
                            character(1), USE.NAMES = FALSE)
  data
}

# ===== Persistence =====

#' Save V/J vocabulary
#' @param vocab Vocabulary list
#' @param filepath Output path (RDS)
#' @export
save_vj_vocabulary <- function(vocab, filepath) {
  saveRDS(vocab, filepath)
  message("V/J vocabulary saved: ", filepath)
}

#' Load V/J vocabulary
#' @param filepath Path to vocabulary file
#' @return Vocabulary list
#' @export
load_vj_vocabulary <- function(filepath) {
  vocab <- readRDS(filepath)
  message(sprintf("V/J vocabulary loaded: %d V, %d J genes", vocab$v$size, vocab$j$size))
  vocab
}

# ===== Diagnostics =====

#' Print V/J vocabulary summary
#' @param vocab Vocabulary list
#' @export
print_vj_vocab_summary <- function(vocab) {

  cat("\n", strrep("=", 50), "\n")
  cat("V/J GENE VOCABULARY SUMMARY\n")
  cat(strrep("=", 50), "\n")
  cat("\nChain:", vocab$chain, "\n")

  for (type in c("v", "j")) {
    cat(sprintf("\n%s Genes (vocab size: %d):\n", toupper(type), vocab[[type]]$size))
    cat("  Special: <PAD>=0, <UNK>=1, <MISSING>=2\n")
    cat("  Top 5:\n")
    print(head(vocab[[type]]$counts, 5))
  }

  cat(strrep("=", 50), "\n\n")
}

#' Analyze V/J gene encoding coverage
#' @param data Encoded data frame with v_idx, j_idx
#' @param vocab Vocabulary used for encoding
#' @return Invisible list with coverage stats
#' @export
analyze_vj_coverage <- function(data, vocab) {

  message("\nV/J Gene Encoding Coverage:")
  n <- nrow(data)

  coverage <- lapply(c(v = "v_idx", j = "j_idx"), function(col) {
    idx <- data[[col]]
    known <- sum(idx >= 3)
    unk <- sum(idx == .VJ_SPECIAL$UNK)
    miss <- sum(idx == .VJ_SPECIAL$MISSING)

    type <- toupper(substr(col, 1, 1))
    message(sprintf("  %s: %d known (%.1f%%), %d unknown (%.1f%%), %d missing (%.1f%%)",
                    type, known, 100 * known / n, unk, 100 * unk / n, miss, 100 * miss / n))

    list(known = known, unknown = unk, missing = miss)
  })

  invisible(coverage)
}
