# VDJdb Data Acquisition and Loading

library(data.table)
library(dplyr)
library(stringr)

# ===== Helper Functions =====

#' Format file size in MB
format_mb <- function(path) {

  if (file.exists(path)) round(file.size(path) / 1024^2, 2) else NA
}

#' Find column by pattern (case-insensitive)
find_col <- function(df, pattern) {
  matches <- grep(pattern, names(df), ignore.case = TRUE, value = TRUE)
  if (length(matches) > 0) matches[1] else NA_character_
}

# ===== Download Functions =====

#' Get VDJdb release URL
#'
#' @param use_latest If TRUE, get most recent release
#' @param version Specific version date (e.g., "2025-07-30") if use_latest=FALSE
#' @return URL to release zip file
get_vdjdb_release_url <- function(use_latest = TRUE, version = NULL) {

  url <- "https://raw.githubusercontent.com/antigenomics/vdjdb-db/master/latest-version.txt"

  response <- tryCatch(
    readLines(url, warn = FALSE),
    error = function(e) stop("Failed to fetch VDJdb release info: ", e$message)
  )

  release_urls <- response[grepl("^https://", response)]
  if (length(release_urls) == 0) stop("No release URLs found")


  if (use_latest) {
    selected <- release_urls[1]
  } else {
    if (is.null(version)) stop("Specify version when use_latest=FALSE")
    idx <- grep(version, release_urls, fixed = TRUE)
    if (length(idx) == 0) {
      versions <- str_extract(head(release_urls, 5), "\\d{4}-\\d{2}-\\d{2}")
      stop("Version '", version, "' not found. Available: ", paste(versions, collapse = ", "))
    }
    selected <- release_urls[idx[1]]
  }

  message("Release: ", basename(selected))
  selected
}


#' List available VDJdb releases
#' @return Tibble of available releases
list_vdjdb_releases <- function() {

  url <- "https://raw.githubusercontent.com/antigenomics/vdjdb-db/master/latest-version.txt"
  response <- readLines(url, warn = FALSE)
  release_urls <- response[grepl("^https://", response)]

  releases <- tibble(url = release_urls) %>%
    mutate(
      filename = basename(url),
      date = str_extract(filename, "\\d{4}-\\d{2}-\\d{2}"),
      tag = str_extract(url, "download/([^/]+)/", group = 1)
    ) %>%
    arrange(desc(date))

  message("Available releases: ", nrow(releases))
  print(select(head(releases, 10), date, tag, filename))
  releases
}


#' Download VDJdb database
#'
#' Downloads pre-built VDJdb from GitHub releases.
#'
#' @param output_dir Output directory
#' @param use_latest Download most recent release
#' @param version Specific version if use_latest=FALSE
#' @param force_download Re-download even if files exist
#' @return List with paths to database files
download_vdjdb <- function(output_dir = "data",
                           use_latest = TRUE,
                           version = NULL,
                           force_download = FALSE) {

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  output_files <- list(
    vdjdb = file.path(output_dir, "vdjdb.txt"),
    vdjdb_slim = file.path(output_dir, "vdjdb.slim.txt"),
    vdjdb_full = file.path(output_dir, "vdjdb_full.txt")
  )

  # Check existing
  if (file.exists(output_files$vdjdb) && !force_download) {
    message("VDJdb exists in ", output_dir, " (use force_download=TRUE to re-download)")
    return(output_files)
  }

  # Download
  release_url <- get_vdjdb_release_url(use_latest, version)
  temp_zip <- tempfile(fileext = ".zip")

  message("Downloading VDJdb...")
  download.file(release_url, temp_zip, mode = "wb", quiet = FALSE)

  if (!file.exists(temp_zip) || file.size(temp_zip) == 0) {
    stop("Download failed")
  }
  message("Downloaded: ", format_mb(temp_zip), " MB")

  # Extract
  temp_dir <- tempfile(pattern = "vdjdb_")
  dir.create(temp_dir)
  unzip(temp_zip, exdir = temp_dir)

  txt_files <- list.files(temp_dir, pattern = "\\.txt$",
                          recursive = TRUE, full.names = TRUE)

  for (f in txt_files) {
    file.copy(f, file.path(output_dir, basename(f)), overwrite = TRUE)
    message("  ", basename(f), " (", format_mb(f), " MB)")
  }

  # Cleanup
  unlink(c(temp_zip, temp_dir), recursive = TRUE)

  message("VDJdb saved to: ", output_dir)
  output_files
}

# ===== Load and Filter Functions =====

#' Load VDJdb from file
#'
#' @param file_path Path to vdjdb.txt
#' @param verbose Print summary statistics
#' @return Tibble with VDJdb data
load_vdjdb <- function(file_path, verbose = TRUE) {

  if (!file.exists(file_path)) stop("File not found: ", file_path)

  message("Loading: ", file_path, " (", format_mb(file_path), " MB)")

  vdjdb <- fread(file_path, sep = "\t", header = TRUE, quote = "",
                 fill = TRUE, encoding = "UTF-8") %>%
    as_tibble()

  message("Loaded ", format(nrow(vdjdb), big.mark = ","), " entries, ",
          ncol(vdjdb), " columns")

  if (verbose) print_vdjdb_summary(vdjdb)

  vdjdb
}


#' Print VDJdb summary statistics
#' @param vdjdb VDJdb tibble
print_vdjdb_summary <- function(vdjdb) {

  cols <- get_vdjdb_cols(vdjdb)

  cat("\n", strrep("-", 50), "\n")

  # Species
  if (!is.na(cols$species)) {
    cat("Host species:\n")
    print(table(vdjdb[[cols$species]]))
  }

  # Chains
  if (!is.na(cols$gene)) {
    cat("\nChains:\n")
    print(table(vdjdb[[cols$gene]]))
  }

  # Scores
  if (!is.na(cols$score)) {
    cat("\nScores:\n")
    print(table(vdjdb[[cols$score]], useNA = "ifany"))
  }

  cat(strrep("-", 50), "\n")
}


#' Filter VDJdb data
#'
#' Flexible filtering for TCR species, antigen species, chain, and score.
#'
#' @param vdjdb VDJdb tibble
#' @param tcr_species Host species ("HomoSapiens", "MusMusculus", or "all")
#' @param antigen_species Antigen source species (or "all")
#' @param chain Chain type: "TRA", "TRB", or "both"
#' @param min_score Minimum confidence score (0-3)
#' @param exclude_self Exclude self-antigens (where antigen_species == tcr_species)
#' @param verbose Print filtering summary
#' @return Filtered tibble
filter_vdjdb <- function(vdjdb,
                         tcr_species = "all",
                         antigen_species = "all",
                         chain = "TRB",
                         min_score = 0,
                         exclude_self = FALSE,
                         verbose = TRUE) {

  cols <- get_vdjdb_cols(vdjdb)
  filtered <- vdjdb
  n_start <- nrow(filtered)

  # TCR species
  if (tcr_species != "all" && !is.na(cols$species)) {
    filtered <- filter(filtered, .data[[cols$species]] == tcr_species)
  }

  # Antigen species
  if (antigen_species != "all" && !is.na(cols$antigen_species)) {
    filtered <- filter(filtered, .data[[cols$antigen_species]] == antigen_species)
  }

  # Exclude self
  if (exclude_self && !is.na(cols$antigen_species) && !is.na(cols$species)) {
    filtered <- filter(filtered, .data[[cols$antigen_species]] != .data[[cols$species]])
  }

  # Chain
  if (chain != "both" && !is.na(cols$gene)) {
    filtered <- filter(filtered, .data[[cols$gene]] == chain)
  }

  # Score
  if (min_score > 0 && !is.na(cols$score)) {
    filtered <- filter(filtered, .data[[cols$score]] >= min_score)
  }

  if (verbose) {
    message(sprintf("Filtered: %s → %s entries",
                    format(n_start, big.mark = ","),
                    format(nrow(filtered), big.mark = ",")))

    if (!is.na(cols$cdr3)) {
      message("  Unique CDR3: ", format(n_distinct(filtered[[cols$cdr3]]), big.mark = ","))
    }
    if (!is.na(cols$epitope)) {
      message("  Unique epitopes: ", format(n_distinct(filtered[[cols$epitope]]), big.mark = ","))
    }
  }

  filtered
}


#' Convenience wrapper for human TCR data
#'
#' @param vdjdb VDJdb tibble
#' @param ... Additional arguments passed to filter_vdjdb()
#' @return Filtered tibble with human TCRs
filter_vdjdb_human <- function(vdjdb, ...) {
  filter_vdjdb(vdjdb, tcr_species = "HomoSapiens", ...)
}


#' Convenience wrapper for mouse TCR data
#'
#' @param vdjdb VDJdb tibble
#' @param ... Additional arguments passed to filter_vdjdb()
#' @return Filtered tibble with mouse TCRs
filter_vdjdb_mouse <- function(vdjdb, ...) {
  filter_vdjdb(vdjdb, tcr_species = "MusMusculus", ...)
}

# ===== Analysis Functions =====

#' Analyze VDJdb composition
#'
#' @param vdjdb VDJdb tibble
#' @return Summary tibble
analyze_vdjdb_composition <- function(vdjdb) {

  cols <- get_vdjdb_cols(vdjdb)

  cat("\n", strrep("=", 60), "\n")
  cat("VDJdb COMPOSITION ANALYSIS\n")
  cat(strrep("=", 60), "\n")

  cat("\nTotal entries:", format(nrow(vdjdb), big.mark = ","), "\n")

  # By TCR species
  if (!is.na(cols$species)) {
    cat("\nBy TCR species:\n")
    species_tbl <- vdjdb %>% count(.data[[cols$species]], sort = TRUE)
    for (i in seq_len(nrow(species_tbl))) {
      pct <- round(100 * species_tbl$n[i] / nrow(vdjdb), 1)
      cat("  ", species_tbl[[cols$species]][i], ": ",
          format(species_tbl$n[i], big.mark = ","), " (", pct, "%)\n", sep = "")
    }
  }

  # Human breakdown
  human <- vdjdb %>% filter(.data[[cols$species]] == "HomoSapiens")
  cat("\nHuman TCR entries:", format(nrow(human), big.mark = ","), "\n")

  # By chain
  if (!is.na(cols$gene)) {
    cat("  By chain: ")
    cat(paste(names(table(human[[cols$gene]])), "=",
              format(as.vector(table(human[[cols$gene]])), big.mark = ","),
              collapse = ", "), "\n")
  }

  # Top antigen sources
  if (!is.na(cols$antigen_species)) {
    cat("\n  Top antigen sources:\n")
    ag_tbl <- human %>% count(.data[[cols$antigen_species]], sort = TRUE) %>% head(10)
    for (i in seq_len(nrow(ag_tbl))) {
      cat("    ", ag_tbl[[cols$antigen_species]][i], ": ",
          format(ag_tbl$n[i], big.mark = ","), "\n", sep = "")
    }
  }

  # Scores
  if (!is.na(cols$score)) {
    cat("\n  By score: ")
    score_tbl <- table(human[[cols$score]])
    cat(paste(names(score_tbl), "=", format(as.vector(score_tbl), big.mark = ","),
              collapse = ", "), "\n")
  }

  cat(strrep("=", 60), "\n\n")

  # Return summary tibble
  human_trb <- human %>% filter(.data[[cols$gene]] == "TRB")

  tibble(
    metric = c("total", "human", "human_trb", "human_trb_epitopes", "mouse"),
    value = c(
      nrow(vdjdb),
      nrow(human),
      nrow(human_trb),
      n_distinct(human_trb[[cols$epitope]]),
      nrow(filter(vdjdb, .data[[cols$species]] == "MusMusculus"))
    )
  )
}

# ===== Example Usage =====
#
# # Download and load
# vdjdb_files <- download_vdjdb("data")
# vdjdb <- load_vdjdb(vdjdb_files$vdjdb)
#
# # Filter for mouse TCRs recognizing mouse antigens
# vdjdb_mouse <- filter_vdjdb(vdjdb,
#                             tcr_species = "MusMusculus",
#                             antigen_species = "MusMusculus",
#                             chain = "TRB")
#
# # Filter for human TCRs (all antigens)
# vdjdb_human <- filter_vdjdb_human(vdjdb, chain = "TRB")
#
# # Analyze composition
# composition <- analyze_vdjdb_composition(vdjdb)
