# Amino Acid Sequence Encoding
#
# Primary encoding: sequences_to_indices() - converts AA sequences to integer
# indices for PyTorch embedding layers.
#
# Also includes physicochemical encoding functions for analysis/visualization.
#

library(reticulate)

# ===== Amino Acid Properties =====

# Standard amino acid alphabet
AA_STANDARD <- c("A","C","D","E","F","G","H","I","K","L",
                 "M","N","P","Q","R","S","T","V","W","Y")

#' Atchley physicochemical factors
#'
#' Five principal components: polarity, secondary structure, volume, codon diversity, charge
#' Reference: Atchley et al., 2005 - PNAS
#'
#' @return Named list of 5-element vectors
get_atchley_factors <- function() {
  list(
    A = c(-0.591, -1.302, -0.733,  1.570, -0.146),
    C = c(-1.343,  0.465, -0.862, -1.020, -0.255),
    D = c( 1.050,  0.302, -3.656, -0.259, -3.242),
    E = c( 1.357, -1.453,  1.477,  0.113, -0.837),
    F = c(-1.006, -0.590,  1.891, -0.397,  0.412),
    G = c(-0.384,  1.652,  1.330,  1.045,  2.064),
    H = c( 0.336, -0.417, -1.673, -1.474, -0.078),
    I = c(-1.239, -0.547,  2.131,  0.393,  0.816),
    K = c( 1.831, -0.561,  0.533, -0.277,  1.648),
    L = c(-1.019, -0.987, -1.505,  1.266, -0.912),
    M = c(-0.663, -1.524,  2.219, -1.005,  1.212),
    N = c( 0.945,  0.828,  1.299, -0.169,  0.933),
    P = c( 0.189,  2.081, -1.628,  0.421, -1.392),
    Q = c( 0.931, -0.179, -3.005, -0.503, -1.853),
    R = c( 1.538, -0.055,  1.502,  0.440,  2.897),
    S = c(-0.228,  1.399, -4.760,  0.670, -2.647),
    T = c(-0.032,  0.326,  2.213,  0.908,  1.313),
    V = c(-1.337, -0.279, -0.544,  1.242, -1.262),
    W = c(-0.595,  0.009,  0.672, -2.128, -0.184),
    Y = c( 0.260,  0.830,  3.097, -0.838,  1.512)
  )
}


#' BLOSUM62 substitution matrix
#' @return 20x20 similarity matrix with AA row/column names
get_blosum62_matrix <- function() {

  aa <- c("A","R","N","D","C","Q","E","G","H","I",
          "L","K","M","F","P","S","T","W","Y","V")

  m <- matrix(c(
    4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0,
    -1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3,
    -2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3,
    -2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3,
    0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,
    -1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2,
    -1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2,
    0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3,
    -2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3,
    -1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3,
    -1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1,
    -1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2,
    -1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1,
    -2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1,
    -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2,
    1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2,
    0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0,
    -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3,
    -2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1,
    0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4
  ), nrow = 20, ncol = 20, byrow = TRUE, dimnames = list(aa, aa))

  m
}

# ===== R-based Encoding (for analysis/visualization) =====

#' Encode sequence using specified method
#'
#' @param sequence Single AA sequence
#' @param max_length Pad/truncate to this length
#' @param method "onehot", "atchley", or "combined"
#' @return Numeric matrix (max_length × features)
encode_sequence <- function(sequence, max_length = 25, method = "combined") {

  chars <- strsplit(toupper(sequence), "")[[1]][1:min(nchar(sequence), max_length)]
  n <- length(chars)

  if (method == "onehot" || method == "combined") {
    onehot <- matrix(0, nrow = max_length, ncol = 20)
    colnames(onehot) <- AA_STANDARD
    for (i in seq_len(n)) {
      if (chars[i] %in% AA_STANDARD) onehot[i, chars[i]] <- 1
    }
  }

  if (method == "atchley" || method == "combined") {
    atchley <- get_atchley_factors()
    atch_mat <- matrix(0, nrow = max_length, ncol = 5,
                       dimnames = list(NULL, c("polarity","secondary","volume","codon","charge")))
    for (i in seq_len(n)) {
      if (chars[i] %in% names(atchley)) atch_mat[i, ] <- atchley[[chars[i]]]
    }
  }

  switch(method,
         onehot = onehot,
         atchley = atch_mat,
         combined = cbind(onehot, atch_mat),
         stop("Unknown method: ", method)
  )
}


#' Encode batch of sequences
#'
#' @param sequences Character vector
#' @param max_length Max sequence length
#' @param method Encoding method
#' @param flatten If TRUE, return matrix with one row per sequence
#' @return Matrix or list of matrices
encode_sequences <- function(sequences, max_length = 25, method = "combined", flatten = TRUE) {

  n_feat <- switch(method, onehot = 20, atchley = 5, combined = 25)

  if (flatten) {
    result <- matrix(0, nrow = length(sequences), ncol = max_length * n_feat)
    for (i in seq_along(sequences)) {
      if (!is.na(sequences[i])) {
        result[i, ] <- as.vector(t(encode_sequence(sequences[i], max_length, method)))
      }
    }
    result
  } else {
    lapply(sequences, function(s) {
      if (is.na(s)) matrix(0, max_length, n_feat)
      else encode_sequence(s, max_length, method)
    })
  }
}

# ===== Python-based Integer Encoding (for PyTorch model) =====

py_run_string("
import numpy as np

# Amino acid vocabulary: 0=PAD, 1-20=AA, 21=UNK
AA_VOCAB = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
AA_VOCAB['<PAD>'] = 0
AA_VOCAB['<UNK>'] = 21

def sequence_to_indices(sequence, max_length=25):
    '''Convert sequence to padded integer indices'''
    indices = [AA_VOCAB.get(aa, 21) for aa in str(sequence)[:max_length].upper()]
    return indices + [0] * (max_length - len(indices))

def batch_to_indices(sequences, max_length=25):
    '''Convert batch of sequences to index array'''
    result = np.array([sequence_to_indices(seq, max_length) for seq in sequences],
                      dtype=np.int64)
    result.flags.writeable = True
    return result
")


#' Convert sequences to integer indices for embedding layers
#'
#' Primary encoding function for model training. Converts amino acid
#' sequences to integer indices (0=PAD, 1-20=AA, 21=UNK).
#'
#' @param sequences Character vector of AA sequences
#' @param max_length Maximum sequence length (default 25)
#' @return Integer matrix (n_sequences × max_length)
sequences_to_indices <- function(sequences, max_length = 25L) {

  result <- py$batch_to_indices(sequences, as.integer(max_length))

  # Return proper R matrix (ensures writability when passed back to Python)
  matrix(as.integer(result), nrow = nrow(result), ncol = ncol(result))
}

# ===== Utility Functions =====

#' Decode integer indices back to sequences
#'
#' @param indices Integer matrix from sequences_to_indices()
#' @return Character vector of sequences (PAD tokens removed)
indices_to_sequences <- function(indices) {

  aa_lookup <- c("<PAD>", strsplit("ACDEFGHIKLMNPQRSTVWY", "")[[1]], "<UNK>")

  apply(indices, 1, function(row) {
    chars <- aa_lookup[row + 1]  # +1 for R indexing
    paste(chars[chars != "<PAD>"], collapse = "")
  })
}


#' Get vocabulary info
#' @return List with vocabulary details
get_aa_vocab <- function() {
  list(
    standard = AA_STANDARD,
    size = 22,  # 20 AA + PAD + UNK
    pad_idx = 0L,
    unk_idx = 21L,
    aa_to_idx = setNames(1:20, AA_STANDARD)
  )
}

# ===== Example Usage =====
# # For model training (primary usage)
# cdr3_indices <- sequences_to_indices(c("CASSLAPGATNEKLFF", "CASSLGQAYEQYF"), 25)
#
# # For analysis/visualization
# encoded <- encode_sequence("CASSLAPGATNEKLFF", method = "atchley")
#
# # Decode back to sequences
# sequences <- indices_to_sequences(cdr3_indices)
