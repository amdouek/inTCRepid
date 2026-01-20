# ESM-2 Protein Language Model Embeddings (V10)
#
# Provides ESM-2 embedding extraction and HDF5 caching for TCR-epitope model.
#
# Key functions:
#   - load_esm2_model(): Load ESM-2 model and tokenizer
#   - extract_esm_embeddings(): Batch embedding extraction
#   - cache_embeddings_hdf5(): Save embeddings to HDF5
#   - load_cached_embeddings(): Load from HDF5 cache
#   - prepare_training_embeddings(): Full pipeline for training data
#

library(reticulate)
library(dplyr)
library(stringr)

# ===== Python Infrastructure =====

py_run_string("
import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

# Global model cache to avoid reloading
_ESM_MODEL_CACHE = {}

def load_esm2(model_name: str = 'facebook/esm2_t30_150M_UR50D',
              device: str = 'cuda') -> Tuple:
    '''
    Load ESM-2 model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model identifier
        device: 'cuda' or 'cpu'

    Returns:
        (model, tokenizer, embed_dim)
    '''
    global _ESM_MODEL_CACHE

    cache_key = f'{model_name}_{device}'
    if cache_key in _ESM_MODEL_CACHE:
        print(f'Using cached ESM-2 model')
        return _ESM_MODEL_CACHE[cache_key]

    print(f'Loading ESM-2: {model_name}')
    print(f'Device: {device}')

    from transformers import EsmModel, EsmTokenizer

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Get embedding dimension
    embed_dim = model.config.hidden_size

    print(f'ESM-2 loaded: {sum(p.numel() for p in model.parameters()):,} params, {embed_dim}-dim embeddings')

    result = (model, tokenizer, embed_dim)
    _ESM_MODEL_CACHE[cache_key] = result

    return result


def extract_embeddings_batch(model, tokenizer, sequences: List[str],
                            max_length: int = 30,
                            device: str = 'cuda',
                            pooling: str = 'mean') -> np.ndarray:
    '''
    Extract ESM-2 embeddings for a batch of sequences.

    Args:
        model: ESM-2 model
        tokenizer: ESM tokenizer
        sequences: List of amino acid sequences
        max_length: Maximum sequence length (truncate longer)
        device: Compute device
        pooling: 'mean' (average over residues) or 'per_residue' (padded 3D tensor)

    Returns:
        If pooling='mean': (batch_size, embed_dim) array
        If pooling='per_residue': (batch_size, max_length, embed_dim) array
    '''
    # Clean sequences - remove any non-standard characters
    clean_seqs = []
    for seq in sequences:
        if seq is None or str(seq).upper() in ['NA', 'NAN', 'NONE', '']:
            clean_seqs.append('A' * 10)  # Placeholder for missing
        else:
            # Keep only standard amino acids
            cleaned = ''.join(c for c in str(seq).upper() if c in 'ACDEFGHIKLMNPQRSTVWY')
            if len(cleaned) == 0:
                cleaned = 'A' * 10
            clean_seqs.append(cleaned[:max_length])

    # Tokenize
    encoded = tokenizer(
        clean_seqs,
        padding=True,
        truncation=True,
        max_length=max_length + 2,  # +2 for special tokens
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Extract embeddings
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, embed_dim)

    if pooling == 'mean':
        # Mean pooling over non-padding tokens (excluding BOS/EOS)
        # Positions: 0=BOS, 1:L+1=sequence, L+1=EOS
        mask = attention_mask[:, 1:-1].unsqueeze(-1).float()  # Exclude BOS/EOS
        seq_hidden = hidden_states[:, 1:-1, :]  # Exclude BOS/EOS

        # Handle edge case where mask sums to 0
        mask_sum = mask.sum(dim=1).clamp(min=1)
        pooled = (seq_hidden * mask).sum(dim=1) / mask_sum

        return pooled.cpu().numpy()

    elif pooling == 'per_residue':
        # Return padded per-residue embeddings
        embed_dim = hidden_states.size(-1)
        batch_size = len(sequences)

        # Initialize output with zeros (padding)
        output = np.zeros((batch_size, max_length, embed_dim), dtype=np.float32)

        # Copy sequence embeddings (excluding BOS at position 0)
        seq_hidden = hidden_states[:, 1:, :].cpu().numpy()  # Skip BOS

        for i in range(batch_size):
            seq_len = min(len(clean_seqs[i]), max_length)
            output[i, :seq_len, :] = seq_hidden[i, :seq_len, :]

        return output

    else:
        raise ValueError(f'Unknown pooling: {pooling}')


def extract_all_embeddings(model, tokenizer, sequences: List[str],
                          batch_size: int = 32,
                          max_length: int = 30,
                          device: str = 'cuda',
                          pooling: str = 'mean',
                          show_progress: bool = True) -> np.ndarray:
    '''
    Extract embeddings for all sequences with batching.

    Args:
        model: ESM-2 model
        tokenizer: ESM tokenizer
        sequences: List of sequences
        batch_size: Batch size for processing
        max_length: Max sequence length
        device: Compute device
        pooling: Pooling strategy
        show_progress: Print progress

    Returns:
        Embeddings array
    '''
    n_sequences = len(sequences)
    n_batches = (n_sequences + batch_size - 1) // batch_size

    all_embeddings = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_sequences)
        batch_seqs = sequences[start:end]

        embeddings = extract_embeddings_batch(
            model, tokenizer, batch_seqs,
            max_length, device, pooling
        )
        all_embeddings.append(embeddings)

        if show_progress and (i + 1) % 50 == 0:
            print(f'  Processed {end}/{n_sequences} sequences ({100*end/n_sequences:.1f}%)')

    result = np.concatenate(all_embeddings, axis=0)

    if show_progress:
        print(f'  Complete: {result.shape}')

    # Clear GPU cache
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    return result


# ===== HDF5 Caching =====

def save_embeddings_hdf5(embeddings_dict: Dict[str, Dict],
                         cache_path: str,
                         compression: str = 'gzip') -> None:
    '''
    Save embeddings to HDF5 file.

    Args:
        embeddings_dict: Dict with structure:
            {
                'cdr3_alpha': {'sequences': [...], 'embeddings': array, 'seq_to_idx': {...}},
                'cdr3_beta': {...},
                'epitope': {...}
            }
        cache_path: Output file path
        compression: Compression algorithm
    '''
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(cache_path, 'w') as f:
        f.attrs['version'] = '1.0'
        f.attrs['model'] = 'esm2_t30_150M'

        for key, data in embeddings_dict.items():
            grp = f.create_group(key)

            # Save sequences as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            grp.create_dataset('sequences', data=data['sequences'], dtype=dt)

            # Save embeddings with compression
            grp.create_dataset('embeddings', data=data['embeddings'],
                             compression=compression, compression_opts=4)

            # Save sequence-to-index mapping as JSON string
            import json
            grp.attrs['seq_to_idx'] = json.dumps(data['seq_to_idx'])

            print(f'  Saved {key}: {len(data[\"sequences\"])} sequences, embeddings {data[\"embeddings\"].shape}')

    print(f'Cache saved: {cache_path}')


def load_embeddings_hdf5(cache_path: str) -> Dict[str, Dict]:
    '''
    Load embeddings from HDF5 cache.

    Args:
        cache_path: Path to HDF5 file

    Returns:
        Dict with same structure as save_embeddings_hdf5 input
    '''
    import json

    result = {}

    with h5py.File(cache_path, 'r') as f:
        print(f'Loading cache: {cache_path}')
        print(f'  Version: {f.attrs.get(\"version\", \"unknown\")}')
        print(f'  Model: {f.attrs.get(\"model\", \"unknown\")}')

        for key in f.keys():
            grp = f[key]

            sequences = [s.decode() if isinstance(s, bytes) else s
                        for s in grp['sequences'][:]]
            embeddings = grp['embeddings'][:]
            seq_to_idx = json.loads(grp.attrs['seq_to_idx'])

            result[key] = {
                'sequences': sequences,
                'embeddings': embeddings,
                'seq_to_idx': seq_to_idx
            }

            print(f'  Loaded {key}: {len(sequences)} sequences, embeddings {embeddings.shape}')

    return result


def get_embedding_indices(sequences: List[str], seq_to_idx: Dict[str, int],
                         placeholder_idx: int = 0) -> np.ndarray:
    '''
    Map sequences to their embedding indices.

    Args:
        sequences: List of sequences to look up
        seq_to_idx: Mapping from sequence to embedding index
        placeholder_idx: Index to use for missing sequences

    Returns:
        Array of indices
    '''
    indices = []
    n_missing = 0

    for seq in sequences:
        if seq is None or str(seq).upper() in ['NA', 'NAN', 'NONE', '']:
            indices.append(placeholder_idx)
            n_missing += 1
        elif seq in seq_to_idx:
            indices.append(seq_to_idx[seq])
        else:
            # Try uppercase
            seq_upper = str(seq).upper()
            if seq_upper in seq_to_idx:
                indices.append(seq_to_idx[seq_upper])
            else:
                indices.append(placeholder_idx)
                n_missing += 1

    if n_missing > 0:
        print(f'  Warning: {n_missing}/{len(sequences)} sequences not in cache, using placeholder')

    return np.array(indices, dtype=np.int64)
")

# ===== R Interface Functions =====

#' Load ESM-2 model
#'
#' @param model_name HuggingFace model identifier (default: esm2_t30_150M)
#' @param device 'cuda' or 'cpu'
#' @return List with model, tokenizer, embed_dim
#' @export
load_esm2_model <- function(model_name = "facebook/esm2_t30_150M_UR50D",
                            device = "cuda") {

  if (device == "cuda" && !py$torch$cuda$is_available()) {
    message("CUDA not available, using CPU")
    device <- "cpu"
  }

  result <- py$load_esm2(model_name, device)

  list(
    model = result[[1]],
    tokenizer = result[[2]],
    embed_dim = as.integer(result[[3]]),
    model_name = model_name,
    device = device
  )
}

#' Extract ESM-2 embeddings for sequences
#'
#' @param esm ESM-2 model list from load_esm2_model()
#' @param sequences Character vector of amino acid sequences
#' @param batch_size Batch size for processing
#' @param max_length Maximum sequence length
#' @param pooling 'mean' or 'per_residue'
#' @return Numeric matrix of embeddings
#' @export
extract_esm_embeddings <- function(esm, sequences, batch_size = 32L,
                                   max_length = 30L, pooling = "mean") {

  message(sprintf("Extracting ESM-2 embeddings for %d sequences...", length(sequences)))

  embeddings <- py$extract_all_embeddings(
    model = esm$model,
    tokenizer = esm$tokenizer,
    sequences = as.list(sequences),
    batch_size = as.integer(batch_size),
    max_length = as.integer(max_length),
    device = esm$device,
    pooling = pooling,
    show_progress = TRUE
  )

  embeddings
}

#' Prepare and cache training embeddings
#'
#' Extracts ESM-2 embeddings for all unique CDR3α, CDR3β, and epitope
#' sequences in the training data and saves to HDF5 cache.
#'
#' @param data_splits Training data splits from prepare_combined_data()
#' @param esm ESM-2 model list (will load if NULL)
#' @param cache_path Output HDF5 file path
#' @param batch_size Batch size for embedding extraction
#' @param max_length Maximum sequence length
#' @param pooling Pooling strategy ('mean' recommended for V10)
#' @param device Compute device
#' @return List with embedding cache data
#' @export
prepare_training_embeddings <- function(data_splits,
                                        esm = NULL,
                                        cache_path = "data/esm_embeddings_cache.h5",
                                        batch_size = 32L,
                                        max_length = 30L,
                                        pooling = "mean",
                                        device = "cuda") {

  message("\n", strrep("=", 60))
  message("PREPARING ESM-2 TRAINING EMBEDDINGS")
  message(strrep("=", 60))

  # Load ESM-2 if not provided
  if (is.null(esm)) {
    esm <- load_esm2_model(device = device)
  }

  # Collect all data
  all_data <- bind_rows(
    data_splits$train$data,
    data_splits$validation$data,
    data_splits$test$data
  )

  message(sprintf("\nTotal samples: %s", format(nrow(all_data), big.mark = ",")))

  embeddings_dict <- list()

  # --- CDR3 Alpha ---
  message("\n--- CDR3 Alpha ---")
  cdr3_alpha_seqs <- all_data$cdr3_alpha %>%
    na.omit() %>%
    unique() %>%
    as.character()

  # Add placeholder for missing
  cdr3_alpha_seqs <- c("AAAAAAAAAA", cdr3_alpha_seqs)  # Index 0 = placeholder

  message(sprintf("Unique CDR3α sequences: %d", length(cdr3_alpha_seqs)))

  cdr3_alpha_emb <- extract_esm_embeddings(
    esm, cdr3_alpha_seqs, batch_size, max_length, pooling
  )

  embeddings_dict$cdr3_alpha <- list(
    sequences = cdr3_alpha_seqs,
    embeddings = cdr3_alpha_emb,
    seq_to_idx = setNames(as.list(seq_along(cdr3_alpha_seqs) - 1L), cdr3_alpha_seqs)
  )

  # --- CDR3 Beta ---
  message("\n--- CDR3 Beta ---")
  cdr3_beta_seqs <- all_data$cdr3_beta %>%
    na.omit() %>%
    unique() %>%
    as.character()

  cdr3_beta_seqs <- c("AAAAAAAAAA", cdr3_beta_seqs)  # Index 0 = placeholder

  message(sprintf("Unique CDR3β sequences: %d", length(cdr3_beta_seqs)))

  cdr3_beta_emb <- extract_esm_embeddings(
    esm, cdr3_beta_seqs, batch_size, max_length, pooling
  )

  embeddings_dict$cdr3_beta <- list(
    sequences = cdr3_beta_seqs,
    embeddings = cdr3_beta_emb,
    seq_to_idx = setNames(as.list(seq_along(cdr3_beta_seqs) - 1L), cdr3_beta_seqs)
  )

  # --- Epitope ---
  message("\n--- Epitope ---")
  epitope_seqs <- all_data$epitope %>%
    na.omit() %>%
    unique() %>%
    as.character()

  epitope_seqs <- c("AAAAAAAAAA", epitope_seqs)  # Index 0 = placeholder

  message(sprintf("Unique epitope sequences: %d", length(epitope_seqs)))

  epitope_emb <- extract_esm_embeddings(
    esm, epitope_seqs, batch_size, max_length, pooling
  )

  embeddings_dict$epitope <- list(
    sequences = epitope_seqs,
    embeddings = epitope_emb,
    seq_to_idx = setNames(as.list(seq_along(epitope_seqs) - 1L), epitope_seqs)
  )

  # Save to HDF5
  message("\n--- Saving to HDF5 ---")
  py$save_embeddings_hdf5(
    r_to_py(embeddings_dict),
    cache_path
  )

  # Summary
  message("\n", strrep("-", 40))
  message("EMBEDDING CACHE SUMMARY")
  message(strrep("-", 40))
  message(sprintf("CDR3α: %d sequences, %d-dim embeddings",
                  length(cdr3_alpha_seqs), ncol(cdr3_alpha_emb)))
  message(sprintf("CDR3β: %d sequences, %d-dim embeddings",
                  length(cdr3_beta_seqs), ncol(cdr3_beta_emb)))
  message(sprintf("Epitope: %d sequences, %d-dim embeddings",
                  length(epitope_seqs), ncol(epitope_emb)))
  message(sprintf("Cache file: %s", cache_path))
  message(sprintf("File size: %.1f MB", file.size(cache_path) / 1024^2))

  invisible(list(
    cache_path = cache_path,
    cdr3_alpha = embeddings_dict$cdr3_alpha,
    cdr3_beta = embeddings_dict$cdr3_beta,
    epitope = embeddings_dict$epitope,
    embed_dim = esm$embed_dim,
    model_name = esm$model_name
  ))
}

#' Load cached embeddings from HDF5
#'
#' @param cache_path Path to HDF5 cache file
#' @return List with embedding data
#' @export
load_embedding_cache <- function(cache_path) {

  if (!file.exists(cache_path)) {
    stop("Cache file not found: ", cache_path)
  }

  py_cache <- py$load_embeddings_hdf5(cache_path)

  # Convert to R-friendly format
  list(
    cdr3_alpha = list(
      sequences = py_cache$cdr3_alpha$sequences,
      embeddings = py_cache$cdr3_alpha$embeddings,
      seq_to_idx = py_cache$cdr3_alpha$seq_to_idx
    ),
    cdr3_beta = list(
      sequences = py_cache$cdr3_beta$sequences,
      embeddings = py_cache$cdr3_beta$embeddings,
      seq_to_idx = py_cache$cdr3_beta$seq_to_idx
    ),
    epitope = list(
      sequences = py_cache$epitope$sequences,
      embeddings = py_cache$epitope$embeddings,
      seq_to_idx = py_cache$epitope$seq_to_idx
    ),
    cache_path = cache_path
  )
}

#' Add embedding indices to data splits
#'
#' Maps sequences to their embedding cache indices.
#'
#' @param data_splits Data splits from prepare_combined_data()
#' @param emb_cache Embedding cache from load_embedding_cache()
#' @return Modified data_splits with embedding indices
#' @export
add_embedding_indices <- function(data_splits, emb_cache) {

  message("Adding ESM-2 embedding indices to data splits...")

  add_indices_to_split <- function(split, cache) {
    n <- nrow(split$data)

    # CDR3 Alpha indices
    cdr3_alpha_idx <- py$get_embedding_indices(
      as.list(split$data$cdr3_alpha),
      r_to_py(cache$cdr3_alpha$seq_to_idx),
      placeholder_idx = 0L
    )

    # CDR3 Beta indices
    cdr3_beta_idx <- py$get_embedding_indices(
      as.list(split$data$cdr3_beta),
      r_to_py(cache$cdr3_beta$seq_to_idx),
      placeholder_idx = 0L
    )

    # Epitope indices
    epitope_emb_idx <- py$get_embedding_indices(
      as.list(split$data$epitope),
      r_to_py(cache$epitope$seq_to_idx),
      placeholder_idx = 0L
    )

    split$esm_cdr3_alpha_idx <- as.integer(cdr3_alpha_idx)
    split$esm_cdr3_beta_idx <- as.integer(cdr3_beta_idx)
    split$esm_epitope_idx <- as.integer(epitope_emb_idx)

    split
  }

  data_splits$train <- add_indices_to_split(data_splits$train, emb_cache)
  data_splits$validation <- add_indices_to_split(data_splits$validation, emb_cache)
  data_splits$test <- add_indices_to_split(data_splits$test, emb_cache)

  # Also add for unique epitopes
  data_splits$unique_epitope_esm_idx <- as.integer(py$get_embedding_indices(
    as.list(data_splits$unique_epitopes),
    r_to_py(emb_cache$epitope$seq_to_idx),
    placeholder_idx = 0L
  ))

  # Store embedding cache reference
  data_splits$emb_cache <- emb_cache

  message("  Train: ", length(data_splits$train$esm_cdr3_alpha_idx), " samples")
  message("  Validation: ", length(data_splits$validation$esm_cdr3_alpha_idx), " samples")
  message("  Test: ", length(data_splits$test$esm_cdr3_alpha_idx), " samples")

  data_splits
}

#' Verify ESM-2 setup
#'
#' Quick test to verify ESM-2 is working.
#'
#' @param device 'cuda' or 'cpu'
#' @return TRUE if successful
#' @export
verify_esm2_setup <- function(device = "cuda") {

  message("Verifying ESM-2 setup...")

  tryCatch({
    esm <- load_esm2_model(device = device)

    test_seqs <- c("CASSLAPGATNEKLFF", "GILGFVFTL", "CASSLGQAYEQYF")

    embeddings <- extract_esm_embeddings(
      esm, test_seqs,
      batch_size = 3L,
      max_length = 25L,
      pooling = "mean"
    )

    message(sprintf("✓ ESM-2 working: %d sequences → (%d, %d) embeddings",
                    length(test_seqs), nrow(embeddings), ncol(embeddings)))
    message(sprintf("  Model: %s", esm$model_name))
    message(sprintf("  Device: %s", esm$device))
    message(sprintf("  Embedding dim: %d", esm$embed_dim))

    TRUE

  }, error = function(e) {
    message("✗ ESM-2 verification failed: ", e$message)
    FALSE
  })
}
