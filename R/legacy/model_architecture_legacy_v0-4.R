# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    LEGACY MODEL ARCHITECTURE (V0-V4)                       ║
# ║                                                                            ║
# ║  Status: ARCHIVED - Do not use for new model development                   ║
# ║  Maintained for: Reproducibility of V0-V4 trained models                   ║
# ║  Superseded by: model_architecture.R (V5+)                                 ║
# ║                                                                            ║
# ║  To load a V0-V4 model:                                                    ║
# ║    source("scripts/model_architecture_legacy.R")                           ║
# ║    model <- load_model_package("models/tcr_epitope_v4/", legacy = TRUE)    ║
# ║                                                                            ║
# ║  Architecture parameters (V4):                                             ║
# ║    - token_embedding_dim: 64                                               ║
# ║    - hidden_dim: 128                                                       ║
# ║    - output_dim: 128                                                       ║
# ║    - Total parameters: ~284K                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ======= TCR-Epitope Model Architecture (PyTorch via reticulate) =============

# Define the model architecture in Python
py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CDR3Encoder(nn.Module):
    '''
    Encoder for CDR3 sequences using CNN + attention

    Architecture: Multi-scale parallel convolutions followed by attention pooling
    - Three parallel conv layers with different kernel sizes (3, 5, 7)
    - Each captures motifs at different scales
    - Attention-weighted pooling combines positions
    - Concatenated multi-scale features projected to output dim
    '''
    def __init__(self,
                 vocab_size=22,
                 embedding_dim=64,
                 hidden_dim=128,
                 output_dim=128,
                 max_length=25,
                 dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # CORRECTED: All conv layers take embedding_dim as input (parallel architecture)
        # Each conv layer captures different scale motifs from the same input
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=5, padding=2)  # Fixed: was hidden_dim
        self.conv3 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=7, padding=3)  # Fixed: was hidden_dim

        # Batch normalization for each conv output
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Attention mechanism for position-wise weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Output projection: 3 conv outputs concatenated -> output_dim
        self.fc = nn.Linear(hidden_dim * 3, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, seq_len) - integer indices

        # Create mask for padding tokens (index 0)
        if mask is None:
            mask = (x != 0).float()  # (batch, seq_len)

        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # Transpose for conv1d: (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # Parallel multi-scale convolutions (all from same input)
        c1 = F.relu(self.bn1(self.conv1(embedded)))  # (batch, hidden_dim, seq_len)
        c2 = F.relu(self.bn2(self.conv2(embedded)))  # (batch, hidden_dim, seq_len)
        c3 = F.relu(self.bn3(self.conv3(embedded)))  # (batch, hidden_dim, seq_len)

        # Transpose back for attention: (batch, hidden_dim, seq_len) -> (batch, seq_len, hidden_dim)
        c1 = c1.transpose(1, 2)
        c2 = c2.transpose(1, 2)
        c3 = c3.transpose(1, 2)

        # Attention-weighted pooling for each conv output
        def attention_pool(features, mask):
            # features: (batch, seq_len, hidden_dim)
            # mask: (batch, seq_len)
            attn_scores = self.attention(features).squeeze(-1)  # (batch, seq_len)

            # Mask padding positions
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len)

            # Weighted sum: (batch, 1, seq_len) @ (batch, seq_len, hidden_dim) -> (batch, 1, hidden_dim)
            pooled = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)  # (batch, hidden_dim)
            return pooled

        # Pool each scale
        p1 = attention_pool(c1, mask)  # (batch, hidden_dim)
        p2 = attention_pool(c2, mask)  # (batch, hidden_dim)
        p3 = attention_pool(c3, mask)  # (batch, hidden_dim)

        # Concatenate multi-scale features
        combined = torch.cat([p1, p2, p3], dim=-1)  # (batch, hidden_dim * 3)
        combined = self.dropout(combined)

        # Project to output dimension
        output = self.fc(combined)  # (batch, output_dim)

        return output


class EpitopeEncoder(nn.Module):
    '''
    Encoder for epitope sequences

    Similar architecture to CDR3 encoder but with different pooling
    since epitopes have different length characteristics.
    '''
    def __init__(self,
                 vocab_size=22,
                 embedding_dim=64,
                 hidden_dim=128,
                 output_dim=128,
                 max_length=30,
                 dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # CORRECTED: Both conv layers take embedding_dim as input
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=5, padding=2)  # Fixed: was hidden_dim

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Global pooling + projection
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len) - integer indices

        # Create mask for padding
        mask = (x != 0).float()  # (batch, seq_len)

        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)

        # Parallel convolutions
        c1 = F.relu(self.bn1(self.conv1(embedded)))  # (batch, hidden_dim, seq_len)
        c2 = F.relu(self.bn2(self.conv2(embedded)))  # (batch, hidden_dim, seq_len)

        # Masked global pooling (max and average, ignoring padding)
        # Expand mask for broadcasting: (batch, seq_len) -> (batch, 1, seq_len)
        mask_expanded = mask.unsqueeze(1)

        # Masked max pooling
        c1_masked = c1.masked_fill(mask_expanded == 0, -1e9)
        c2_masked = c2.masked_fill(mask_expanded == 0, -1e9)
        p1_max = c1_masked.max(dim=-1)[0]  # (batch, hidden_dim)
        p2_max = c2_masked.max(dim=-1)[0]  # (batch, hidden_dim)

        # Masked average pooling
        c1_masked_avg = c1 * mask_expanded
        c2_masked_avg = c2 * mask_expanded
        lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (batch, 1)
        p1_avg = c1_masked_avg.sum(dim=-1) / lengths  # (batch, hidden_dim)
        p2_avg = c2_masked_avg.sum(dim=-1) / lengths  # (batch, hidden_dim)

        # Combine max and average pooling
        p1 = p1_max + p1_avg  # (batch, hidden_dim)
        p2 = p2_max + p2_avg  # (batch, hidden_dim)

        # Concatenate and project
        combined = torch.cat([p1, p2], dim=-1)  # (batch, hidden_dim * 2)
        combined = self.dropout(combined)

        return self.fc(combined)  # (batch, output_dim)


class TCREpitopeModel(nn.Module):
    '''
    Full model for TCR-epitope binding prediction

    Architecture:
    - Separate encoders for CDR3 and epitope sequences
    - Both produce normalized embeddings in shared space
    - Binding predicted via cosine similarity
    - Learnable temperature parameter for scaling

    Training: Contrastive learning - matching CDR3-epitope pairs should
    have higher similarity than non-matching pairs.

    Inference: Compare query CDR3 embedding to all epitope embeddings,
    return most similar epitopes as predictions.
    '''
    def __init__(self,
                 embedding_dim=128,
                 cdr3_max_length=25,
                 epitope_max_length=30,
                 dropout=0.3):
        super().__init__()

        self.cdr3_encoder = CDR3Encoder(
            output_dim=embedding_dim,
            max_length=cdr3_max_length,
            dropout=dropout
        )

        self.epitope_encoder = EpitopeEncoder(
            output_dim=embedding_dim,
            max_length=epitope_max_length,
            dropout=dropout
        )

        # Learnable temperature for scaling similarities
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def encode_cdr3(self, cdr3_indices):
        '''Encode CDR3 sequences to normalized embeddings'''
        embeddings = self.cdr3_encoder(cdr3_indices)
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_epitope(self, epitope_indices):
        '''Encode epitope sequences to normalized embeddings'''
        embeddings = self.epitope_encoder(epitope_indices)
        return F.normalize(embeddings, p=2, dim=-1)

    def forward(self, cdr3_indices, epitope_indices):
        '''
        Forward pass computing similarity matrix

        Args:
            cdr3_indices: (batch, cdr3_seq_len) integer indices
            epitope_indices: (n_epitopes, epitope_seq_len) integer indices

        Returns:
            similarity: (batch, n_epitopes) similarity scores
            cdr3_emb: (batch, embedding_dim) CDR3 embeddings
            epitope_emb: (n_epitopes, embedding_dim) epitope embeddings
        '''
        cdr3_emb = self.encode_cdr3(cdr3_indices)
        epitope_emb = self.encode_epitope(epitope_indices)

        # Cosine similarity (embeddings already normalized) scaled by temperature
        similarity = torch.mm(cdr3_emb, epitope_emb.t()) / self.temperature.clamp(min=0.01)

        return similarity, cdr3_emb, epitope_emb


def create_model(embedding_dim=128, cdr3_max_len=25, epitope_max_len=30, dropout=0.3):
    '''Factory function to create model instance'''
    model = TCREpitopeModel(
        embedding_dim=embedding_dim,
        cdr3_max_length=cdr3_max_len,
        epitope_max_length=epitope_max_len,
        dropout=dropout
    )
    return model
")

# ======= R Wrapper Functions for Model =======================================

#' Create TCR-Epitope model
#'
#' @param embedding_dim Dimension of learned embeddings
#' @param cdr3_max_len Maximum CDR3 sequence length
#' @param epitope_max_len Maximum epitope sequence length
#' @param dropout Dropout rate
#' @return PyTorch model object
create_tcr_epitope_model <- function(embedding_dim = 128,
                                     cdr3_max_len = 25,
                                     epitope_max_len = 30,
                                     dropout = 0.3) {

  model <- py$create_model(
    embedding_dim = as.integer(embedding_dim),
    cdr3_max_len = as.integer(cdr3_max_len),
    epitope_max_len = as.integer(epitope_max_len),
    dropout = dropout
  )

  return(model)
}


#' Prepare data for model training
#'
#' @param processed_data Preprocessed VDJdb tibble
#' @param test_fraction Fraction of data for testing
#' @param validation_fraction Fraction of training data for validation
#' @param stratify_by Column to stratify by (e.g., "epitope")
#' @param seed Random seed
#' @return List with train/validation/test splits
prepare_model_data <- function(processed_data,
                               test_fraction = 0.15,
                               validation_fraction = 0.15,
                               stratify_by = "epitope",
                               seed = 42) {

  set.seed(seed)

  # Create epitope-to-index mapping (for classification formulation)
  unique_epitopes <- unique(processed_data$epitope)
  epitope_to_idx <- setNames(seq_along(unique_epitopes) - 1, unique_epitopes)

  processed_data <- processed_data %>%
    mutate(epitope_idx = epitope_to_idx[epitope])

  # Stratified split by epitope to ensure all epitopes in train set
  # First, identify epitopes with enough samples
  epitope_counts <- processed_data %>%
    count(epitope) %>%
    arrange(desc(n))

  # For epitopes with few samples, keep all in training
  min_samples_for_split <- 3

  splittable_epitopes <- epitope_counts %>%
    filter(n >= min_samples_for_split) %>%
    pull(epitope)

  # Split data
  data_splittable <- processed_data %>%
    filter(epitope %in% splittable_epitopes)

  data_unsplittable <- processed_data %>%
    filter(!epitope %in% splittable_epitopes)

  # Stratified test split
  if (nrow(data_splittable) > 0) {
    test_indices <- createDataPartition(
      data_splittable$epitope,
      p = test_fraction,
      list = FALSE
    )

    test_data <- data_splittable[test_indices, ]
    train_val_data <- data_splittable[-test_indices, ]

    # Validation split
    val_indices <- createDataPartition(
      train_val_data$epitope,
      p = validation_fraction / (1 - test_fraction),
      list = FALSE
    )

    val_data <- train_val_data[val_indices, ]
    train_data <- train_val_data[-val_indices, ]

    # Add unsplittable data to training
    train_data <- bind_rows(train_data, data_unsplittable)

  } else {
    # Not enough data for stratified split
    n_total <- nrow(processed_data)
    indices <- sample(n_total)

    n_test <- floor(n_total * test_fraction)
    n_val <- floor(n_total * validation_fraction)

    test_data <- processed_data[indices[1:n_test], ]
    val_data <- processed_data[indices[(n_test+1):(n_test+n_val)], ]
    train_data <- processed_data[indices[(n_test+n_val+1):n_total], ]
  }

  message(paste("Training samples:", nrow(train_data)))
  message(paste("Validation samples:", nrow(val_data)))
  message(paste("Test samples:", nrow(test_data)))
  message(paste("Unique epitopes in training:", n_distinct(train_data$epitope)))

  # Convert sequences to indices
  cdr3_max_len <- 25L
  epitope_max_len <- 30L

  train_cdr3_idx <- sequences_to_indices(train_data$cdr3, cdr3_max_len)
  train_epitope_idx <- sequences_to_indices(train_data$epitope, epitope_max_len)
  train_weights <- as.numeric(train_data$sample_weight)
  train_labels <- as.integer(train_data$epitope_idx)

  val_cdr3_idx <- sequences_to_indices(val_data$cdr3, cdr3_max_len)
  val_epitope_idx <- sequences_to_indices(val_data$epitope, epitope_max_len)
  val_weights <- as.numeric(val_data$sample_weight)
  val_labels <- as.integer(val_data$epitope_idx)

  test_cdr3_idx <- sequences_to_indices(test_data$cdr3, cdr3_max_len)
  test_epitope_idx <- sequences_to_indices(test_data$epitope, epitope_max_len)
  test_weights <- as.numeric(test_data$sample_weight)
  test_labels <- as.integer(test_data$epitope_idx)

  return(list(
    train = list(
      data = train_data,
      cdr3_idx = train_cdr3_idx,
      epitope_idx = train_epitope_idx,
      weights = train_weights,
      labels = train_labels
    ),
    validation = list(
      data = val_data,
      cdr3_idx = val_cdr3_idx,
      epitope_idx = val_epitope_idx,
      weights = val_weights,
      labels = val_labels
    ),
    test = list(
      data = test_data,
      cdr3_idx = test_cdr3_idx,
      epitope_idx = test_epitope_idx,
      weights = test_weights,
      labels = test_labels
    ),
    epitope_to_idx = epitope_to_idx,
    idx_to_epitope = setNames(names(epitope_to_idx), epitope_to_idx),
    unique_epitopes = unique_epitopes
  ))
}
