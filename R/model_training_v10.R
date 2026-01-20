# TCR-Epitope Model Training Infrastructure V10: ESM-2 Integration
#
# V10 trainer handles ESM embedding indices instead of raw sequences.
#
# Batch structure (10 tensors):
#   (esm_cdr3_alpha_idx, esm_cdr3_beta_idx, labels, weights,
#    v_alpha_idx, j_alpha_idx, v_beta_idx, j_beta_idx,
#    mhc_class_idx, mhc_allele_idx)
#
# Key differences from V9.1:
#   - No sequence token indices (CDR3/epitope) - replaced by ESM lookup indices
#   - Unique epitope ESM indices passed separately to trainer
#   - Model looks up ESM embeddings from internal frozen buffers
#

library(reticulate)
library(dplyr)
library(tidyr)

# ===== Python Training Infrastructure =====

py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Subset
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, Tuple, List

# --- Focal Loss (same as V9.1) ---
class FocalLossV10(nn.Module):
    '''Focal Loss for class imbalance.'''

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0,
                 reduction: str = 'none'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction='none',
                            label_smoothing=self.label_smoothing)
        p_t = F.softmax(inputs, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = ((1 - p_t) ** self.gamma) * ce

        if self.reduction == 'mean': return focal.mean()
        elif self.reduction == 'sum': return focal.sum()
        return focal


# --- TCR Trainer V10 ---
class TCRTrainerV10:
    '''
    Trainer for TCR-Epitope model V10 with ESM-2 embeddings.

    Batch structure (10 tensors):
        (esm_cdr3_alpha_idx, esm_cdr3_beta_idx, labels, weights,
         v_alpha_idx, j_alpha_idx, v_beta_idx, j_beta_idx,
         mhc_class_idx, mhc_allele_idx)

    The model uses indices to look up ESM embeddings from its internal buffers.
    '''

    def __init__(self, model: nn.Module, device: str = 'cpu',
                 loss_type: str = 'focal', focal_gamma: float = 2.0,
                 label_smoothing: float = 0.0,
                 mhc_class_unk_idx: int = 3, mhc_allele_unk_idx: int = 1):

        self.model = model.to(device)
        self.device = device
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.mhc_class_unk_idx = mhc_class_unk_idx
        self.mhc_allele_unk_idx = mhc_allele_unk_idx

        # Loss function
        if loss_type == 'focal':
            self.loss_fn = FocalLossV10(focal_gamma, label_smoothing, 'none')
        else:
            self.loss_fn = None

        self.history = defaultdict(list)

        print(f'TCRTrainerV10: {loss_type} loss, gamma={focal_gamma}, device={device}')
        print(f'  ESM initialized: {model._esm_initialized}')

    def _to_tensor(self, arr, dtype=torch.long):
        '''Convert array to tensor, ensuring contiguous memory.'''
        return torch.tensor(np.array(arr, copy=True), dtype=dtype)

    def create_dataloader(self, esm_cdr3_alpha_idx, esm_cdr3_beta_idx,
                          labels, weights,
                          v_alpha_idx, j_alpha_idx, v_beta_idx, j_beta_idx,
                          mhc_class_idx, mhc_allele_idx,
                          batch_size=32, shuffle=True, weighted_sampling=True):
        '''
        Create dataloader with 10-tensor batch structure.

        Args:
            esm_cdr3_alpha_idx: Indices into model's ESM CDR3α buffer
            esm_cdr3_beta_idx: Indices into model's ESM CDR3β buffer
            labels: Epitope class labels
            weights: Sample weights
            v_alpha_idx, j_alpha_idx: TRA V/J gene indices
            v_beta_idx, j_beta_idx: TRB V/J gene indices
            mhc_class_idx, mhc_allele_idx: MHC indices
            batch_size: Batch size
            shuffle: Shuffle data
            weighted_sampling: Use weighted random sampling

        Returns:
            DataLoader
        '''
        tensors = [
            self._to_tensor(esm_cdr3_alpha_idx),      # 0
            self._to_tensor(esm_cdr3_beta_idx),       # 1
            self._to_tensor(labels),                   # 2
            self._to_tensor(weights, torch.float),     # 3
            self._to_tensor(v_alpha_idx),              # 4
            self._to_tensor(j_alpha_idx),              # 5
            self._to_tensor(v_beta_idx),               # 6
            self._to_tensor(j_beta_idx),               # 7
            self._to_tensor(mhc_class_idx),            # 8
            self._to_tensor(mhc_allele_idx),           # 9
        ]

        dataset = TensorDataset(*tensors)

        if weighted_sampling and shuffle:
            sampler = WeightedRandomSampler(
                tensors[3].numpy().tolist(),
                len(tensors[3]),
                replacement=True
            )
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _compute_loss(self, similarity: torch.Tensor, labels: torch.Tensor,
                      weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Compute weighted loss.'''
        if self.loss_type == 'focal':
            base_loss = self.loss_fn(similarity, labels)
        else:
            base_loss = F.cross_entropy(similarity, labels, reduction='none',
                                        label_smoothing=self.label_smoothing)

        weighted_loss = (base_loss * weights).mean()
        return weighted_loss, weighted_loss

    def train_epoch(self, dataloader: DataLoader,
                    optimizer: optim.Optimizer,
                    unique_epitope_esm_idx: np.ndarray,
                    unique_mhc_class: Optional[np.ndarray] = None,
                    unique_mhc_allele: Optional[np.ndarray] = None) -> Tuple[float, float]:
        '''
        Train for one epoch.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            unique_epitope_esm_idx: ESM indices for all unique epitopes
            unique_mhc_class: MHC class for each unique epitope
            unique_mhc_allele: MHC allele for each unique epitope

        Returns:
            (loss, accuracy)
        '''
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        # Prepare unique epitope tensors
        unique_epi_idx = self._to_tensor(unique_epitope_esm_idx).to(self.device)
        n_epitopes = unique_epi_idx.size(0)

        if unique_mhc_class is not None:
            unique_mhc_cls = self._to_tensor(unique_mhc_class).to(self.device)
        else:
            unique_mhc_cls = torch.full((n_epitopes,), self.mhc_class_unk_idx,
                                        dtype=torch.long, device=self.device)

        if unique_mhc_allele is not None:
            unique_mhc_all = self._to_tensor(unique_mhc_allele).to(self.device)
        else:
            unique_mhc_all = torch.full((n_epitopes,), self.mhc_allele_unk_idx,
                                        dtype=torch.long, device=self.device)

        for batch in dataloader:
            # Unpack batch (10 tensors)
            (esm_alpha, esm_beta, labels, weights,
             v_a, j_a, v_b, j_b, mhc_cls, mhc_all) = [x.to(self.device) for x in batch]

            optimizer.zero_grad()

            # Forward pass
            sim, _, _ = self.model(
                cdr3_alpha_idx=esm_alpha,
                cdr3_beta_idx=esm_beta,
                unique_epitope_idx=unique_epi_idx,
                v_alpha=v_a,
                j_alpha=j_a,
                v_beta=v_b,
                j_beta=j_b,
                unique_mhc_class=unique_mhc_cls,
                unique_mhc_allele=unique_mhc_all
            )

            # Loss
            loss, base_loss = self._compute_loss(sim, labels, weights)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Metrics
            total_loss += base_loss.item()
            preds = sim.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(dataloader), correct / total

    def evaluate(self, dataloader: DataLoader,
                 unique_epitope_esm_idx: np.ndarray,
                 unique_mhc_class: Optional[np.ndarray] = None,
                 unique_mhc_allele: Optional[np.ndarray] = None) -> Dict:
        '''
        Evaluate model on data.

        Returns:
            Dict with loss, accuracy, predictions, labels, probabilities
        '''
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        # Prepare unique epitope tensors
        unique_epi_idx = self._to_tensor(unique_epitope_esm_idx).to(self.device)
        n_epitopes = unique_epi_idx.size(0)

        if unique_mhc_class is not None:
            unique_mhc_cls = self._to_tensor(unique_mhc_class).to(self.device)
        else:
            unique_mhc_cls = torch.full((n_epitopes,), self.mhc_class_unk_idx,
                                        dtype=torch.long, device=self.device)

        if unique_mhc_allele is not None:
            unique_mhc_all = self._to_tensor(unique_mhc_allele).to(self.device)
        else:
            unique_mhc_all = torch.full((n_epitopes,), self.mhc_allele_unk_idx,
                                        dtype=torch.long, device=self.device)

        with torch.no_grad():
            for batch in dataloader:
                (esm_alpha, esm_beta, labels, weights,
                 v_a, j_a, v_b, j_b, mhc_cls, mhc_all) = [x.to(self.device) for x in batch]

                # Forward pass
                sim, _, _ = self.model(
                    cdr3_alpha_idx=esm_alpha,
                    cdr3_beta_idx=esm_beta,
                    unique_epitope_idx=unique_epi_idx,
                    v_alpha=v_a,
                    j_alpha=j_a,
                    v_beta=v_b,
                    j_beta=j_b,
                    unique_mhc_class=unique_mhc_cls,
                    unique_mhc_allele=unique_mhc_all
                )

                # Loss
                if self.loss_type == 'focal':
                    loss = self.loss_fn(sim, labels)
                else:
                    loss = F.cross_entropy(sim, labels, reduction='none',
                                          label_smoothing=self.label_smoothing)
                total_loss += (loss * weights).mean().item()

                # Predictions
                probs = F.softmax(sim, dim=1)
                preds = sim.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.vstack(all_probs)
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            unique_epitope_esm_idx: np.ndarray,
            unique_mhc_class: Optional[np.ndarray] = None,
            unique_mhc_allele: Optional[np.ndarray] = None,
            epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4,
            patience: int = 10, min_delta: float = 1e-4) -> Dict:
        '''
        Full training loop with early stopping.

        Returns:
            Training history dict
        '''
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        print(f'Training: {len(train_loader.dataset)} samples, {epochs} max epochs')
        print(f'  Unique epitopes: {len(unique_epitope_esm_idx)}')

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer,
                unique_epitope_esm_idx, unique_mhc_class, unique_mhc_allele
            )

            # Validate
            val_results = self.evaluate(
                val_loader,
                unique_epitope_esm_idx, unique_mhc_class, unique_mhc_allele
            )

            scheduler.step(val_results['loss'])

            # History
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_acc'].append(val_results['accuracy'])

            # Early stopping
            if val_results['loss'] < best_loss - min_delta:
                best_loss = val_results['loss']
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            # Logging
            if (epoch + 1) % 5 == 0 or patience_counter == 0:
                print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
                      f'val_loss={val_results[\"loss\"]:.4f}, val_acc={val_results[\"accuracy\"]:.4f}, '
                      f'lr={optimizer.param_groups[0][\"lr\"]:.2e}, patience={patience_counter}/{patience}')

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return dict(self.history)

    def get_tcr_embeddings(self, esm_cdr3_alpha_idx, esm_cdr3_beta_idx,
                           v_alpha_idx, j_alpha_idx,
                           v_beta_idx, j_beta_idx) -> np.ndarray:
        '''Get TCR embeddings for a batch.'''
        self.model.eval()

        with torch.no_grad():
            emb = self.model.encode_tcr(
                self._to_tensor(esm_cdr3_alpha_idx).to(self.device),
                self._to_tensor(esm_cdr3_beta_idx).to(self.device),
                self._to_tensor(v_alpha_idx).to(self.device),
                self._to_tensor(j_alpha_idx).to(self.device),
                self._to_tensor(v_beta_idx).to(self.device),
                self._to_tensor(j_beta_idx).to(self.device)
            )

        return emb.cpu().numpy()
")

# ===== R Wrapper Functions =====

#' Train TCR-Epitope Model V10
#'
#' @param model V10 model with ESM embeddings initialized
#' @param data_splits Data splits with ESM indices from add_embedding_indices()
#' @param epochs Maximum training epochs
#' @param batch_size Batch size
#' @param learning_rate Learning rate
#' @param patience Early stopping patience
#' @param device Device ('cpu' or 'cuda')
#' @param loss_type Loss function: 'ce' or 'focal'
#' @param focal_gamma Focal loss gamma
#' @param label_smoothing Label smoothing factor
#' @param mhc_class_unk_idx UNK index for MHC class
#' @param mhc_allele_unk_idx UNK index for MHC allele
#' @return List with trained model, trainer, and history
#' @export
train_tcr_epitope_model_v10 <- function(model, data_splits,
                                        epochs = 100L,
                                        batch_size = 32L,
                                        learning_rate = 1e-3,
                                        patience = 15L,
                                        device = "cuda",
                                        loss_type = "focal",
                                        focal_gamma = 2.0,
                                        label_smoothing = 0.0,
                                        mhc_class_unk_idx = 3L,
                                        mhc_allele_unk_idx = 1L) {

  # Validate device
  if (device == "cuda" && !py$torch$cuda$is_available()) {
    message("CUDA not available, using CPU")
    device <- "cpu"
  }

  # Validate data has ESM indices
  required_cols <- c("esm_cdr3_alpha_idx", "esm_cdr3_beta_idx")
  missing <- setdiff(required_cols, names(data_splits$train))
  if (length(missing) > 0) {
    stop("Missing ESM indices in data_splits. Run add_embedding_indices() first.\n",
         "Missing: ", paste(missing, collapse = ", "))
  }

  # Create trainer
  trainer <- py$TCRTrainerV10(
    model = model,
    device = device,
    loss_type = loss_type,
    focal_gamma = focal_gamma,
    label_smoothing = label_smoothing,
    mhc_class_unk_idx = as.integer(mhc_class_unk_idx),
    mhc_allele_unk_idx = as.integer(mhc_allele_unk_idx)
  )

  # Create data loaders
  train_loader <- trainer$create_dataloader(
    esm_cdr3_alpha_idx = data_splits$train$esm_cdr3_alpha_idx,
    esm_cdr3_beta_idx = data_splits$train$esm_cdr3_beta_idx,
    labels = data_splits$train$labels,
    weights = data_splits$train$weights,
    v_alpha_idx = data_splits$train$v_alpha_idx,
    j_alpha_idx = data_splits$train$j_alpha_idx,
    v_beta_idx = data_splits$train$v_beta_idx,
    j_beta_idx = data_splits$train$j_beta_idx,
    mhc_class_idx = data_splits$train$mhc_class_idx,
    mhc_allele_idx = data_splits$train$mhc_allele_idx,
    batch_size = as.integer(batch_size),
    shuffle = TRUE,
    weighted_sampling = TRUE
  )

  val_loader <- trainer$create_dataloader(
    esm_cdr3_alpha_idx = data_splits$validation$esm_cdr3_alpha_idx,
    esm_cdr3_beta_idx = data_splits$validation$esm_cdr3_beta_idx,
    labels = data_splits$validation$labels,
    weights = data_splits$validation$weights,
    v_alpha_idx = data_splits$validation$v_alpha_idx,
    j_alpha_idx = data_splits$validation$j_alpha_idx,
    v_beta_idx = data_splits$validation$v_beta_idx,
    j_beta_idx = data_splits$validation$j_beta_idx,
    mhc_class_idx = data_splits$validation$mhc_class_idx,
    mhc_allele_idx = data_splits$validation$mhc_allele_idx,
    batch_size = as.integer(batch_size),
    shuffle = FALSE,
    weighted_sampling = FALSE
  )

  # Train
  history <- trainer$fit(
    train_loader = train_loader,
    val_loader = val_loader,
    unique_epitope_esm_idx = data_splits$unique_epitope_esm_idx,
    unique_mhc_class = data_splits$unique_epitope_mhc_class,
    unique_mhc_allele = data_splits$unique_epitope_mhc_allele,
    epochs = as.integer(epochs),
    lr = learning_rate,
    patience = as.integer(patience)
  )

  list(
    model = model,
    trainer = trainer,
    history = history,
    train_loader = train_loader,
    val_loader = val_loader
  )
}

#' Evaluate V10 model on test set
#'
#' @param trainer TCRTrainerV10 instance
#' @param data_splits Data splits with ESM indices
#' @param idx_to_epitope Mapping from index to epitope sequence
#' @return List with evaluation metrics and predictions
#' @export
evaluate_model_v10 <- function(trainer, data_splits, idx_to_epitope = NULL) {

  # Create test loader

  test_loader <- trainer$create_dataloader(
    esm_cdr3_alpha_idx = data_splits$test$esm_cdr3_alpha_idx,
    esm_cdr3_beta_idx = data_splits$test$esm_cdr3_beta_idx,
    labels = data_splits$test$labels,
    weights = data_splits$test$weights,
    v_alpha_idx = data_splits$test$v_alpha_idx,
    j_alpha_idx = data_splits$test$j_alpha_idx,
    v_beta_idx = data_splits$test$v_beta_idx,
    j_beta_idx = data_splits$test$j_beta_idx,
    mhc_class_idx = data_splits$test$mhc_class_idx,
    mhc_allele_idx = data_splits$test$mhc_allele_idx,
    batch_size = 64L,
    shuffle = FALSE,
    weighted_sampling = FALSE
  )

  # Evaluate
  results <- trainer$evaluate(
    test_loader,
    data_splits$unique_epitope_esm_idx,
    data_splits$unique_epitope_mhc_class,
    data_splits$unique_epitope_mhc_allele
  )

  preds <- as.integer(results$predictions)
  labels <- as.integer(results$labels)
  probs <- results$probabilities

  # Top-k accuracies
  top5 <- calc_topk_accuracy(probs, labels, k = 5)
  top10 <- calc_topk_accuracy(probs, labels, k = 10)

  # Per-class metrics
  epi_names <- if (!is.null(idx_to_epitope)) {
    idx_to_epitope[as.character(0:(ncol(probs) - 1))]
  } else NULL
  per_class <- calc_per_class_metrics(preds, labels, epi_names)

  # Build predictions dataframe
  pred_df <- tibble(
    cdr3_alpha = data_splits$test$data$cdr3_alpha,
    cdr3_beta = data_splits$test$data$cdr3_beta,
    true_epitope = data_splits$test$data$epitope,
    true_label = labels,
    predicted_label = preds,
    predicted_epitope = if (!is.null(idx_to_epitope)) idx_to_epitope[as.character(preds)] else as.character(preds),
    correct = preds == labels,
    confidence = apply(probs, 1, max),
    is_paired = data_splits$test$data$is_paired,
    source = data_splits$test$data$source,
    source_type = data_splits$test$data$source_type
  )

  # Add MHC info
  if ("mhc_class_inferred" %in% names(data_splits$test$data)) {
    pred_df$mhc_class <- data_splits$test$data$mhc_class_inferred
  }
  if ("mhc_allele_std" %in% names(data_splits$test$data)) {
    pred_df$mhc_allele <- data_splits$test$data$mhc_allele_std
  }

  # Summary
  cat("\n", strrep("=", 50), "\nV10 EVALUATION RESULTS\n", strrep("=", 50), "\n")
  cat(sprintf("Accuracy: %.1f%%, Top-5: %.1f%%, Top-10: %.1f%%\n",
              results$accuracy * 100, top5 * 100, top10 * 100))

  list(
    overall = list(
      test_loss = results$loss,
      accuracy = results$accuracy,
      top5_accuracy = top5,
      top10_accuracy = top10,
      macro_f1 = mean(per_class$f1, na.rm = TRUE)
    ),
    per_class = per_class,
    predictions = pred_df,
    probabilities = probs
  )
}

#' Create Phase 2 splits for V10 (Mouse fine-tuning)
#'
#' @param data_splits Full data splits with ESM indices
#' @param replay_fraction Fraction of human data to replay
#' @param replay_stratified Stratify replay by epitope
#' @param seed Random seed
#' @return List with Phase 2 training splits
#' @export
create_phase2_splits_v10 <- function(data_splits, replay_fraction = 0.03,
                                     replay_stratified = TRUE, seed = 42) {
  set.seed(seed)

  # Mouse indices
  train_mouse <- which(data_splits$train$data$source == "mouse")
  val_mouse <- which(data_splits$validation$data$source == "mouse")
  test_mouse <- which(data_splits$test$data$source == "mouse")

  message("Mouse samples: train=", length(train_mouse),
          ", val=", length(val_mouse), ", test=", length(test_mouse))

  # Replay buffer from human training data
  train_human <- which(data_splits$train$data$source == "human")
  n_human <- length(train_human)
  replay_idx <- integer(0)

  if (replay_fraction > 0 && n_human > 0) {
    n_replay <- ceiling(n_human * replay_fraction)

    if (replay_stratified) {
      human_data <- data_splits$train$data[train_human, ] %>%
        mutate(orig_idx = train_human)

      epi_groups <- human_data %>%
        group_by(epitope) %>%
        summarise(indices = list(orig_idx), .groups = "drop")

      per_epi <- max(1, n_replay %/% nrow(epi_groups))
      replay_idx <- unlist(lapply(epi_groups$indices, function(x) {
        sample(x, min(per_epi, length(x)))
      }))
      if (length(replay_idx) > n_replay) replay_idx <- sample(replay_idx, n_replay)
    } else {
      replay_idx <- sample(train_human, n_replay)
    }
    message("Replay buffer: ", length(replay_idx), " samples (",
            round(length(replay_idx) / n_human * 100, 1), "% of human)")
  }

  phase2_train_idx <- c(train_mouse, replay_idx)

  # Subset helper for V10 (includes ESM indices)
  subset_split_v10 <- function(split, idx) {
    list(
      data = split$data[idx, ],
      esm_cdr3_alpha_idx = split$esm_cdr3_alpha_idx[idx],
      esm_cdr3_beta_idx = split$esm_cdr3_beta_idx[idx],
      v_alpha_idx = split$v_alpha_idx[idx],
      j_alpha_idx = split$j_alpha_idx[idx],
      v_beta_idx = split$v_beta_idx[idx],
      j_beta_idx = split$j_beta_idx[idx],
      mhc_class_idx = split$mhc_class_idx[idx],
      mhc_allele_idx = split$mhc_allele_idx[idx],
      labels = split$labels[idx],
      weights = split$weights[idx]
    )
  }

  p2_train <- subset_split_v10(data_splits$train, phase2_train_idx)
  p2_train$is_replay <- c(rep(FALSE, length(train_mouse)), rep(TRUE, length(replay_idx)))

  message("Phase 2 train: ", length(train_mouse), " mouse + ", length(replay_idx), " replay")

  list(
    train = p2_train,
    validation = subset_split_v10(data_splits$validation, val_mouse),
    test = subset_split_v10(data_splits$test, test_mouse),
    epitope_to_idx = data_splits$epitope_to_idx,
    idx_to_epitope = data_splits$idx_to_epitope,
    unique_epitopes = data_splits$unique_epitopes,
    unique_epitope_esm_idx = data_splits$unique_epitope_esm_idx,
    unique_epitope_mhc_class = data_splits$unique_epitope_mhc_class,
    unique_epitope_mhc_allele = data_splits$unique_epitope_mhc_allele,
    trb_vocab = data_splits$trb_vocab,
    tra_vocab = data_splits$tra_vocab,
    mhc_vocab = data_splits$mhc_vocab,
    emb_cache = data_splits$emb_cache,
    replay_config = list(
      fraction = replay_fraction,
      n_replay = length(replay_idx),
      stratified = replay_stratified
    )
  )
}

#' V10 Default configuration
#' @return List with V10 default training configuration
#' @export
default_config_v10 <- function() {
  list(
    # ESM-2
    esm_model_name = "facebook/esm2_t30_150M_UR50D",
    esm_dim = 640L,
    esm_batch_size = 32L,
    esm_cache_path = "data/esm_embeddings_cache.h5",

    # Architecture
    hidden_dim = 256L,
    output_dim = 256L,
    dropout = 0.3,
    v_embed_dim = 32L,
    j_embed_dim = 16L,
    fusion_type = "concat",

    # MHC
    mhc_class_embed_dim = 8L,
    mhc_allele_embed_dim = 32L,
    mhc_class_unk_idx = 3L,
    mhc_allele_unk_idx = 1L,

    # Loss
    loss_type = "focal",
    focal_gamma = 2.0,
    label_smoothing = 0.1,

    # Phase 1
    phase1_epochs = 500L,
    phase1_batch_size = 2048L,  # Smaller than V9.1 due to ESM buffer memory
    phase1_learning_rate = 1e-3,
    phase1_patience = 35L,

    # Phase 2
    phase2_epochs = 200L,
    phase2_batch_size = 512L,
    phase2_learning_rate = 5e-5,
    phase2_patience = 20L,

    # Replay
    replay_fraction = 0.03,
    replay_stratified = TRUE,

    # Data
    include_unpaired = TRUE,
    iedb_filter = "all",
    mhc_min_allele_freq = 10L,
    test_fraction = 0.15,
    validation_fraction = 0.15,

    # Compute
    device = "cuda",

    # Output
    output_dir = "results_v10_esm",
    model_name = "tcr_epitope_v10_esm"
  )
}

#' Print Phase 2 splits summary for V10
#' @param phase2_splits Phase 2 splits from create_phase2_splits_v10()
#' @export
print_phase2_summary_v10 <- function(phase2_splits) {
  cat("\n", strrep("-", 50), "\n")
  cat("Phase 2 Data Summary (V10)\n")
  cat(strrep("-", 50), "\n")
  cat(sprintf("Training: %d samples (mouse + replay)\n", length(phase2_splits$train$labels)))
  cat(sprintf("  Mouse: %d\n", sum(!phase2_splits$train$is_replay)))
  cat(sprintf("  Replay: %d (%.1f%%)\n",
              sum(phase2_splits$train$is_replay),
              phase2_splits$replay_config$fraction * 100))
  cat(sprintf("Validation: %d samples\n", length(phase2_splits$validation$labels)))
  cat(sprintf("Test: %d samples\n", length(phase2_splits$test$labels)))
  cat(strrep("-", 50), "\n\n")
}
