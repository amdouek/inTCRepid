# TCR-Epitope model training infrastructure (V7: dual-chain TRA+TRB)

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
from typing import Optional, Dict, Tuple
import copy

# --- Focal Loss ---
class FocalLoss(nn.Module):
    '''Focal Loss for class imbalance. FL(p_t) = -(1-p_t)^gamma * log(p_t)'''

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

# --- EWC Regularizer ---
class EWCRegularizer:
    '''Elastic Weight Consolidation for catastrophic forgetting prevention.'''

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.fisher: Optional[Dict[str, torch.Tensor]] = None
        self.optimal_params: Optional[Dict[str, torch.Tensor]] = None

    def compute_fisher_information(self, dataloader: DataLoader,
                                   unique_epitope_indices: np.ndarray,
                                   num_samples: int = 5000,
                                   stratified_labels: Optional[np.ndarray] = None):
        self.model.eval()
        fisher = {n: torch.zeros_like(p, device=self.device)
                  for n, p in self.model.named_parameters() if p.requires_grad}

        unique_epi = torch.tensor(unique_epitope_indices, dtype=torch.long, device=self.device)
        dataset = dataloader.dataset
        n_total = len(dataset)

        # Sample selection (stratified or random)
        if stratified_labels is not None:
            sample_idx = self._stratified_sample(stratified_labels, num_samples, n_total)
        else:
            sample_idx = np.random.choice(n_total, min(num_samples, n_total), replace=False)

        subset_loader = DataLoader(Subset(dataset, sample_idx), batch_size=64, shuffle=False)
        n_proc = 0

        for batch in subset_loader:
            # V7 batch: 9 tensors
            tensors = [x.to(self.device) for x in batch]
            cdr3_a, cdr3_b, _, labels, _, v_a, j_a, v_b, j_b = tensors

            self.model.zero_grad()
            tcr_emb = self.model.encode_tcr(cdr3_a, cdr3_b, v_a, j_a, v_b, j_b)
            epi_emb = self.model.encode_epitope(unique_epi)
            sim = torch.mm(tcr_emb, epi_emb.t()) / self.model.temperature.clamp(min=0.01)

            probs = F.softmax(sim, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            loss = F.nll_loss(F.log_softmax(sim, dim=-1), sampled)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            n_proc += len(labels)

        for n in fisher: fisher[n] /= n_proc
        self.fisher = fisher
        print(f'  Fisher computed ({n_proc} samples)')

    def _stratified_sample(self, labels, num_samples, total):
        unique = np.unique(labels)
        per_class = max(1, num_samples // len(unique))
        selected = []
        for lbl in unique:
            idx = np.where(labels == lbl)[0]
            n = min(per_class, len(idx))
            selected.extend(np.random.choice(idx, n, replace=False))
        if len(selected) < num_samples:
            remaining = list(set(range(total)) - set(selected))
            n_add = min(num_samples - len(selected), len(remaining))
            selected.extend(np.random.choice(remaining, n_add, replace=False))
        return np.array(selected[:num_samples])

    def store_optimal_params(self):
        self.optimal_params = {n: p.clone().detach()
                               for n, p in self.model.named_parameters() if p.requires_grad}

    def compute_ewc_loss(self) -> torch.Tensor:
        if self.fisher is None or self.optimal_params is None:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return loss

# --- V7 Trainer (Dual Chain) ---
class TCRTrainerV7:
    '''Trainer for dual-chain TCR-Epitope model (V7).'''

    def __init__(self, model: nn.Module, device: str = 'cpu',
                 loss_type: str = 'focal', focal_gamma: float = 2.0,
                 label_smoothing: float = 0.0, ewc_lambda: float = 0.0,
                 blosum_lambda: float = 0.0):
        self.model = model.to(device)
        self.device = device
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.ewc_lambda = ewc_lambda
        self.blosum_lambda = blosum_lambda

        self.loss_fn = FocalLoss(focal_gamma, label_smoothing, 'none') if loss_type == 'focal' else None
        self.ewc = EWCRegularizer(model, device) if ewc_lambda > 0 else None
        self.use_blosum = blosum_lambda > 0 and hasattr(model, 'use_blosum') and model.use_blosum
        self.history = defaultdict(list)

        print(f'TCRTrainerV7: {loss_type} loss, gamma={focal_gamma}, ewc={ewc_lambda}')

    def _to_tensor(self, arr, dtype=torch.long):
        return torch.tensor(np.array(arr, copy=True), dtype=dtype)

    def create_dataloader(self, cdr3_alpha_idx, cdr3_beta_idx, epitope_idx,
                          labels, weights, v_alpha_idx, j_alpha_idx,
                          v_beta_idx, j_beta_idx, batch_size=32,
                          shuffle=True, weighted_sampling=True):
        tensors = [
            self._to_tensor(cdr3_alpha_idx), self._to_tensor(cdr3_beta_idx),
            self._to_tensor(epitope_idx), self._to_tensor(labels),
            self._to_tensor(weights, torch.float),
            self._to_tensor(v_alpha_idx), self._to_tensor(j_alpha_idx),
            self._to_tensor(v_beta_idx), self._to_tensor(j_beta_idx)
        ]
        dataset = TensorDataset(*tensors)

        if weighted_sampling and shuffle:
            sampler = WeightedRandomSampler(tensors[4].numpy().tolist(),
                                           len(tensors[4]), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _compute_loss(self, similarity, labels, weights):
        if self.loss_type == 'focal':
            base = self.loss_fn(similarity, labels)
        else:
            base = F.cross_entropy(similarity, labels, reduction='none',
                                  label_smoothing=self.label_smoothing)
        weighted = (base * weights).mean()
        total = weighted

        if self.ewc_lambda > 0 and self.ewc:
            total = total + self.ewc_lambda * self.ewc.compute_ewc_loss()
        if self.use_blosum and self.blosum_lambda > 0:
            total = total + self.blosum_lambda * self.model.compute_blosum_regularization_loss()
        return total, weighted

    def train_epoch(self, dataloader, optimizer, unique_epitope_idx):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        unique_epi = self._to_tensor(unique_epitope_idx).to(self.device)

        for batch in dataloader:
            cdr3_a, cdr3_b, _, labels, weights, v_a, j_a, v_b, j_b = [x.to(self.device) for x in batch]

            optimizer.zero_grad()
            tcr_emb = self.model.encode_tcr(cdr3_a, cdr3_b, v_a, j_a, v_b, j_b)
            epi_emb = self.model.encode_epitope(unique_epi)
            sim = torch.mm(tcr_emb, epi_emb.t()) / self.model.temperature.clamp(min=0.01)

            loss, base_loss = self._compute_loss(sim, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += base_loss.item()
            preds = sim.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(dataloader), correct / total

    def evaluate(self, dataloader, unique_epitope_idx):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        unique_epi = self._to_tensor(unique_epitope_idx).to(self.device)

        with torch.no_grad():
            epi_emb = self.model.encode_epitope(unique_epi)
            for batch in dataloader:
                cdr3_a, cdr3_b, _, labels, weights, v_a, j_a, v_b, j_b = [x.to(self.device) for x in batch]

                tcr_emb = self.model.encode_tcr(cdr3_a, cdr3_b, v_a, j_a, v_b, j_b)
                sim = torch.mm(tcr_emb, epi_emb.t()) / self.model.temperature.clamp(min=0.01)

                if self.loss_type == 'focal':
                    loss = self.loss_fn(sim, labels)
                else:
                    loss = F.cross_entropy(sim, labels, reduction='none',
                                          label_smoothing=self.label_smoothing)
                total_loss += (loss * weights).mean().item()

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

    def fit(self, train_loader, val_loader, unique_epitope_indices,
            epochs=100, lr=1e-3, weight_decay=1e-4, patience=10, min_delta=1e-4):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5, verbose=True)

        best_loss, patience_cnt, best_state = float('inf'), 0, None
        print(f'Training: {len(train_loader.dataset)} samples, {epochs} max epochs')

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, unique_epitope_indices)
            val_res = self.evaluate(val_loader, unique_epitope_indices)
            scheduler.step(val_res['loss'])

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_res['loss'])
            self.history['val_acc'].append(val_res['accuracy'])

            if val_res['loss'] < best_loss - min_delta:
                best_loss, patience_cnt = val_res['loss'], 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1

            if (epoch + 1) % 5 == 0 or patience_cnt == 0:
                v_loss = val_res['loss']
                v_acc = val_res['accuracy']
                curr_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={v_loss:.4f}, val_acc={v_acc:.4f}, lr={curr_lr:.2e}, patience={patience_cnt}/{patience}')

            if patience_cnt >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        if best_state: self.model.load_state_dict(best_state)
        return dict(self.history)

    def setup_ewc(self, dataloader, unique_epitope_indices, num_samples=5000, labels=None):
        if not self.ewc:
            print('EWC not enabled'); return
        self.ewc.compute_fisher_information(dataloader, unique_epitope_indices, num_samples, labels)
        self.ewc.store_optimal_params()

    def get_embeddings(self, cdr3_alpha_idx, cdr3_beta_idx, v_alpha_idx,
                       j_alpha_idx, v_beta_idx, j_beta_idx):
        self.model.eval()
        with torch.no_grad():
            emb = self.model.encode_tcr(
                self._to_tensor(cdr3_alpha_idx).to(self.device),
                self._to_tensor(cdr3_beta_idx).to(self.device),
                self._to_tensor(v_alpha_idx).to(self.device),
                self._to_tensor(j_alpha_idx).to(self.device),
                self._to_tensor(v_beta_idx).to(self.device),
                self._to_tensor(j_beta_idx).to(self.device)
            )
        return emb.cpu().numpy()

def compute_ewc_sample_count(n_train, min_samples=5000, max_samples=10000, fraction=0.05):
    return min(max_samples, max(min_samples, int(fraction * n_train)))
")

# ===== R Wrapper Functions =====

#' Train TCR-Epitope model (V7: dual-chain)
#'
#' @param model PyTorch model (V7)
#' @param data_splits Output from prepare_paired_chain_data()
#' @param epochs Maximum training epochs
#' @param batch_size Batch size
#' @param learning_rate Learning rate
#' @param patience Early stopping patience
#' @param device Device ('cpu' or 'cuda')
#' @param loss_type Loss function: 'ce' or 'focal'
#' @param focal_gamma Focal loss gamma parameter
#' @param label_smoothing Label smoothing factor
#' @param ewc_lambda EWC regularization strength
#' @param blosum_lambda BLOSUM regularization strength
#' @return List with trained model, trainer, and history
#' @export
train_tcr_epitope_model <- function(model, data_splits,
                                    epochs = 100, batch_size = 32,
                                    learning_rate = 1e-3, patience = 15,
                                    device = "cpu", loss_type = "focal",
                                    focal_gamma = 2.0, label_smoothing = 0.0,
                                    ewc_lambda = 0.0, blosum_lambda = 0.0) {

  if (device == "cuda" && !py$torch$cuda$is_available()) {
    message("CUDA not available, using CPU")
    device <- "cpu"
  }

  # Validate V7 data structure
  required <- c("cdr3_alpha_idx", "cdr3_beta_idx", "v_alpha_idx", "j_alpha_idx",
                "v_beta_idx", "j_beta_idx")
  missing <- setdiff(required, names(data_splits$train))
  if (length(missing) > 0) {
    stop("Missing V7 columns: ", paste(missing, collapse = ", "),
         "\nUse prepare_paired_chain_data() for V7-compatible data.")
  }

  trainer <- py$TCRTrainerV7(
    model = model, device = device, loss_type = loss_type,
    focal_gamma = focal_gamma, label_smoothing = label_smoothing,
    ewc_lambda = ewc_lambda, blosum_lambda = blosum_lambda
  )

  train_loader <- trainer$create_dataloader(
    cdr3_alpha_idx = data_splits$train$cdr3_alpha_idx,
    cdr3_beta_idx = data_splits$train$cdr3_beta_idx,
    epitope_idx = data_splits$train$epitope_idx,
    labels = data_splits$train$labels,
    weights = data_splits$train$weights,
    v_alpha_idx = data_splits$train$v_alpha_idx,
    j_alpha_idx = data_splits$train$j_alpha_idx,
    v_beta_idx = data_splits$train$v_beta_idx,
    j_beta_idx = data_splits$train$j_beta_idx,
    batch_size = as.integer(batch_size),
    shuffle = TRUE, weighted_sampling = TRUE
  )

  val_loader <- trainer$create_dataloader(
    cdr3_alpha_idx = data_splits$validation$cdr3_alpha_idx,
    cdr3_beta_idx = data_splits$validation$cdr3_beta_idx,
    epitope_idx = data_splits$validation$epitope_idx,
    labels = data_splits$validation$labels,
    weights = data_splits$validation$weights,
    v_alpha_idx = data_splits$validation$v_alpha_idx,
    j_alpha_idx = data_splits$validation$j_alpha_idx,
    v_beta_idx = data_splits$validation$v_beta_idx,
    j_beta_idx = data_splits$validation$j_beta_idx,
    batch_size = as.integer(batch_size),
    shuffle = FALSE, weighted_sampling = FALSE
  )

  history <- trainer$fit(
    train_loader = train_loader,
    val_loader = val_loader,
    unique_epitope_indices = data_splits$unique_epitope_idx,
    epochs = as.integer(epochs),
    lr = learning_rate,
    patience = as.integer(patience)
  )

  list(model = model, trainer = trainer, history = history,
       unique_epitope_idx = data_splits$unique_epitope_idx,
       train_loader = train_loader)
}

#' Default V7 configuration
#' @return List with default training configuration
#' @export
default_config <- function() {
  list(
    # Architecture
    vocab_size = 22L, token_embedding_dim = 128L, hidden_dim = 256L,
    output_dim = 256L, dropout = 0.3, v_embed_dim = 32L, j_embed_dim = 16L,
    fusion_type = "concat", use_atchley_init = TRUE, use_blosum_reg = FALSE,

    # Loss
    loss_type = "focal", focal_gamma = 2.0, label_smoothing = 0.0,

    # Regularization
    ewc_lambda = 0, ewc_min_samples = 5000, ewc_max_samples = 10000,
    ewc_fraction = 0.05, blosum_lambda = 0.0,

    # Phase 1
    phase1_epochs = 500, phase1_batch_size = 4096,
    phase1_learning_rate = 3e-3, phase1_patience = 50,

    # Phase 2
    phase2_epochs = 200, phase2_batch_size = 512,
    phase2_learning_rate = 5e-5, phase2_patience = 15,

    # Replay
    replay_fraction = 0.05, replay_stratified = TRUE,

    # Data
    include_unpaired = TRUE, cdr3_max_len = 25L, epitope_max_len = 30L,
    test_fraction = 0.15, validation_fraction = 0.15,

    # General
    device = "cuda", output_dir = "results_transfer_v7"
  )
}

#' Evaluate model on test set
#'
#' @param trainer TCRTrainerV7 instance
#' @param data_splits Data splits from prepare_paired_chain_data()
#' @param idx_to_epitope Mapping from index to epitope sequence
#' @return List with evaluation metrics and predictions
#' @export
evaluate_model <- function(trainer, data_splits, idx_to_epitope = NULL) {

  test_loader <- trainer$create_dataloader(
    cdr3_alpha_idx = data_splits$test$cdr3_alpha_idx,
    cdr3_beta_idx = data_splits$test$cdr3_beta_idx,
    epitope_idx = data_splits$test$epitope_idx,
    labels = data_splits$test$labels,
    weights = data_splits$test$weights,
    v_alpha_idx = data_splits$test$v_alpha_idx,
    j_alpha_idx = data_splits$test$j_alpha_idx,
    v_beta_idx = data_splits$test$v_beta_idx,
    j_beta_idx = data_splits$test$j_beta_idx,
    batch_size = 64L, shuffle = FALSE, weighted_sampling = FALSE
  )

  results <- trainer$evaluate(test_loader, data_splits$unique_epitope_idx)

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

  pred_df <- tibble(
    cdr3_alpha = data_splits$test$data$cdr3_alpha,
    cdr3_beta = data_splits$test$data$cdr3_beta,
    true_epitope = data_splits$test$data$epitope,
    true_label = labels, predicted_label = preds,
    predicted_epitope = if (!is.null(idx_to_epitope)) idx_to_epitope[as.character(preds)] else as.character(preds),
    correct = preds == labels,
    confidence = apply(probs, 1, max),
    is_paired = data_splits$test$data$is_paired,
    source = data_splits$test$data$source,
    source_type = data_splits$test$data$source_type
  )

  # Summary
  cat("\n", strrep("=", 50), "\nEVALUATION RESULTS\n", strrep("=", 50), "\n")
  cat(sprintf("Accuracy: %.1f%%, Top-5: %.1f%%, Top-10: %.1f%%\n",
              results$accuracy * 100, top5 * 100, top10 * 100))

  list(
    overall = list(test_loss = results$loss, accuracy = results$accuracy,
                   top5_accuracy = top5, top10_accuracy = top10,
                   macro_f1 = mean(per_class$f1, na.rm = TRUE)),
    per_class = per_class, predictions = pred_df, probabilities = probs
  )
}

#' Evaluate by species
#' @param predictions Predictions tibble
#' @param test_data Test data with source column
#' @return Tibble with per-species metrics
#' @export
evaluate_by_species <- function(predictions, test_data) {

  src_col <- if ("source" %in% names(test_data)) "source"
  else if ("tcr_species" %in% names(test_data)) "tcr_species"
  else NULL

  if (is.null(src_col)) {
    return(tibble(source = "unknown", n = nrow(predictions),
                  accuracy = mean(predictions$correct, na.rm = TRUE)))
  }

  if (!"source" %in% names(predictions) && nrow(predictions) == nrow(test_data)) {
    predictions$source <- test_data[[src_col]]
  }

  by_species <- predictions %>%
    group_by(source) %>%
    summarise(n = n(), accuracy = mean(correct, na.rm = TRUE),
              mean_confidence = mean(confidence, na.rm = TRUE), .groups = "drop")

  message("\nPerformance by species:")
  for (i in seq_len(nrow(by_species))) {
    message("  ", by_species$source[i], ": ", round(by_species$accuracy[i] * 100, 1), "%")
  }
  by_species
}

# ===== Phase 2: Mouse Fine-tuning with Experience Replay =====

#' Create Phase 2 splits with experience replay
#'
#' @param data_splits Full data splits from prepare_paired_chain_data()
#' @param replay_fraction Fraction of human data to replay
#' @param replay_stratified Stratify replay by epitope
#' @param seed Random seed
#' @return List with Phase 2 training splits
#' @export
create_phase2_splits <- function(data_splits, replay_fraction = 0.05,
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
            round(length(replay_idx)/n_human*100, 1), "% of human)")
  }

  # Combine indices
  phase2_train_idx <- c(train_mouse, replay_idx)

  # Helper to subset split
  subset_split <- function(split, idx) {
    list(
      data = split$data[idx, ],
      cdr3_alpha_idx = split$cdr3_alpha_idx[idx, , drop = FALSE],
      cdr3_beta_idx = split$cdr3_beta_idx[idx, , drop = FALSE],
      epitope_idx = split$epitope_idx[idx, , drop = FALSE],
      v_alpha_idx = split$v_alpha_idx[idx],
      j_alpha_idx = split$j_alpha_idx[idx],
      v_beta_idx = split$v_beta_idx[idx],
      j_beta_idx = split$j_beta_idx[idx],
      labels = split$labels[idx],
      weights = split$weights[idx]
    )
  }

  p2_train <- subset_split(data_splits$train, phase2_train_idx)
  p2_train$is_replay <- c(rep(FALSE, length(train_mouse)), rep(TRUE, length(replay_idx)))

  message("Phase 2 train: ", length(train_mouse), " mouse + ", length(replay_idx), " replay")

  list(
    train = p2_train,
    validation = subset_split(data_splits$validation, val_mouse),
    test = subset_split(data_splits$test, test_mouse),
    epitope_to_idx = data_splits$epitope_to_idx,
    idx_to_epitope = data_splits$idx_to_epitope,
    unique_epitopes = data_splits$unique_epitopes,
    unique_epitope_idx = data_splits$unique_epitope_idx,
    trb_vocab = data_splits$trb_vocab,
    tra_vocab = data_splits$tra_vocab,
    replay_config = list(fraction = replay_fraction, n_replay = length(replay_idx),
                         stratified = replay_stratified)
  )
}

# ===== Transfer Learning Pipeline =====

#' Run transfer learning pipeline
#'
#' Phase 1: Train on combined human + mouse data
#' Phase 2: Fine-tune on mouse with experience replay
#'
#' @param vdjdb_raw Raw VDJdb data
#' @param trb_vocab TRB V/J vocabulary
#' @param tra_vocab TRA V/J vocabulary
#' @param config Configuration list (see default_config())
#' @param fine_tune_mouse If TRUE, perform Phase 2
#' @return Training results
#' @export
run_transfer_learning_pipeline <- function(vdjdb_raw, trb_vocab, tra_vocab,
                                           config = NULL, fine_tune_mouse = TRUE) {

  config <- modifyList(default_config(), config %||% list())
  if (!dir.exists(config$output_dir)) dir.create(config$output_dir, recursive = TRUE)

  message("\n", strrep("=", 70))
  message("TRANSFER LEARNING PIPELINE (V7 Dual Chain)")
  message(strrep("=", 70))

  # === Data Preparation ===
  message("\nPreparing paired chain data...")
  data_splits <- prepare_paired_chain_data(
    vdjdb_raw, trb_vocab, tra_vocab,
    include_unpaired = config$include_unpaired,
    test_fraction = config$test_fraction,
    validation_fraction = config$validation_fraction
  )
  print_paired_data_summary(data_splits)

  # === Phase 1 ===
  message("\n", strrep("-", 70))
  message("PHASE 1: Combined Training")
  message(strrep("-", 70))

  model <- create_tcr_epitope_model(
    vocab_size = config$vocab_size,
    embed_dim = config$token_embedding_dim,
    hidden_dim = config$hidden_dim,
    output_dim = config$output_dim,
    dropout = config$dropout,
    trb_v_vocab_size = as.integer(trb_vocab$v$size),
    trb_j_vocab_size = as.integer(trb_vocab$j$size),
    tra_v_vocab_size = as.integer(tra_vocab$v$size),
    tra_j_vocab_size = as.integer(tra_vocab$j$size),
    v_embed_dim = config$v_embed_dim,
    j_embed_dim = config$j_embed_dim,
    fusion = config$fusion_type,
    use_atchley_init = config$use_atchley_init,
    use_blosum_reg = config$use_blosum_reg
  )

  p1_result <- train_tcr_epitope_model(
    model, data_splits,
    epochs = config$phase1_epochs,
    batch_size = config$phase1_batch_size,
    learning_rate = config$phase1_learning_rate,
    patience = config$phase1_patience,
    device = config$device,
    loss_type = config$loss_type,
    focal_gamma = config$focal_gamma,
    ewc_lambda = 0.0
  )

  p1_eval <- evaluate_model(p1_result$trainer, data_splits, data_splits$idx_to_epitope)
  p1_species <- evaluate_by_species(p1_eval$predictions, data_splits$test$data)

  results <- list(
    phase1 = list(model = p1_result$model, trainer = p1_result$trainer,
                  history = p1_result$history, evaluation = p1_eval,
                  by_species = p1_species),
    data_splits = data_splits, config = config
  )

  # === Phase 2 ===
  if (fine_tune_mouse) {
    message("\n", strrep("-", 70))
    message("PHASE 2: Mouse Fine-tuning (", config$replay_fraction*100, "% replay)")
    message(strrep("-", 70))

    p2_splits <- create_phase2_splits(data_splits, config$replay_fraction,
                                      config$replay_stratified)

    p2_trainer <- py$TCRTrainerV7(
      model = p1_result$model, device = config$device,
      loss_type = config$loss_type, focal_gamma = config$focal_gamma,
      label_smoothing = config$label_smoothing, ewc_lambda = 0.0
    )

    p2_train_loader <- p2_trainer$create_dataloader(
      cdr3_alpha_idx = p2_splits$train$cdr3_alpha_idx,
      cdr3_beta_idx = p2_splits$train$cdr3_beta_idx,
      epitope_idx = p2_splits$train$epitope_idx,
      labels = p2_splits$train$labels,
      weights = p2_splits$train$weights,
      v_alpha_idx = p2_splits$train$v_alpha_idx,
      j_alpha_idx = p2_splits$train$j_alpha_idx,
      v_beta_idx = p2_splits$train$v_beta_idx,
      j_beta_idx = p2_splits$train$j_beta_idx,
      batch_size = as.integer(config$phase2_batch_size),
      shuffle = TRUE, weighted_sampling = TRUE
    )

    p2_val_loader <- p2_trainer$create_dataloader(
      cdr3_alpha_idx = p2_splits$validation$cdr3_alpha_idx,
      cdr3_beta_idx = p2_splits$validation$cdr3_beta_idx,
      epitope_idx = p2_splits$validation$epitope_idx,
      labels = p2_splits$validation$labels,
      weights = p2_splits$validation$weights,
      v_alpha_idx = p2_splits$validation$v_alpha_idx,
      j_alpha_idx = p2_splits$validation$j_alpha_idx,
      v_beta_idx = p2_splits$validation$v_beta_idx,
      j_beta_idx = p2_splits$validation$j_beta_idx,
      batch_size = as.integer(config$phase2_batch_size),
      shuffle = FALSE, weighted_sampling = FALSE
    )

    p2_history <- p2_trainer$fit(
      p2_train_loader, p2_val_loader, p2_splits$unique_epitope_idx,
      epochs = as.integer(config$phase2_epochs),
      lr = config$phase2_learning_rate,
      patience = as.integer(config$phase2_patience)
    )

    p2_eval_mouse <- evaluate_model(p2_trainer, p2_splits, data_splits$idx_to_epitope)
    p2_eval_full <- evaluate_model(p2_trainer, data_splits, data_splits$idx_to_epitope)
    p2_species <- evaluate_by_species(p2_eval_full$predictions, data_splits$test$data)

    # Forgetting analysis
    p1_human <- p1_species$accuracy[p1_species$source == "human"]
    p2_human <- p2_species$accuracy[p2_species$source == "human"]
    p1_mouse <- p1_species$accuracy[p1_species$source == "mouse"]
    p2_mouse <- p2_species$accuracy[p2_species$source == "mouse"]

    results$phase2 <- list(
      model = p1_result$model, trainer = p2_trainer, history = p2_history,
      evaluation_mouse = p2_eval_mouse, evaluation_full = p2_eval_full,
      by_species = p2_species, phase2_splits = p2_splits,
      forgetting = list(
        phase1_human = p1_human, phase2_human = p2_human,
        phase1_mouse = p1_mouse, phase2_mouse = p2_mouse,
        human_change = p2_human - p1_human,
        mouse_change = p2_mouse - p1_mouse
      )
    )
  }

  # === Summary ===
  message("\n", strrep("=", 70))
  message("PIPELINE COMPLETE")
  message(strrep("=", 70))
  message("\nPhase 1: ", round(p1_eval$overall$accuracy * 100, 1), "% accuracy")

  if (fine_tune_mouse && !is.null(results$phase2)) {
    message("Phase 2 (mouse): ", round(results$phase2$evaluation_mouse$overall$accuracy * 100, 1), "%")
    message("Forgetting: ", sprintf("%+.1f%%", results$phase2$forgetting$human_change * 100))
  }

  results
}
