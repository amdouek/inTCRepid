# TCR-Epitope Model Training Infrastructure (V9.1: MHC Integration)
#
# Batch structure: 11 tensors
#   (cdr3_a, cdr3_b, epitope_idx, labels, weights,
#    v_a, j_a, v_b, j_b, mhc_class, mhc_allele)
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

    def __init__(self, model: nn.Module, device: str = 'cpu',
                 mhc_class_unk: int = 3, mhc_allele_unk: int = 1):
        self.model = model
        self.device = device
        self.mhc_class_unk = mhc_class_unk
        self.mhc_allele_unk = mhc_allele_unk
        self.fisher: Optional[Dict[str, torch.Tensor]] = None
        self.optimal_params: Optional[Dict[str, torch.Tensor]] = None

    def compute_fisher_information(self, dataloader: DataLoader,
                                   unique_epitope_indices: np.ndarray,
                                   num_samples: int = 5000,
                                   stratified_labels: Optional[np.ndarray] = None,
                                   unique_epitope_mhc: Optional[Tuple] = None):
        self.model.eval()
        fisher = {n: torch.zeros_like(p, device=self.device)
                  for n, p in self.model.named_parameters() if p.requires_grad}

        unique_epi = torch.tensor(unique_epitope_indices, dtype=torch.long, device=self.device)
        n_epitopes = unique_epi.size(0)

        # MHC for unique epitopes
        if unique_epitope_mhc is not None:
            epi_mhc_class = torch.tensor(unique_epitope_mhc[0], dtype=torch.long, device=self.device)
            epi_mhc_allele = torch.tensor(unique_epitope_mhc[1], dtype=torch.long, device=self.device)
        else:
            epi_mhc_class = torch.full((n_epitopes,), self.mhc_class_unk, dtype=torch.long, device=self.device)
            epi_mhc_allele = torch.full((n_epitopes,), self.mhc_allele_unk, dtype=torch.long, device=self.device)

        dataset = dataloader.dataset
        n_total = len(dataset)

        if stratified_labels is not None:
            sample_idx = self._stratified_sample(stratified_labels, num_samples, n_total)
        else:
            sample_idx = np.random.choice(n_total, min(num_samples, n_total), replace=False)

        subset_loader = DataLoader(Subset(dataset, sample_idx), batch_size=64, shuffle=False)
        n_proc = 0

        for batch in subset_loader:
            tensors = [x.to(self.device) for x in batch]
            cdr3_a, cdr3_b, _, labels, _, v_a, j_a, v_b, j_b, mhc_class, mhc_allele = tensors

            self.model.zero_grad()
            tcr_emb = self.model.encode_tcr(cdr3_a, cdr3_b, v_a, j_a, v_b, j_b)

            if hasattr(self.model, 'encode_epitope_mhc'):
                epi_emb = self.model.encode_epitope_mhc(unique_epi, epi_mhc_class, epi_mhc_allele)
            else:
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

# --- TCR Trainer ---
class TCRTrainer:
    '''
    Trainer for TCR-Epitope model with MHC support.

    Batch structure: 11 tensors
      (cdr3_a, cdr3_b, epitope_idx, labels, weights,
       v_a, j_a, v_b, j_b, mhc_class, mhc_allele)
    '''

    def __init__(self, model: nn.Module, device: str = 'cpu',
                 loss_type: str = 'focal', focal_gamma: float = 2.0,
                 label_smoothing: float = 0.0, ewc_lambda: float = 0.0,
                 blosum_lambda: float = 0.0,
                 mhc_class_unk_idx: int = 3, mhc_allele_unk_idx: int = 1):
        self.model = model.to(device)
        self.device = device
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.ewc_lambda = ewc_lambda
        self.blosum_lambda = blosum_lambda
        self.mhc_class_unk_idx = mhc_class_unk_idx
        self.mhc_allele_unk_idx = mhc_allele_unk_idx

        self.loss_fn = FocalLoss(focal_gamma, label_smoothing, 'none') if loss_type == 'focal' else None
        self.ewc = EWCRegularizer(model, device, mhc_class_unk_idx, mhc_allele_unk_idx) if ewc_lambda > 0 else None
        self.use_blosum = blosum_lambda > 0 and hasattr(model, 'use_blosum') and model.use_blosum
        self.history = defaultdict(list)
        self.model_has_mhc = hasattr(model, 'encode_epitope_mhc')

        print(f'TCRTrainer: {loss_type} loss, gamma={focal_gamma}, ewc={ewc_lambda}, mhc={self.model_has_mhc}')

    def _to_tensor(self, arr, dtype=torch.long):
        return torch.tensor(np.array(arr, copy=True), dtype=dtype)

    def create_dataloader(self, cdr3_alpha_idx, cdr3_beta_idx, epitope_idx,
                          labels, weights, v_alpha_idx, j_alpha_idx,
                          v_beta_idx, j_beta_idx, mhc_class_idx, mhc_allele_idx,
                          batch_size=32, shuffle=True, weighted_sampling=True):
        '''Create dataloader with 11-tensor batch.'''
        tensors = [
            self._to_tensor(cdr3_alpha_idx),
            self._to_tensor(cdr3_beta_idx),
            self._to_tensor(epitope_idx),
            self._to_tensor(labels),
            self._to_tensor(weights, torch.float),
            self._to_tensor(v_alpha_idx),
            self._to_tensor(j_alpha_idx),
            self._to_tensor(v_beta_idx),
            self._to_tensor(j_beta_idx),
            self._to_tensor(mhc_class_idx),
            self._to_tensor(mhc_allele_idx),
        ]
        dataset = TensorDataset(*tensors)

        if weighted_sampling and shuffle:
            sampler = WeightedRandomSampler(tensors[4].numpy().tolist(),
                                           len(tensors[4]), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _encode_unique_epitopes(self, unique_epitope_idx: torch.Tensor,
                                unique_mhc_class: Optional[torch.Tensor] = None,
                                unique_mhc_allele: Optional[torch.Tensor] = None):
        '''Encode unique epitopes with optional MHC context.'''
        n_epitopes = unique_epitope_idx.size(0)

        if not self.model_has_mhc:
            return self.model.encode_epitope(unique_epitope_idx)

        if unique_mhc_class is None:
            unique_mhc_class = torch.full((n_epitopes,), self.mhc_class_unk_idx,
                                          dtype=torch.long, device=self.device)
        if unique_mhc_allele is None:
            unique_mhc_allele = torch.full((n_epitopes,), self.mhc_allele_unk_idx,
                                           dtype=torch.long, device=self.device)

        return self.model.encode_epitope_mhc(unique_epitope_idx, unique_mhc_class, unique_mhc_allele)

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

    def train_epoch(self, dataloader, optimizer, unique_epitope_idx,
                    unique_mhc_class=None, unique_mhc_allele=None):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        unique_epi = self._to_tensor(unique_epitope_idx).to(self.device)
        if unique_mhc_class is not None:
            unique_mhc_class = self._to_tensor(unique_mhc_class).to(self.device)
        if unique_mhc_allele is not None:
            unique_mhc_allele = self._to_tensor(unique_mhc_allele).to(self.device)

        for batch in dataloader:
            (cdr3_a, cdr3_b, _, labels, weights,
             v_a, j_a, v_b, j_b, mhc_class, mhc_allele) = [x.to(self.device) for x in batch]

            optimizer.zero_grad()
            tcr_emb = self.model.encode_tcr(cdr3_a, cdr3_b, v_a, j_a, v_b, j_b)
            epi_emb = self._encode_unique_epitopes(unique_epi, unique_mhc_class, unique_mhc_allele)
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

    def evaluate(self, dataloader, unique_epitope_idx,
                 unique_mhc_class=None, unique_mhc_allele=None):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        unique_epi = self._to_tensor(unique_epitope_idx).to(self.device)
        if unique_mhc_class is not None:
            unique_mhc_class = self._to_tensor(unique_mhc_class).to(self.device)
        if unique_mhc_allele is not None:
            unique_mhc_allele = self._to_tensor(unique_mhc_allele).to(self.device)

        with torch.no_grad():
            epi_emb = self._encode_unique_epitopes(unique_epi, unique_mhc_class, unique_mhc_allele)

            for batch in dataloader:
                (cdr3_a, cdr3_b, _, labels, weights,
                 v_a, j_a, v_b, j_b, mhc_class, mhc_allele) = [x.to(self.device) for x in batch]

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
            unique_mhc_class=None, unique_mhc_allele=None,
            epochs=100, lr=1e-3, weight_decay=1e-4, patience=10, min_delta=1e-4):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5, verbose=True)

        best_loss, patience_cnt, best_state = float('inf'), 0, None
        print(f'Training: {len(train_loader.dataset)} samples, {epochs} max epochs')

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, unique_epitope_indices,
                unique_mhc_class, unique_mhc_allele)
            val_res = self.evaluate(
                val_loader, unique_epitope_indices,
                unique_mhc_class, unique_mhc_allele)
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
                print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
                      f'val_loss={val_res[\"loss\"]:.4f}, val_acc={val_res[\"accuracy\"]:.4f}, '
                      f'lr={optimizer.param_groups[0][\"lr\"]:.2e}, patience={patience_cnt}/{patience}')

            if patience_cnt >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        if best_state: self.model.load_state_dict(best_state)
        return dict(self.history)

    def setup_ewc(self, dataloader, unique_epitope_indices,
                  num_samples=5000, labels=None, unique_mhc=None):
        if not self.ewc:
            print('EWC not enabled'); return
        self.ewc.compute_fisher_information(
            dataloader, unique_epitope_indices, num_samples, labels, unique_mhc)
        self.ewc.store_optimal_params()

    def get_tcr_embeddings(self, cdr3_alpha_idx, cdr3_beta_idx, v_alpha_idx,
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

#' Train TCR-Epitope model
#'
#' @param model PyTorch model with MHC support
#' @param data_splits Output from prepare_combined_data() with MHC columns
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
#' @param mhc_class_unk_idx Index for UNK MHC class (default 3)
#' @param mhc_allele_unk_idx Index for UNK MHC allele (default 1)
#' @return List with trained model, trainer, and history
#' @export
train_tcr_epitope_model <- function(model, data_splits,
                                    epochs = 100, batch_size = 32,
                                    learning_rate = 1e-3, patience = 15,
                                    device = "cpu", loss_type = "focal",
                                    focal_gamma = 2.0, label_smoothing = 0.0,
                                    ewc_lambda = 0.0, blosum_lambda = 0.0,
                                    mhc_class_unk_idx = 3L,
                                    mhc_allele_unk_idx = 1L) {

  if (device == "cuda" && !py$torch$cuda$is_available()) {
    message("CUDA not available, using CPU")
    device <- "cpu"
  }

  # Validate data structure
  required <- c("cdr3_alpha_idx", "cdr3_beta_idx", "v_alpha_idx", "j_alpha_idx",
                "v_beta_idx", "j_beta_idx", "mhc_class_idx", "mhc_allele_idx")
  missing <- setdiff(required, names(data_splits$train))
  if (length(missing) > 0) {
    stop("Missing columns: ", paste(missing, collapse = ", "),
         "\nEnsure data includes MHC encoding from prepare_combined_data().")
  }

  trainer <- py$TCRTrainer(
    model = model, device = device, loss_type = loss_type,
    focal_gamma = focal_gamma, label_smoothing = label_smoothing,
    ewc_lambda = ewc_lambda, blosum_lambda = blosum_lambda,
    mhc_class_unk_idx = as.integer(mhc_class_unk_idx),
    mhc_allele_unk_idx = as.integer(mhc_allele_unk_idx)
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
    mhc_class_idx = data_splits$train$mhc_class_idx,
    mhc_allele_idx = data_splits$train$mhc_allele_idx,
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
    mhc_class_idx = data_splits$validation$mhc_class_idx,
    mhc_allele_idx = data_splits$validation$mhc_allele_idx,
    batch_size = as.integer(batch_size),
    shuffle = FALSE, weighted_sampling = FALSE
  )

  history <- trainer$fit(
    train_loader = train_loader,
    val_loader = val_loader,
    unique_epitope_indices = data_splits$unique_epitope_idx,
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
    unique_epitope_idx = data_splits$unique_epitope_idx,
    unique_mhc_class = data_splits$unique_epitope_mhc_class,
    unique_mhc_allele = data_splits$unique_epitope_mhc_allele,
    train_loader = train_loader
  )
}

#' Default configuration
#' @return List with default training configuration
#' @export
default_config <- function() {
  list(
    # Architecture
    vocab_size = 22L,
    token_embedding_dim = 128L,
    hidden_dim = 256L,
    output_dim = 256L,
    dropout = 0.3,
    v_embed_dim = 32L,
    j_embed_dim = 16L,
    fusion_type = "concat",
    use_atchley_init = TRUE,
    use_blosum_reg = FALSE,

    # MHC
    mhc_class_embed_dim = 8L,
    mhc_allele_embed_dim = 32L,
    mhc_class_vocab_size = 4L,
    mhc_allele_vocab_size = 116L,
    mhc_class_unk_idx = 3L,
    mhc_allele_unk_idx = 1L,

    # Loss
    loss_type = "focal",
    focal_gamma = 2.0,
    label_smoothing = 0.0,

    # Regularization
    ewc_lambda = 0,
    ewc_min_samples = 5000,
    ewc_max_samples = 10000,
    ewc_fraction = 0.05,
    blosum_lambda = 0.0,

    # Phase 1
    phase1_epochs = 500,
    phase1_batch_size = 4096,
    phase1_learning_rate = 3e-3,
    phase1_patience = 25,

    # Phase 2
    phase2_epochs = 200,
    phase2_batch_size = 512,
    phase2_learning_rate = 5e-5,
    phase2_patience = 20,

    # Replay
    replay_fraction = 0.03,
    replay_stratified = TRUE,

    # Data
    include_unpaired = TRUE,
    cdr3_max_len = 25L,
    epitope_max_len = 30L,
    test_fraction = 0.15,
    validation_fraction = 0.15,

    # General
    device = "cuda",
    output_dir = "results_v9-1"
  )
}

#' Evaluate model on test set
#'
#' @param trainer TCRTrainer instance
#' @param data_splits Data splits with MHC columns
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
    mhc_class_idx = data_splits$test$mhc_class_idx,
    mhc_allele_idx = data_splits$test$mhc_allele_idx,
    batch_size = 64L, shuffle = FALSE, weighted_sampling = FALSE
  )

  results <- trainer$evaluate(
    test_loader,
    data_splits$unique_epitope_idx,
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

  # Add MHC info if available
  if ("mhc_class_inferred" %in% names(data_splits$test$data)) {
    pred_df$mhc_class <- data_splits$test$data$mhc_class_inferred
  }
  if ("mhc_allele_std" %in% names(data_splits$test$data)) {
    pred_df$mhc_allele <- data_splits$test$data$mhc_allele_std
  }

  # Summary
  cat("\n", strrep("=", 50), "\nEVALUATION RESULTS\n", strrep("=", 50), "\n")
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
    summarise(
      n = n(),
      accuracy = mean(correct, na.rm = TRUE),
      mean_confidence = mean(confidence, na.rm = TRUE),
      .groups = "drop"
    )

  message("\nPerformance by species:")
  for (i in seq_len(nrow(by_species))) {
    message("  ", by_species$source[i], ": ", round(by_species$accuracy[i] * 100, 1), "%")
  }
  by_species
}

#' Evaluate by pairing status
#' @param predictions Predictions tibble with is_paired column
#' @return Tibble with per-pairing metrics
#' @export
evaluate_by_pairing <- function(predictions) {

  if (!"is_paired" %in% names(predictions)) {
    return(NULL)
  }

  predictions %>%
    mutate(status = ifelse(is_paired, "Paired (TRA+TRB)", "Unpaired (TRB only)")) %>%
    group_by(status) %>%
    summarise(
      n = n(),
      accuracy = mean(correct, na.rm = TRUE),
      mean_confidence = mean(confidence, na.rm = TRUE),
      .groups = "drop"
    )
}

#' Evaluate by MHC class
#' @param predictions Predictions tibble with mhc_class column
#' @return Tibble with per-MHC-class metrics
#' @export
evaluate_by_mhc_class <- function(predictions) {

  if (!"mhc_class" %in% names(predictions)) {
    message("No mhc_class column in predictions")
    return(NULL)
  }

  by_mhc <- predictions %>%
    mutate(mhc_class = ifelse(is.na(mhc_class), "Unknown", mhc_class)) %>%
    group_by(mhc_class) %>%
    summarise(
      n = n(),
      accuracy = mean(correct, na.rm = TRUE),
      mean_confidence = mean(confidence, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(n))

  message("\nPerformance by MHC class:")
  for (i in seq_len(nrow(by_mhc))) {
    message(sprintf("  %s: %.1f%% (n=%s)",
                    by_mhc$mhc_class[i],
                    by_mhc$accuracy[i] * 100,
                    format(by_mhc$n[i], big.mark = ",")))
  }

  by_mhc
}

# ===== Phase 2: Mouse Fine-tuning with Experience Replay =====

#' Create Phase 2 splits with experience replay
#'
#' @param data_splits Full data splits from prepare_combined_data()
#' @param replay_fraction Fraction of human data to replay
#' @param replay_stratified Stratify replay by epitope
#' @param seed Random seed
#' @return List with Phase 2 training splits
#' @export
create_phase2_splits <- function(data_splits, replay_fraction = 0.03,
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

  # Subset helper
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
      mhc_class_idx = split$mhc_class_idx[idx],
      mhc_allele_idx = split$mhc_allele_idx[idx],
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
    unique_epitope_mhc_class = data_splits$unique_epitope_mhc_class,
    unique_epitope_mhc_allele = data_splits$unique_epitope_mhc_allele,
    trb_vocab = data_splits$trb_vocab,
    tra_vocab = data_splits$tra_vocab,
    mhc_vocab = data_splits$mhc_vocab,
    replay_config = list(
      fraction = replay_fraction,
      n_replay = length(replay_idx),
      stratified = replay_stratified
    )
  )
}

# ===== Transfer Learning Pipeline =====

#' Run transfer learning pipeline
#'
#' Phase 1: Train on combined human + mouse data with MHC
#' Phase 2: Fine-tune on mouse with experience replay
#'
#' @param data_splits Pre-prepared data splits with MHC encoding
#' @param trb_vocab TRB V/J vocabulary
#' @param tra_vocab TRA V/J vocabulary
#' @param mhc_vocab MHC vocabulary
#' @param config Configuration list (see default_config())
#' @param fine_tune_mouse If TRUE, perform Phase 2
#' @return Training results
#' @export
run_transfer_learning_pipeline <- function(data_splits, trb_vocab, tra_vocab,
                                           mhc_vocab, config = NULL,
                                           fine_tune_mouse = TRUE) {

  config <- modifyList(default_config(), config %||% list())
  if (!dir.exists(config$output_dir)) dir.create(config$output_dir, recursive = TRUE)

  message("\n", strrep("=", 70))
  message("TRANSFER LEARNING PIPELINE (V9.1: Dual Chain + MHC)")
  message(strrep("=", 70))

  print_paired_data_summary(data_splits)

  # === Phase 1 ===
  message("\n", strrep("-", 70))
  message("PHASE 1: Combined Training")
  message(strrep("-", 70))

  model <- create_tcr_epitope_model_v91(
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
    mhc_class_vocab_size = as.integer(mhc_vocab$class$size),
    mhc_allele_vocab_size = as.integer(mhc_vocab$allele$size),
    mhc_class_embed_dim = config$mhc_class_embed_dim,
    mhc_allele_embed_dim = config$mhc_allele_embed_dim,
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
    ewc_lambda = 0.0,
    mhc_class_unk_idx = config$mhc_class_unk_idx,
    mhc_allele_unk_idx = config$mhc_allele_unk_idx
  )

  p1_eval <- evaluate_model(p1_result$trainer, data_splits, data_splits$idx_to_epitope)
  p1_species <- evaluate_by_species(p1_eval$predictions, data_splits$test$data)
  p1_mhc <- evaluate_by_mhc_class(p1_eval$predictions)

  results <- list(
    phase1 = list(
      model = p1_result$model,
      trainer = p1_result$trainer,
      history = p1_result$history,
      evaluation = p1_eval,
      by_species = p1_species,
      by_mhc = p1_mhc
    ),
    data_splits = data_splits,
    config = config,
    mhc_vocab = mhc_vocab
  )

  # === Phase 2 ===
  if (fine_tune_mouse) {
    message("\n", strrep("-", 70))
    message("PHASE 2: Mouse Fine-tuning (", config$replay_fraction * 100, "% replay)")
    message(strrep("-", 70))

    p2_splits <- create_phase2_splits(
      data_splits,
      config$replay_fraction,
      config$replay_stratified
    )

    p2_trainer <- py$TCRTrainer(
      model = p1_result$model,
      device = config$device,
      loss_type = config$loss_type,
      focal_gamma = config$focal_gamma,
      label_smoothing = config$label_smoothing,
      ewc_lambda = config$ewc_lambda,
      mhc_class_unk_idx = as.integer(config$mhc_class_unk_idx),
      mhc_allele_unk_idx = as.integer(config$mhc_allele_unk_idx)
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
      mhc_class_idx = p2_splits$train$mhc_class_idx,
      mhc_allele_idx = p2_splits$train$mhc_allele_idx,
      batch_size = as.integer(config$phase2_batch_size),
      shuffle = TRUE,
      weighted_sampling = TRUE
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
      mhc_class_idx = p2_splits$validation$mhc_class_idx,
      mhc_allele_idx = p2_splits$validation$mhc_allele_idx,
      batch_size = as.integer(config$phase2_batch_size),
      shuffle = FALSE,
      weighted_sampling = FALSE
    )

    p2_history <- p2_trainer$fit(
      p2_train_loader,
      p2_val_loader,
      p2_splits$unique_epitope_idx,
      unique_mhc_class = p2_splits$unique_epitope_mhc_class,
      unique_mhc_allele = p2_splits$unique_epitope_mhc_allele,
      epochs = as.integer(config$phase2_epochs),
      lr = config$phase2_learning_rate,
      patience = as.integer(config$phase2_patience)
    )

    p2_eval_mouse <- evaluate_model(p2_trainer, p2_splits, data_splits$idx_to_epitope)
    p2_eval_full <- evaluate_model(p2_trainer, data_splits, data_splits$idx_to_epitope)
    p2_species <- evaluate_by_species(p2_eval_full$predictions, data_splits$test$data)
    p2_mhc <- evaluate_by_mhc_class(p2_eval_full$predictions)

    # Forgetting analysis
    p1_human <- p1_species$accuracy[p1_species$source == "human"]
    p2_human <- p2_species$accuracy[p2_species$source == "human"]
    p1_mouse <- p1_species$accuracy[p1_species$source == "mouse"]
    p2_mouse <- p2_species$accuracy[p2_species$source == "mouse"]

    results$phase2 <- list(
      model = p1_result$model,
      trainer = p2_trainer,
      history = p2_history,
      evaluation_mouse = p2_eval_mouse,
      evaluation_full = p2_eval_full,
      by_species = p2_species,
      by_mhc = p2_mhc,
      phase2_splits = p2_splits,
      forgetting = list(
        phase1_human = p1_human,
        phase2_human = p2_human,
        phase1_mouse = p1_mouse,
        phase2_mouse = p2_mouse,
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
