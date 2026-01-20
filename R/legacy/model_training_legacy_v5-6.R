# Legacy TCR-Epitope model training (V5, V6) - DEPRECATED
# Preserved for backward compatibility with saved models
# Current development uses V7 (dual-chain) in model_training.R

library(reticulate)
library(dplyr)
library(tidyr)
library(caret)

# ============================================================================
# Python Training Infrastructure (V5/V6)
# ============================================================================

py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Subset
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, Tuple

# --- Focal Loss ---
class FocalLoss(nn.Module):
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
        print(f'Computing Fisher Information ({num_samples} samples)...')
        self.model.eval()

        fisher = {n: torch.zeros_like(p, device=self.device)
                  for n, p in self.model.named_parameters() if p.requires_grad}

        unique_epi = torch.tensor(unique_epitope_indices, dtype=torch.long, device=self.device)
        dataset = dataloader.dataset
        n_total = len(dataset)

        if stratified_labels is not None:
            sample_idx = self._stratified_sample(stratified_labels, num_samples, n_total)
        else:
            sample_idx = np.random.choice(n_total, min(num_samples, n_total), replace=False)

        subset_loader = DataLoader(Subset(dataset, sample_idx), batch_size=64, shuffle=False)
        n_proc = 0

        for batch in subset_loader:
            # V5/V6 batch structure: 4 or 6 tensors
            tensors = [x.to(self.device) for x in batch]
            cdr3, epitope, labels, weights = tensors[:4]

            self.model.zero_grad()
            cdr3_emb = self.model.encode_cdr3(cdr3)
            epi_emb = self.model.encode_epitope(unique_epi)
            sim = torch.mm(cdr3_emb, epi_emb.t()) / self.model.temperature.clamp(min=0.01)

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
        print('  Stored optimal parameters')

    def compute_ewc_loss(self) -> torch.Tensor:
        if self.fisher is None or self.optimal_params is None:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return loss

# --- V5 Trainer (CDR3 only) ---
class TCRTrainerV5:
    '''Trainer for V5 model (TRB CDR3 only, no V/J genes).'''

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

        print(f'TCRTrainerV5: {loss_type} loss, gamma={focal_gamma}')
        if ewc_lambda > 0: print(f'  EWC enabled (lambda={ewc_lambda})')

    def _to_tensor(self, arr, dtype=torch.long):
        return torch.tensor(np.array(arr, copy=True), dtype=dtype)

    def create_dataloader(self, cdr3_idx, epitope_idx, labels, weights,
                          batch_size=32, shuffle=True, weighted_sampling=True):
        tensors = [
            self._to_tensor(cdr3_idx), self._to_tensor(epitope_idx),
            self._to_tensor(labels), self._to_tensor(weights, torch.float)
        ]
        dataset = TensorDataset(*tensors)

        if weighted_sampling and shuffle:
            sampler = WeightedRandomSampler(tensors[3].numpy().tolist(),
                                           len(tensors[3]), replacement=True)
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
            cdr3, epitope, labels, weights = [x.to(self.device) for x in batch]

            optimizer.zero_grad()
            cdr3_emb = self.model.encode_cdr3(cdr3)
            epi_emb = self.model.encode_epitope(unique_epi)
            sim = torch.mm(cdr3_emb, epi_emb.t()) / self.model.temperature.clamp(min=0.01)

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
                cdr3, epitope, labels, weights = [x.to(self.device) for x in batch]

                cdr3_emb = self.model.encode_cdr3(cdr3)
                sim = torch.mm(cdr3_emb, epi_emb.t()) / self.model.temperature.clamp(min=0.01)

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
                print(f'Epoch {epoch+1}: loss={train_loss:.4f}, val_acc={val_res[\"accuracy\"]:.4f}')

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

    def get_embeddings(self, cdr3_indices):
        self.model.eval()
        with torch.no_grad():
            emb = self.model.encode_cdr3(self._to_tensor(cdr3_indices).to(self.device))
        return emb.cpu().numpy()

    def get_epitope_embeddings(self, epitope_indices):
        self.model.eval()
        with torch.no_grad():
            emb = self.model.encode_epitope(self._to_tensor(epitope_indices).to(self.device))
        return emb.cpu().numpy()

# --- V6 Trainer (with V/J genes) ---
class TCRTrainerV6(TCRTrainerV5):
    '''Trainer for V6 model (TRB CDR3 + V/J genes).'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_vj = hasattr(self.model, 'use_vj_genes') and self.model.use_vj_genes
        if self.use_vj: print('  V/J gene integration enabled')

    def create_dataloader(self, cdr3_idx, epitope_idx, labels, weights,
                          v_gene_idx=None, j_gene_idx=None,
                          batch_size=32, shuffle=True, weighted_sampling=True):
        tensors = [
            self._to_tensor(cdr3_idx), self._to_tensor(epitope_idx),
            self._to_tensor(labels), self._to_tensor(weights, torch.float)
        ]

        if self.use_vj and v_gene_idx is not None and j_gene_idx is not None:
            tensors.extend([self._to_tensor(v_gene_idx), self._to_tensor(j_gene_idx)])
        else:
            dummy = torch.zeros(len(labels), dtype=torch.long)
            tensors.extend([dummy, dummy])

        dataset = TensorDataset(*tensors)

        if weighted_sampling and shuffle:
            sampler = WeightedRandomSampler(tensors[3].numpy().tolist(),
                                           len(tensors[3]), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, dataloader, optimizer, unique_epitope_idx):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        unique_epi = self._to_tensor(unique_epitope_idx).to(self.device)

        for batch in dataloader:
            cdr3, epitope, labels, weights, v_gene, j_gene = [x.to(self.device) for x in batch]

            optimizer.zero_grad()
            if self.use_vj:
                cdr3_emb = self.model.encode_cdr3(cdr3, v_gene=v_gene, j_gene=j_gene)
            else:
                cdr3_emb = self.model.encode_cdr3(cdr3)

            epi_emb = self.model.encode_epitope(unique_epi)
            sim = torch.mm(cdr3_emb, epi_emb.t()) / self.model.temperature.clamp(min=0.01)

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
                cdr3, epitope, labels, weights, v_gene, j_gene = [x.to(self.device) for x in batch]

                if self.use_vj:
                    cdr3_emb = self.model.encode_cdr3(cdr3, v_gene=v_gene, j_gene=j_gene)
                else:
                    cdr3_emb = self.model.encode_cdr3(cdr3)

                sim = torch.mm(cdr3_emb, epi_emb.t()) / self.model.temperature.clamp(min=0.01)

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

    def get_embeddings(self, cdr3_indices, v_gene_idx=None, j_gene_idx=None):
        self.model.eval()
        with torch.no_grad():
            cdr3 = self._to_tensor(cdr3_indices).to(self.device)
            if self.use_vj and v_gene_idx is not None and j_gene_idx is not None:
                v = self._to_tensor(v_gene_idx).to(self.device)
                j = self._to_tensor(j_gene_idx).to(self.device)
                emb = self.model.encode_cdr3(cdr3, v_gene=v, j_gene=j)
            else:
                emb = self.model.encode_cdr3(cdr3)
        return emb.cpu().numpy()

# --- Encoder Freezing Utilities ---
def freeze_module(module):
    if module is None: return 0
    count = 0
    for p in module.parameters():
        p.requires_grad = False
        count += p.numel()
    return count

def unfreeze_module(module):
    if module is None: return 0
    count = 0
    for p in module.parameters():
        p.requires_grad = True
        count += p.numel()
    return count

def freeze_encoder_light(model):
    '''Light freeze: Conv layers only.'''
    frozen = 0
    frozen += freeze_module(model.cdr3_encoder.conv1)
    frozen += freeze_module(model.cdr3_encoder.conv2)
    frozen += freeze_module(model.cdr3_encoder.conv3)
    frozen += freeze_module(model.epitope_encoder.conv1)
    frozen += freeze_module(model.epitope_encoder.conv2)
    return frozen

def freeze_encoder_medium(model):
    '''Medium freeze: Conv + BatchNorm + Embeddings.'''
    frozen = freeze_encoder_light(model)
    frozen += freeze_module(model.cdr3_encoder.bn1)
    frozen += freeze_module(model.cdr3_encoder.bn2)
    frozen += freeze_module(model.cdr3_encoder.bn3)
    frozen += freeze_module(model.epitope_encoder.bn1)
    frozen += freeze_module(model.epitope_encoder.bn2)
    if model.cdr3_encoder.embedding: frozen += freeze_module(model.cdr3_encoder.embedding)
    if model.epitope_encoder.embedding: frozen += freeze_module(model.epitope_encoder.embedding)
    return frozen

def freeze_encoder_heavy(model):
    '''Heavy freeze: Conv + BatchNorm + Embeddings + Attention.'''
    frozen = freeze_encoder_medium(model)
    frozen += freeze_module(model.cdr3_encoder.attention)
    return frozen

def unfreeze_all(model):
    for p in model.parameters(): p.requires_grad = True
    return sum(p.numel() for p in model.parameters())

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}

def compute_ewc_sample_count(n_train, min_samples=5000, max_samples=10000, fraction=0.05):
    return min(max_samples, max(min_samples, int(fraction * n_train)))
")

# ============================================================================
# R Wrapper Functions (V5)
# ============================================================================

#' Train TCR-Epitope model V5 (CDR3 only) - DEPRECATED
#' @export
train_tcr_epitope_model_v5 <- function(model, data_splits,
                                       epochs = 100, batch_size = 32,
                                       learning_rate = 1e-3, patience = 15,
                                       device = "cpu", loss_type = "focal",
                                       focal_gamma = 2.0, label_smoothing = 0.0,
                                       ewc_lambda = 0.0, blosum_lambda = 0.0) {

  .Deprecated("train_tcr_epitope_model",
              msg = "V5 is deprecated. Use V7 (dual-chain) from model_training.R")

  if (device == "cuda" && !py$torch$cuda$is_available()) {
    message("CUDA not available, using CPU")
    device <- "cpu"
  }

  trainer <- py$TCRTrainerV5(
    model = model, device = device, loss_type = loss_type,
    focal_gamma = focal_gamma, label_smoothing = label_smoothing,
    ewc_lambda = ewc_lambda, blosum_lambda = blosum_lambda
  )

  unique_epitope_idx <- sequences_to_indices(data_splits$unique_epitopes, 30L)

  train_loader <- trainer$create_dataloader(
    cdr3_idx = data_splits$train$cdr3_idx,
    epitope_idx = data_splits$train$epitope_idx,
    labels = data_splits$train$labels,
    weights = data_splits$train$weights,
    batch_size = as.integer(batch_size),
    shuffle = TRUE, weighted_sampling = TRUE
  )

  val_loader <- trainer$create_dataloader(
    cdr3_idx = data_splits$validation$cdr3_idx,
    epitope_idx = data_splits$validation$epitope_idx,
    labels = data_splits$validation$labels,
    weights = data_splits$validation$weights,
    batch_size = as.integer(batch_size),
    shuffle = FALSE, weighted_sampling = FALSE
  )

  history <- trainer$fit(
    train_loader = train_loader, val_loader = val_loader,
    unique_epitope_indices = unique_epitope_idx,
    epochs = as.integer(epochs), lr = learning_rate,
    patience = as.integer(patience)
  )

  list(model = model, trainer = trainer, history = history,
       unique_epitope_idx = unique_epitope_idx, train_loader = train_loader)
}

#' Train TCR-Epitope model V6 (with V/J genes) - DEPRECATED
#' @export
train_tcr_epitope_model_v6 <- function(model, data_splits,
                                       epochs = 100, batch_size = 32,
                                       learning_rate = 1e-3, patience = 15,
                                       device = "cpu", loss_type = "focal",
                                       focal_gamma = 2.0, label_smoothing = 0.0,
                                       ewc_lambda = 0.0, blosum_lambda = 0.0) {

  .Deprecated("train_tcr_epitope_model",
              msg = "V6 is deprecated. Use V7 (dual-chain) from model_training.R")

  if (device == "cuda" && !py$torch$cuda$is_available()) {
    message("CUDA not available, using CPU")
    device <- "cpu"
  }

  trainer <- py$TCRTrainerV6(
    model = model, device = device, loss_type = loss_type,
    focal_gamma = focal_gamma, label_smoothing = label_smoothing,
    ewc_lambda = ewc_lambda, blosum_lambda = blosum_lambda
  )

  unique_epitope_idx <- sequences_to_indices(data_splits$unique_epitopes, 30L)
  has_vj <- "v_idx" %in% names(data_splits$train)

  train_loader <- trainer$create_dataloader(
    cdr3_idx = data_splits$train$cdr3_idx,
    epitope_idx = data_splits$train$epitope_idx,
    labels = data_splits$train$labels,
    weights = data_splits$train$weights,
    v_gene_idx = if (has_vj) data_splits$train$v_idx else NULL,
    j_gene_idx = if (has_vj) data_splits$train$j_idx else NULL,
    batch_size = as.integer(batch_size),
    shuffle = TRUE, weighted_sampling = TRUE
  )

  val_loader <- trainer$create_dataloader(
    cdr3_idx = data_splits$validation$cdr3_idx,
    epitope_idx = data_splits$validation$epitope_idx,
    labels = data_splits$validation$labels,
    weights = data_splits$validation$weights,
    v_gene_idx = if (has_vj) data_splits$validation$v_idx else NULL,
    j_gene_idx = if (has_vj) data_splits$validation$j_idx else NULL,
    batch_size = as.integer(batch_size),
    shuffle = FALSE, weighted_sampling = FALSE
  )

  history <- trainer$fit(
    train_loader = train_loader, val_loader = val_loader,
    unique_epitope_indices = unique_epitope_idx,
    epochs = as.integer(epochs), lr = learning_rate,
    patience = as.integer(patience)
  )

  list(model = model, trainer = trainer, history = history,
       unique_epitope_idx = unique_epitope_idx, train_loader = train_loader)
}

# ============================================================================
# Configuration
# ============================================================================

#' Default V5 configuration - DEPRECATED
#' @export
default_config_v5 <- function() {
  list(
    vocab_size = 22L, token_embedding_dim = 128L, hidden_dim = 256L,
    output_dim = 256L, dropout = 0.3,
    use_atchley_init = TRUE, use_blosum_reg = FALSE,
    loss_type = "focal", focal_gamma = 2.0, label_smoothing = 0.0,
    ewc_lambda = 1000, blosum_lambda = 0.0,
    phase1_epochs = 500, phase1_batch_size = 8192,
    phase1_learning_rate = 4e-3, phase1_patience = 100,
    phase2_epochs = 500, phase2_batch_size = 1024,
    phase2_learning_rate = 1e-4, phase2_patience = 10,
    phase2_freeze_encoder = FALSE,
    replay_fraction = 0.1, replay_stratified = TRUE,
    replay_in_validation = TRUE, replay_species = "human",
    cdr3_max_len = 25L, epitope_max_len = 30L,
    test_fraction = 0.15, validation_fraction = 0.15,
    device = "cpu", output_dir = "results_transfer_v5"
  )
}

#' Default V6 configuration - DEPRECATED
#' @export
default_config_v6 <- function() {
  config <- default_config_v5()
  config$output_dir <- "results_transfer_v6"
  config$device <- "cuda"
  config$ewc_lambda <- 0
  config
}

# ============================================================================
# Data Preparation (V5/V6)
# ============================================================================

#' Prepare transfer learning data splits (V5/V6)
#' @export
prepare_transfer_learning_data <- function(combined_data,
                                           test_fraction = 0.15,
                                           validation_fraction = 0.15,
                                           stratify_by_species = TRUE,
                                           seed = 42) {
  set.seed(seed)
  data <- combined_data$data

  # Determine species column
  species_col <- if ("tcr_species" %in% names(data)) "tcr_species"
  else if ("source" %in% names(data)) "source"
  else stop("No species column found")

  # Stratification variable
  if (stratify_by_species) {
    data <- data %>%
      mutate(strat_var = paste(.data[[species_col]],
                               ifelse(score >= 2, "high", "low"), sep = "_"))
  } else {
    data <- data %>% mutate(strat_var = epitope)
  }

  # Split by epitope sample count
  epitope_counts <- data %>% count(epitope) %>% arrange(desc(n))
  splittable <- epitope_counts %>% filter(n >= 3) %>% pull(epitope)

  data_split <- data %>% filter(epitope %in% splittable)
  data_train_only <- data %>% filter(!epitope %in% splittable)

  # Stratified splits
  test_idx <- createDataPartition(data_split$strat_var, p = test_fraction, list = FALSE)
  test_data <- data_split[test_idx, ]
  train_val <- data_split[-test_idx, ]

  val_idx <- createDataPartition(train_val$strat_var,
                                 p = validation_fraction / (1 - test_fraction), list = FALSE)
  val_data <- train_val[val_idx, ]
  train_data <- bind_rows(train_val[-val_idx, ], data_train_only)

  # Standardize species column
  if (species_col != "source") {
    train_data <- train_data %>% rename(db_source = source, source = !!sym(species_col))
    val_data <- val_data %>% rename(db_source = source, source = !!sym(species_col))
    test_data <- test_data %>% rename(db_source = source, source = !!sym(species_col))
  }

  message("Splits: train=", nrow(train_data), ", val=", nrow(val_data), ", test=", nrow(test_data))

  # Encode sequences
  cdr3_max <- 25L; epi_max <- 30L

  list(
    train = list(
      data = train_data,
      cdr3_idx = sequences_to_indices(train_data$cdr3, cdr3_max),
      epitope_idx = sequences_to_indices(train_data$epitope, epi_max),
      weights = as.numeric(train_data$sample_weight),
      labels = as.integer(train_data$epitope_idx),
      species = train_data$source
    ),
    validation = list(
      data = val_data,
      cdr3_idx = sequences_to_indices(val_data$cdr3, cdr3_max),
      epitope_idx = sequences_to_indices(val_data$epitope, epi_max),
      weights = as.numeric(val_data$sample_weight),
      labels = as.integer(val_data$epitope_idx),
      species = val_data$source
    ),
    test = list(
      data = test_data,
      cdr3_idx = sequences_to_indices(test_data$cdr3, cdr3_max),
      epitope_idx = sequences_to_indices(test_data$epitope, epi_max),
      weights = as.numeric(test_data$sample_weight),
      labels = as.integer(test_data$epitope_idx),
      species = test_data$source
    ),
    epitope_to_idx = combined_data$epitope_to_idx,
    idx_to_epitope = combined_data$idx_to_epitope,
    unique_epitopes = combined_data$unique_epitopes
  )
}

#' Prepare transfer learning data V6 (with V/J encoding)
#' @export
prepare_transfer_learning_data_v6 <- function(combined_data, vj_vocab,
                                              test_fraction = 0.15,
                                              validation_fraction = 0.15,
                                              stratify_by_species = TRUE,
                                              seed = 42) {

  data_splits <- prepare_transfer_learning_data(
    combined_data, test_fraction, validation_fraction, stratify_by_species, seed
  )

  message("Encoding V/J genes...")

  # Encode V/J for each split (requires encode_vj_for_dataset from vj_gene_encoding.R)
  for (split_name in c("train", "validation", "test")) {
    split_data <- data_splits[[split_name]]$data
    if ("v.segm" %in% names(split_data) || "v_gene" %in% names(split_data)) {
      encoded <- encode_vj_for_dataset(split_data, vj_vocab)
      data_splits[[split_name]]$v_idx <- as.integer(encoded$v_idx)
      data_splits[[split_name]]$j_idx <- as.integer(encoded$j_idx)
    }
  }

  data_splits$vj_vocab <- vj_vocab
  data_splits
}

# ============================================================================
# Encoder Freezing (R Wrappers)
# ============================================================================

#' Freeze encoder layers
#' @param model TCREpitopeModelV5/V6 instance
#' @param level Freezing level: "light", "medium", or "heavy"
#' @export
freeze_encoder_layers <- function(model, level = "heavy") {

  if (!level %in% c("light", "medium", "heavy")) {
    stop("Invalid level. Use: light, medium, heavy")
  }

  frozen <- switch(level,
                   "light" = py$freeze_encoder_light(model),
                   "medium" = py$freeze_encoder_medium(model),
                   "heavy" = py$freeze_encoder_heavy(model)
  )

  counts <- py$count_parameters(model)
  message(sprintf("Frozen %s params (%s level). Trainable: %s/%s (%.1f%%)",
                  format(frozen, big.mark = ","), level,
                  format(counts$trainable, big.mark = ","),
                  format(counts$total, big.mark = ","),
                  100 * counts$trainable / counts$total))

  invisible(list(level = level, frozen = frozen, counts = counts))
}

#' Unfreeze all layers
#' @export
unfreeze_all_layers <- function(model) {
  py$unfreeze_all(model)
  message("All layers unfrozen")
}

# ============================================================================
# Experience Replay (V5/V6)
# ============================================================================

#' Create replay buffer from Phase 1 data
#' @export
create_replay_buffer <- function(train_data, replay_fraction = 0.1,
                                 target_species = "human",
                                 stratify_by = "epitope", seed = 42) {
  set.seed(seed)

  source_data <- if (target_species == "all") {
    train_data %>% filter(source != "mouse")
  } else {
    train_data %>% filter(source == target_species)
  }

  if (nrow(source_data) == 0) {
    warning("No samples for species: ", target_species)
    return(tibble())
  }

  target_size <- ceiling(nrow(source_data) * replay_fraction)

  if (!is.null(stratify_by) && stratify_by %in% names(source_data)) {
    # Stratified sampling
    strata <- source_data %>%
      group_by(!!sym(stratify_by)) %>%
      summarise(indices = list(row_number()), n = n(), .groups = "drop")

    per_stratum <- max(1, target_size %/% nrow(strata))

    replay_rows <- unlist(lapply(seq_len(nrow(strata)), function(i) {
      idx <- strata$indices[[i]]
      sample(idx, min(per_stratum, length(idx)))
    }))

    if (length(replay_rows) > target_size) {
      replay_rows <- sample(replay_rows, target_size)
    }

    replay_samples <- source_data[replay_rows, ]
  } else {
    replay_samples <- source_data %>% sample_n(target_size)
  }

  replay_samples %>% mutate(is_replay = TRUE)
}

#' Create Phase 2 splits with experience replay
#' @export
create_phase2_splits_with_replay <- function(data_splits, replay_fraction = 0.1,
                                             replay_stratified = TRUE,
                                             replay_in_validation = TRUE,
                                             replay_species = "human",
                                             cdr3_max_len = 25L,
                                             epitope_max_len = 30L, seed = 42) {

  # Mouse-only data
  mouse_train <- data_splits$train$data %>% filter(source == "mouse") %>% mutate(is_replay = FALSE)
  mouse_val <- data_splits$validation$data %>% filter(source == "mouse") %>% mutate(is_replay = FALSE)
  mouse_test <- data_splits$test$data %>% filter(source == "mouse") %>% mutate(is_replay = FALSE)

  message("Mouse: train=", nrow(mouse_train), ", val=", nrow(mouse_val), ", test=", nrow(mouse_test))

  # Create replay buffer
  if (replay_fraction > 0) {
    replay_train <- create_replay_buffer(
      data_splits$train$data, replay_fraction, replay_species,
      if (replay_stratified) "epitope" else NULL, seed
    )
    combined_train <- bind_rows(mouse_train, replay_train)

    if (replay_in_validation) {
      replay_val <- create_replay_buffer(
        data_splits$validation$data, replay_fraction, replay_species,
        if (replay_stratified) "epitope" else NULL, seed + 1
      )
      combined_val <- bind_rows(mouse_val, replay_val)
    } else {
      combined_val <- mouse_val
    }
  } else {
    combined_train <- mouse_train
    combined_val <- mouse_val
  }

  message("Phase 2 train: ", nrow(combined_train), " (", sum(!combined_train$is_replay),
          " mouse + ", sum(combined_train$is_replay), " replay)")

  list(
    train = list(
      data = combined_train,
      cdr3_idx = sequences_to_indices(combined_train$cdr3, cdr3_max_len),
      epitope_idx = sequences_to_indices(combined_train$epitope, epitope_max_len),
      weights = as.numeric(combined_train$sample_weight),
      labels = as.integer(combined_train$epitope_idx)
    ),
    validation = list(
      data = combined_val,
      cdr3_idx = sequences_to_indices(combined_val$cdr3, cdr3_max_len),
      epitope_idx = sequences_to_indices(combined_val$epitope, epitope_max_len),
      weights = as.numeric(combined_val$sample_weight),
      labels = as.integer(combined_val$epitope_idx)
    ),
    test = list(
      data = mouse_test,
      cdr3_idx = sequences_to_indices(mouse_test$cdr3, cdr3_max_len),
      epitope_idx = sequences_to_indices(mouse_test$epitope, epitope_max_len),
      weights = as.numeric(mouse_test$sample_weight),
      labels = as.integer(mouse_test$epitope_idx)
    ),
    epitope_to_idx = data_splits$epitope_to_idx,
    idx_to_epitope = data_splits$idx_to_epitope,
    unique_epitopes = data_splits$unique_epitopes,
    replay_config = list(
      fraction = replay_fraction,
      n_train_replay = sum(combined_train$is_replay),
      n_val_replay = if (exists("combined_val")) sum(combined_val$is_replay) else 0
    )
  )
}

# ============================================================================
# EWC Utilities
# ============================================================================

#' Compute EWC state
#' @export
compute_ewc_state <- function(model, train_loader, unique_epitope_idx,
                              train_labels, n_train, device,
                              ewc_min_samples = 5000, ewc_max_samples = 10000,
                              ewc_fraction = 0.05) {

  n_fisher <- py$compute_ewc_sample_count(
    as.integer(n_train), as.integer(ewc_min_samples),
    as.integer(ewc_max_samples), ewc_fraction
  )

  message("Computing EWC state (", n_fisher, " samples)...")

  ewc <- py$EWCRegularizer(model, device)
  ewc$compute_fisher_information(train_loader, unique_epitope_idx,
                                 as.integer(n_fisher), as.integer(train_labels))
  ewc$store_optimal_params()

  list(fisher = ewc$fisher, optimal_params = ewc$optimal_params, n_samples = n_fisher)
}

#' Setup EWC regularization
#' @export
setup_ewc_regularization <- function(trainer, train_loader, unique_epitope_idx,
                                     train_labels, n_train,
                                     ewc_min_samples = 5000, ewc_max_samples = 10000,
                                     ewc_fraction = 0.05) {

  n_fisher <- py$compute_ewc_sample_count(
    as.integer(n_train), as.integer(ewc_min_samples),
    as.integer(ewc_max_samples), ewc_fraction
  )

  message("Setting up EWC (", n_fisher, " samples)...")
  trainer$setup_ewc(train_loader, unique_epitope_idx,
                    as.integer(n_fisher), as.integer(train_labels))
}

# ============================================================================
# Evaluation Utilities
# ============================================================================

#' Evaluate by species
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

  predictions %>%
    group_by(source) %>%
    summarise(n = n(), accuracy = mean(correct, na.rm = TRUE),
              mean_confidence = mean(confidence, na.rm = TRUE), .groups = "drop")
}

# ============================================================================
# Transfer Learning Pipeline (V5)
# ============================================================================

#' Run transfer learning pipeline V5 - DEPRECATED
#' @export
run_transfer_learning_pipeline_v5 <- function(combined_data, config = NULL,
                                              fine_tune_mouse = TRUE) {

  .Deprecated("run_transfer_learning_pipeline",
              msg = "V5 pipeline deprecated. Use V7 from model_training.R")

  config <- modifyList(default_config_v5(), config %||% list())
  if (!dir.exists(config$output_dir)) dir.create(config$output_dir, recursive = TRUE)

  message("\n", strrep("=", 70))
  message("TRANSFER LEARNING PIPELINE V5 (LEGACY)")
  message(strrep("=", 70))

  # Data prep
  data_splits <- prepare_transfer_learning_data(
    combined_data, config$test_fraction, config$validation_fraction
  )

  # Create model (requires model_architecture_legacy_v5-6.R)
  model <- create_tcr_epitope_model_v5(
    vocab_size = config$vocab_size,
    embed_dim = config$token_embedding_dim,
    hidden_dim = config$hidden_dim,
    output_dim = config$output_dim,
    dropout = config$dropout,
    use_atchley_init = config$use_atchley_init,
    use_blosum_reg = config$use_blosum_reg
  )

  # Phase 1
  message("\nPhase 1: Combined training...")
  p1_result <- train_tcr_epitope_model_v5(
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

  results <- list(
    phase1 = list(model = p1_result$model, trainer = p1_result$trainer,
                  history = p1_result$history),
    data_splits = data_splits, config = config
  )

  # Phase 2
  if (fine_tune_mouse) {
    message("\nPhase 2: Mouse fine-tuning...")

    # EWC state
    if (config$ewc_lambda > 0) {
      ewc_state <- compute_ewc_state(
        p1_result$model, p1_result$train_loader, p1_result$unique_epitope_idx,
        data_splits$train$labels, nrow(data_splits$train$data), config$device
      )
    }

    # Phase 2 splits
    p2_splits <- create_phase2_splits_with_replay(
      data_splits, config$replay_fraction, config$replay_stratified,
      config$replay_in_validation, config$replay_species
    )

    # Freeze if requested
    if (!isFALSE(config$phase2_freeze_encoder)) {
      level <- if (isTRUE(config$phase2_freeze_encoder)) "heavy" else config$phase2_freeze_encoder
      freeze_encoder_layers(p1_result$model, level)
    }

    # Phase 2 trainer
    p2_trainer <- py$TCRTrainerV5(
      model = p1_result$model, device = config$device,
      loss_type = config$loss_type, focal_gamma = config$focal_gamma,
      label_smoothing = config$label_smoothing,
      ewc_lambda = config$ewc_lambda, blosum_lambda = config$blosum_lambda
    )

    if (config$ewc_lambda > 0 && exists("ewc_state")) {
      p2_trainer$ewc$fisher <- ewc_state$fisher
      p2_trainer$ewc$optimal_params <- ewc_state$optimal_params
    }

    # Train
    p2_train_loader <- p2_trainer$create_dataloader(
      p2_splits$train$cdr3_idx, p2_splits$train$epitope_idx,
      p2_splits$train$labels, p2_splits$train$weights,
      as.integer(config$phase2_batch_size), TRUE, TRUE
    )

    p2_val_loader <- p2_trainer$create_dataloader(
      p2_splits$validation$cdr3_idx, p2_splits$validation$epitope_idx,
      p2_splits$validation$labels, p2_splits$validation$weights,
      as.integer(config$phase2_batch_size), FALSE, FALSE
    )

    p2_history <- p2_trainer$fit(
      p2_train_loader, p2_val_loader, p1_result$unique_epitope_idx,
      as.integer(config$phase2_epochs), config$phase2_learning_rate,
      patience = as.integer(config$phase2_patience)
    )

    results$phase2 <- list(
      model = p1_result$model, trainer = p2_trainer, history = p2_history,
      phase2_splits = p2_splits
    )
  }

  message("\nPipeline complete.")
  results
}
