# Calibration for V5/V6 models - DEPRECATED

library(reticulate)

py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def calibrate_v5(model, val_cdr3, val_labels, epitope_idx, device='cpu'):
    '''Calibrate V5 model (CDR3 only).'''
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    cdr3_t = torch.tensor(np.array(val_cdr3, copy=True), dtype=torch.long).to(device)
    labels_t = torch.tensor(np.array(val_labels, copy=True), dtype=torch.long).to(device)
    epi_t = torch.tensor(np.array(epitope_idx, copy=True), dtype=torch.long).to(device)

    n = cdr3_t.shape[0]
    all_logits = []

    with torch.no_grad():
        epi_emb = model.encode_epitope(epi_t)
        for i in range(0, n, 256):
            j = min(i + 256, n)
            cdr3_emb = model.encode_cdr3(cdr3_t[i:j])
            logits = torch.mm(cdr3_emb, epi_emb.t()) / model.temperature.clamp(min=0.01)
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels_np = labels_t.cpu().numpy()

    # Learn temperature
    log_temp = nn.Parameter(torch.zeros(1))
    opt = optim.LBFGS([log_temp], lr=0.01, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        t = log_temp.exp().clamp(0.01, 10.0)
        loss = criterion(torch.tensor(logits) / t, torch.tensor(labels_np))
        loss.backward()
        return loss

    opt.step(closure)
    temp = float(log_temp.exp().clamp(0.01, 10.0).item())

    # Metrics
    def ece(probs, labels, n_bins=15):
        confs = np.max(probs, 1)
        preds = np.argmax(probs, 1)
        accs = (preds == labels).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        e = 0.0
        for i in range(n_bins):
            mask = (confs > bins[i]) & (confs <= bins[i+1])
            if np.sum(mask) > 0:
                e += (np.sum(mask) / len(labels)) * np.abs(np.mean(accs[mask]) - np.mean(confs[mask]))
        return e

    probs_before = F.softmax(torch.tensor(logits), dim=1).numpy()
    probs_after = F.softmax(torch.tensor(logits / temp), dim=1).numpy()

    return {
        'optimal_temperature': temp,
        'ece_before': ece(probs_before, labels_np),
        'ece_after': ece(probs_after, labels_np),
        'probs_before': probs_before,
        'probs_after': probs_after,
        'labels': labels_np
    }

def calibrate_v6(model, val_cdr3, val_labels, v_gene, j_gene, epitope_idx, device='cpu'):
    '''Calibrate V6 model (CDR3 + V/J genes).'''
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    cdr3_t = torch.tensor(np.array(val_cdr3, copy=True), dtype=torch.long).to(device)
    labels_t = torch.tensor(np.array(val_labels, copy=True), dtype=torch.long).to(device)
    v_t = torch.tensor(np.array(v_gene, copy=True), dtype=torch.long).to(device)
    j_t = torch.tensor(np.array(j_gene, copy=True), dtype=torch.long).to(device)
    epi_t = torch.tensor(np.array(epitope_idx, copy=True), dtype=torch.long).to(device)

    n = cdr3_t.shape[0]
    all_logits = []

    with torch.no_grad():
        epi_emb = model.encode_epitope(epi_t)
        for i in range(0, n, 256):
            j = min(i + 256, n)
            cdr3_emb = model.encode_cdr3(cdr3_t[i:j], v_gene=v_t[i:j], j_gene=j_t[i:j])
            logits = torch.mm(cdr3_emb, epi_emb.t()) / model.temperature.clamp(min=0.01)
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels_np = labels_t.cpu().numpy()

    # Learn temperature (same as V5)
    log_temp = nn.Parameter(torch.zeros(1))
    opt = optim.LBFGS([log_temp], lr=0.01, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        t = log_temp.exp().clamp(0.01, 10.0)
        loss = criterion(torch.tensor(logits) / t, torch.tensor(labels_np))
        loss.backward()
        return loss

    opt.step(closure)
    temp = float(log_temp.exp().clamp(0.01, 10.0).item())

    return {'optimal_temperature': temp, 'labels': labels_np}
")

#' Calibrate V5 model - DEPRECATED
#' @export
calibrate_model_v5 <- function(model, data_splits, device = "cpu") {

  .Deprecated("calibrate_model", msg = "Use calibrate_model() for V7")

  unique_epi_idx <- sequences_to_indices(data_splits$unique_epitopes, 30L)

  py$calibrate_v5(
    model = model,
    val_cdr3 = data_splits$validation$cdr3_idx,
    val_labels = data_splits$validation$labels,
    epitope_idx = unique_epi_idx,
    device = device
  )
}

#' Calibrate V6 model - DEPRECATED
#' @export
calibrate_model_v6 <- function(model, data_splits, device = "cpu") {

  .Deprecated("calibrate_model", msg = "Use calibrate_model() for V7")

  unique_epi_idx <- sequences_to_indices(data_splits$unique_epitopes, 30L)

  py$calibrate_v6(
    model = model,
    val_cdr3 = data_splits$validation$cdr3_idx,
    val_labels = data_splits$validation$labels,
    v_gene = data_splits$validation$v_idx,
    j_gene = data_splits$validation$j_idx,
    epitope_idx = unique_epi_idx,
    device = device
  )
}
