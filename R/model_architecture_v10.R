# TCR-Epitope Model Architecture V10: ESM-2 Integration
#
# V10 extends V9.1 with ESM-2 protein language model embeddings:
#   - CDR3α, CDR3β, and epitope sequences use pre-computed ESM-2 embeddings
#   - V/J genes and MHC retain learned embeddings
#   - ESM embeddings (640-dim) projected to model dimension (128-dim)
#   - Frozen ESM embeddings stored as model buffers for self-contained inference
#
# Architecture:
#   TCREpitopeModelV10 (~2.5M trainable parameters)
#   ├── ESM-2 Embeddings (frozen buffers, not counted in params)
#   │   ├── CDR3α: (n_unique_alpha, 640)
#   │   ├── CDR3β: (n_unique_beta, 640)
#   │   └── Epitope: (n_unique_epitope, 640)
#   ├── TRA Encoder
#   │   ├── CDR3α: ESM(640) → Proj(128) → CNN → Attn → 128-dim
#   │   ├── Vα: Learned embedding → 32-dim
#   │   └── Jα: Learned embedding → 16-dim
#   ├── TRB Encoder
#   │   ├── CDR3β: ESM(640) → Proj(128) → CNN → Attn → 128-dim
#   │   ├── Vβ: Learned embedding → 32-dim
#   │   └── Jβ: Learned embedding → 16-dim
#   ├── Chain Fusion → 768-dim → Projection → 256-dim TCR embedding
#   ├── Epitope-MHC Encoder
#   │   ├── Epitope: ESM(640) → Proj(128) → CNN → 256-dim
#   │   └── MHC: Class(8) + Allele(32) → Fusion → 256-dim
#   └── Similarity: Cosine(TCR, Epitope-MHC) → Softmax
#

library(reticulate)

# ===== Python Model Definition =====

py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple

# --- Constants ---
AA_VOCAB_SIZE = 22  # For V/J embedding compatibility

# --- ESM Projection Module ---
class ESMProjector(nn.Module):
    '''
    Projects ESM-2 embeddings to model dimension with optional bottleneck.

    ESM-2 t30_150M produces 640-dim embeddings.
    We project to embed_dim (128) for compatibility with CNN layers.
    '''

    def __init__(self, esm_dim: int = 640, output_dim: int = 128,
                 dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(esm_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )

        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()

    def forward(self, esm_embeddings: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            esm_embeddings: (batch, esm_dim) mean-pooled ESM embeddings
        Returns:
            (batch, output_dim) projected embeddings
        '''
        projected = self.projection(esm_embeddings)
        return self.layer_norm(projected)


# --- CDR3 Encoder with ESM (replaces token embedding + CNN) ---
class CDR3EncoderESM(nn.Module):
    '''
    CDR3 encoder using ESM-2 embeddings.

    Since we use mean-pooled ESM embeddings, we apply a lightweight
    MLP instead of CNN (no sequence dimension to convolve over).
    '''

    def __init__(self, esm_dim: int = 640, hidden_dim: int = 256,
                 output_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_dim = output_dim

    def forward(self, esm_embeddings: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            esm_embeddings: (batch, esm_dim) mean-pooled ESM embeddings
        Returns:
            (batch, output_dim) encoded CDR3 features
        '''
        return self.encoder(esm_embeddings)


# --- Dual Chain TCR Encoder V10 ---
class DualChainTCREncoderV10(nn.Module):
    '''
    Dual-chain TCR encoder with ESM-2 embeddings for CDR3 sequences.

    ESM-2 provides rich contextual embeddings for CDR3α and CDR3β.
    V/J genes use learned embeddings (germline context).
    '''

    def __init__(self, esm_dim: int = 640, hidden_dim: int = 256,
                 output_dim: int = 256, dropout: float = 0.3,
                 trb_v_size: int = 115, trb_j_size: int = 35,
                 tra_v_size: int = 187, tra_j_size: int = 78,
                 v_embed_dim: int = 32, j_embed_dim: int = 16,
                 fusion: str = 'concat'):
        super().__init__()

        self.fusion_type = fusion
        cdr3_output_dim = 128  # CDR3 encoder output dimension

        # CDR3 encoders (ESM-based)
        self.tra_cdr3_encoder = CDR3EncoderESM(esm_dim, hidden_dim, cdr3_output_dim, dropout)
        self.trb_cdr3_encoder = CDR3EncoderESM(esm_dim, hidden_dim, cdr3_output_dim, dropout)

        # V/J gene embeddings (learned)
        self.tra_v_emb = nn.Embedding(tra_v_size, v_embed_dim, padding_idx=0)
        self.tra_j_emb = nn.Embedding(tra_j_size, j_embed_dim, padding_idx=0)
        self.trb_v_emb = nn.Embedding(trb_v_size, v_embed_dim, padding_idx=0)
        self.trb_j_emb = nn.Embedding(trb_j_size, j_embed_dim, padding_idx=0)

        # Initialize V/J embeddings
        for emb in [self.tra_v_emb, self.tra_j_emb, self.trb_v_emb, self.trb_j_emb]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.1)
            emb.weight.data[0] = 0.0  # PAD

        # Chain feature dimension: CDR3 + V + J
        chain_feat_dim = cdr3_output_dim + v_embed_dim + j_embed_dim  # 128 + 32 + 16 = 176

        # Fusion layer
        if fusion == 'concat':
            self.fusion_fc = nn.Sequential(
                nn.Linear(chain_feat_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        elif fusion == 'gated':
            self.fusion_gate = nn.Sequential(
                nn.Linear(chain_feat_dim * 2, 2),
                nn.Softmax(dim=-1)
            )
            self.fusion_fc = nn.Sequential(
                nn.Linear(chain_feat_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.chain_feat_dim = chain_feat_dim
        self.output_dim = output_dim

    def _encode_chain(self, cdr3_esm: torch.Tensor, v_gene: torch.Tensor,
                      j_gene: torch.Tensor, cdr3_encoder: nn.Module,
                      v_emb: nn.Module, j_emb: nn.Module) -> torch.Tensor:
        '''Encode a single chain (TRA or TRB).'''

        # CDR3 features from ESM
        cdr3_feat = cdr3_encoder(cdr3_esm)  # (batch, 128)

        # V/J embeddings
        v_feat = v_emb(v_gene)  # (batch, 32)
        j_feat = j_emb(j_gene)  # (batch, 16)

        # Concatenate chain features
        return torch.cat([cdr3_feat, v_feat, j_feat], dim=-1)  # (batch, 176)

    def forward(self, cdr3_alpha_esm: torch.Tensor, cdr3_beta_esm: torch.Tensor,
                v_alpha: torch.Tensor, j_alpha: torch.Tensor,
                v_beta: torch.Tensor, j_beta: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            cdr3_alpha_esm: (batch, esm_dim) ESM embeddings for CDR3α
            cdr3_beta_esm: (batch, esm_dim) ESM embeddings for CDR3β
            v_alpha, j_alpha: (batch,) V/J gene indices for TRA
            v_beta, j_beta: (batch,) V/J gene indices for TRB

        Returns:
            (batch, output_dim) TCR embedding
        '''
        # Encode each chain
        tra_feat = self._encode_chain(
            cdr3_alpha_esm, v_alpha, j_alpha,
            self.tra_cdr3_encoder, self.tra_v_emb, self.tra_j_emb
        )

        trb_feat = self._encode_chain(
            cdr3_beta_esm, v_beta, j_beta,
            self.trb_cdr3_encoder, self.trb_v_emb, self.trb_j_emb
        )

        # Fuse chains
        if self.fusion_type == 'concat':
            combined = torch.cat([tra_feat, trb_feat], dim=-1)
            return self.fusion_fc(combined)

        elif self.fusion_type == 'gated':
            combined = torch.cat([tra_feat, trb_feat], dim=-1)
            gates = self.fusion_gate(combined)
            weighted = gates[:, 0:1] * tra_feat + gates[:, 1:2] * trb_feat
            return self.fusion_fc(weighted)


# --- Epitope Encoder with ESM ---
class EpitopeEncoderESM(nn.Module):
    '''
    Epitope encoder using ESM-2 embeddings.
    '''

    def __init__(self, esm_dim: int = 640, hidden_dim: int = 256,
                 output_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_dim = output_dim

    def forward(self, esm_embeddings: torch.Tensor) -> torch.Tensor:
        return self.encoder(esm_embeddings)


# --- MHC Encoder (unchanged from V9.1) ---
class MHCEncoderV10(nn.Module):
    '''MHC class and allele embeddings.'''

    def __init__(self, mhc_class_vocab_size: int = 4,
                 mhc_allele_vocab_size: int = 116,
                 mhc_class_embed_dim: int = 8,
                 mhc_allele_embed_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()

        self.class_embedding = nn.Embedding(mhc_class_vocab_size, mhc_class_embed_dim, padding_idx=0)
        self.allele_embedding = nn.Embedding(mhc_allele_vocab_size, mhc_allele_embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = mhc_class_embed_dim + mhc_allele_embed_dim

        # Initialize
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.allele_embedding.weight, mean=0.0, std=0.1)
        self.class_embedding.weight.data[0] = 0.0
        self.allele_embedding.weight.data[0] = 0.0

    def forward(self, mhc_class: torch.Tensor, mhc_allele: torch.Tensor) -> torch.Tensor:
        class_emb = self.class_embedding(mhc_class)
        allele_emb = self.allele_embedding(mhc_allele)
        return self.dropout(torch.cat([class_emb, allele_emb], dim=-1))


# --- Epitope-MHC Encoder V10 ---
class EpitopeMHCEncoderV10(nn.Module):
    '''
    Combined epitope (ESM) + MHC encoder.
    '''

    def __init__(self, esm_dim: int = 640, hidden_dim: int = 256,
                 output_dim: int = 256, dropout: float = 0.3,
                 mhc_class_vocab_size: int = 4,
                 mhc_allele_vocab_size: int = 116,
                 mhc_class_embed_dim: int = 8,
                 mhc_allele_embed_dim: int = 32):
        super().__init__()

        self.epitope_encoder = EpitopeEncoderESM(esm_dim, hidden_dim, output_dim, dropout)
        self.mhc_encoder = MHCEncoderV10(
            mhc_class_vocab_size, mhc_allele_vocab_size,
            mhc_class_embed_dim, mhc_allele_embed_dim, dropout
        )

        mhc_dim = mhc_class_embed_dim + mhc_allele_embed_dim

        self.fusion = nn.Sequential(
            nn.Linear(output_dim + mhc_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        self.output_dim = output_dim

    def forward(self, epitope_esm: torch.Tensor,
                mhc_class: torch.Tensor, mhc_allele: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            epitope_esm: (batch, esm_dim) ESM embeddings for epitopes
            mhc_class: (batch,) MHC class indices
            mhc_allele: (batch,) MHC allele indices
        Returns:
            (batch, output_dim) epitope-MHC embedding
        '''
        epi_feat = self.epitope_encoder(epitope_esm)
        mhc_feat = self.mhc_encoder(mhc_class, mhc_allele)
        combined = torch.cat([epi_feat, mhc_feat], dim=-1)
        return self.fusion(combined)

    def encode_epitope_only(self, epitope_esm: torch.Tensor) -> torch.Tensor:
        '''Encode epitope without MHC (for inference with unknown MHC).'''
        return self.epitope_encoder(epitope_esm)


# --- Main Model V10 ---
class TCREpitopeModelV10(nn.Module):
    '''
    TCR-epitope binding prediction with ESM-2 embeddings.

    This model uses pre-computed ESM-2 embeddings for CDR3 and epitope
    sequences, with learned embeddings for V/J genes and MHC.

    The ESM embedding matrices are stored as frozen buffers, enabling
    self-contained model saving/loading for inference.
    '''

    def __init__(self, esm_dim: int = 640, hidden_dim: int = 256,
                 output_dim: int = 256, dropout: float = 0.3,
                 trb_v_size: int = 115, trb_j_size: int = 35,
                 tra_v_size: int = 187, tra_j_size: int = 78,
                 v_embed_dim: int = 32, j_embed_dim: int = 16,
                 fusion: str = 'concat',
                 mhc_class_vocab_size: int = 4,
                 mhc_allele_vocab_size: int = 116,
                 mhc_class_embed_dim: int = 8,
                 mhc_allele_embed_dim: int = 32,
                 temperature: float = 0.07):
        super().__init__()

        self.esm_dim = esm_dim
        self.output_dim = output_dim

        # TCR encoder
        self.tcr_encoder = DualChainTCREncoderV10(
            esm_dim, hidden_dim, output_dim, dropout,
            trb_v_size, trb_j_size, tra_v_size, tra_j_size,
            v_embed_dim, j_embed_dim, fusion
        )

        # Epitope-MHC encoder
        self.epitope_mhc_encoder = EpitopeMHCEncoderV10(
            esm_dim, hidden_dim, output_dim, dropout,
            mhc_class_vocab_size, mhc_allele_vocab_size,
            mhc_class_embed_dim, mhc_allele_embed_dim
        )

        self.temperature = nn.Parameter(torch.ones(1) * temperature)

        # ESM embedding buffers (initialized as empty, set via set_esm_embeddings)
        self.register_buffer('esm_cdr3_alpha', None)
        self.register_buffer('esm_cdr3_beta', None)
        self.register_buffer('esm_epitope', None)
        self._esm_initialized = False

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'TCREpitopeModelV10: {trainable:,} trainable parameters')

    def set_esm_embeddings(self, cdr3_alpha_emb: np.ndarray,
                           cdr3_beta_emb: np.ndarray,
                           epitope_emb: np.ndarray) -> None:
        '''
        Set pre-computed ESM embeddings as frozen buffers.

        Args:
            cdr3_alpha_emb: (n_unique_alpha, esm_dim) embeddings
            cdr3_beta_emb: (n_unique_beta, esm_dim) embeddings
            epitope_emb: (n_unique_epitope, esm_dim) embeddings
        '''
        device = next(self.parameters()).device

        self.esm_cdr3_alpha = torch.tensor(cdr3_alpha_emb, dtype=torch.float32, device=device)
        self.esm_cdr3_beta = torch.tensor(cdr3_beta_emb, dtype=torch.float32, device=device)
        self.esm_epitope = torch.tensor(epitope_emb, dtype=torch.float32, device=device)
        self._esm_initialized = True

        print(f'ESM embeddings set:')
        print(f'  CDR3α: {self.esm_cdr3_alpha.shape}')
        print(f'  CDR3β: {self.esm_cdr3_beta.shape}')
        print(f'  Epitope: {self.esm_epitope.shape}')

    def _lookup_esm(self, indices: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
        '''Look up ESM embeddings by index.'''
        return buffer[indices]

    def encode_tcr(self, cdr3_alpha_idx: torch.Tensor, cdr3_beta_idx: torch.Tensor,
                   v_alpha: torch.Tensor, j_alpha: torch.Tensor,
                   v_beta: torch.Tensor, j_beta: torch.Tensor) -> torch.Tensor:
        '''
        Encode TCR from ESM embedding indices.

        Args:
            cdr3_alpha_idx: (batch,) indices into esm_cdr3_alpha buffer
            cdr3_beta_idx: (batch,) indices into esm_cdr3_beta buffer
            v_alpha, j_alpha, v_beta, j_beta: (batch,) V/J gene indices

        Returns:
            (batch, output_dim) normalized TCR embedding
        '''
        if not self._esm_initialized:
            raise RuntimeError('ESM embeddings not set. Call set_esm_embeddings() first.')

        # Look up ESM embeddings
        cdr3_alpha_esm = self._lookup_esm(cdr3_alpha_idx, self.esm_cdr3_alpha)
        cdr3_beta_esm = self._lookup_esm(cdr3_beta_idx, self.esm_cdr3_beta)

        # Encode
        tcr_emb = self.tcr_encoder(
            cdr3_alpha_esm, cdr3_beta_esm,
            v_alpha, j_alpha, v_beta, j_beta
        )

        return F.normalize(tcr_emb, p=2, dim=-1)

    def encode_epitope_mhc(self, epitope_idx: torch.Tensor,
                           mhc_class: torch.Tensor,
                           mhc_allele: torch.Tensor) -> torch.Tensor:
        '''
        Encode epitope with MHC context from ESM embedding indices.

        Args:
            epitope_idx: (batch,) indices into esm_epitope buffer
            mhc_class: (batch,) MHC class indices
            mhc_allele: (batch,) MHC allele indices

        Returns:
            (batch, output_dim) normalized epitope-MHC embedding
        '''
        if not self._esm_initialized:
            raise RuntimeError('ESM embeddings not set. Call set_esm_embeddings() first.')

        epitope_esm = self._lookup_esm(epitope_idx, self.esm_epitope)
        emb = self.epitope_mhc_encoder(epitope_esm, mhc_class, mhc_allele)
        return F.normalize(emb, p=2, dim=-1)

    def encode_epitope(self, epitope_idx: torch.Tensor) -> torch.Tensor:
        '''Encode epitope without MHC (for inference with unknown MHC).'''
        if not self._esm_initialized:
            raise RuntimeError('ESM embeddings not set. Call set_esm_embeddings() first.')

        epitope_esm = self._lookup_esm(epitope_idx, self.esm_epitope)
        emb = self.epitope_mhc_encoder.encode_epitope_only(epitope_esm)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, cdr3_alpha_idx: torch.Tensor, cdr3_beta_idx: torch.Tensor,
                unique_epitope_idx: torch.Tensor,
                v_alpha: torch.Tensor, j_alpha: torch.Tensor,
                v_beta: torch.Tensor, j_beta: torch.Tensor,
                unique_mhc_class: Optional[torch.Tensor] = None,
                unique_mhc_allele: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass computing similarity between TCRs and epitopes.

        Args:
            cdr3_alpha_idx: (batch,) ESM indices for CDR3α
            cdr3_beta_idx: (batch,) ESM indices for CDR3β
            unique_epitope_idx: (n_epitopes,) ESM indices for unique epitopes
            v_alpha, j_alpha, v_beta, j_beta: (batch,) V/J gene indices
            unique_mhc_class: (n_epitopes,) MHC class for each unique epitope
            unique_mhc_allele: (n_epitopes,) MHC allele for each unique epitope

        Returns:
            sim: (batch, n_epitopes) similarity scores
            tcr_emb: (batch, output_dim) TCR embeddings
            epi_emb: (n_epitopes, output_dim) epitope embeddings
        '''
        # Encode TCRs
        tcr_emb = self.encode_tcr(
            cdr3_alpha_idx, cdr3_beta_idx,
            v_alpha, j_alpha, v_beta, j_beta
        )

        # Encode epitopes
        if unique_mhc_class is not None and unique_mhc_allele is not None:
            epi_emb = self.encode_epitope_mhc(unique_epitope_idx, unique_mhc_class, unique_mhc_allele)
        else:
            epi_emb = self.encode_epitope(unique_epitope_idx)

        # Compute similarity
        sim = torch.mm(tcr_emb, epi_emb.t()) / self.temperature.clamp(min=0.01)

        return sim, tcr_emb, epi_emb


# --- Factory Function ---
def create_model_v10(esm_dim=640, hidden_dim=256, output_dim=256, dropout=0.3,
                     trb_v_size=115, trb_j_size=35, tra_v_size=187, tra_j_size=78,
                     v_embed_dim=32, j_embed_dim=16, fusion='concat',
                     mhc_class_vocab_size=4, mhc_allele_vocab_size=116,
                     mhc_class_embed_dim=8, mhc_allele_embed_dim=32,
                     temperature=0.07):
    '''Create V10 model.'''
    return TCREpitopeModelV10(
        esm_dim, hidden_dim, output_dim, dropout,
        trb_v_size, trb_j_size, tra_v_size, tra_j_size,
        v_embed_dim, j_embed_dim, fusion,
        mhc_class_vocab_size, mhc_allele_vocab_size,
        mhc_class_embed_dim, mhc_allele_embed_dim,
        temperature
    )
")

# ===== R Interface Functions =====

#' Create TCR-Epitope Model V10 (ESM-2 enhanced)
#'
#' @param esm_dim ESM-2 embedding dimension (640 for t30_150M)
#' @param hidden_dim Hidden layer dimension
#' @param output_dim Output embedding dimension
#' @param dropout Dropout rate
#' @param trb_v_vocab_size TRB V gene vocabulary size
#' @param trb_j_vocab_size TRB J gene vocabulary size
#' @param tra_v_vocab_size TRA V gene vocabulary size
#' @param tra_j_vocab_size TRA J gene vocabulary size
#' @param v_embed_dim V gene embedding dimension
#' @param j_embed_dim J gene embedding dimension
#' @param fusion Chain fusion type ('concat' or 'gated')
#' @param mhc_class_vocab_size MHC class vocabulary size
#' @param mhc_allele_vocab_size MHC allele vocabulary size
#' @param mhc_class_embed_dim MHC class embedding dimension
#' @param mhc_allele_embed_dim MHC allele embedding dimension
#' @param temperature Initial temperature for similarity scaling
#' @return PyTorch model object
#' @export
create_tcr_epitope_model_v10 <- function(esm_dim = 640L,
                                         hidden_dim = 256L,
                                         output_dim = 256L,
                                         dropout = 0.3,
                                         trb_v_vocab_size = 115L,
                                         trb_j_vocab_size = 35L,
                                         tra_v_vocab_size = 187L,
                                         tra_j_vocab_size = 78L,
                                         v_embed_dim = 32L,
                                         j_embed_dim = 16L,
                                         fusion = "concat",
                                         mhc_class_vocab_size = 4L,
                                         mhc_allele_vocab_size = 116L,
                                         mhc_class_embed_dim = 8L,
                                         mhc_allele_embed_dim = 32L,
                                         temperature = 0.07) {

  py$create_model_v10(
    esm_dim = as.integer(esm_dim),
    hidden_dim = as.integer(hidden_dim),
    output_dim = as.integer(output_dim),
    dropout = dropout,
    trb_v_size = as.integer(trb_v_vocab_size),
    trb_j_size = as.integer(trb_j_vocab_size),
    tra_v_size = as.integer(tra_v_vocab_size),
    tra_j_size = as.integer(tra_j_vocab_size),
    v_embed_dim = as.integer(v_embed_dim),
    j_embed_dim = as.integer(j_embed_dim),
    fusion = fusion,
    mhc_class_vocab_size = as.integer(mhc_class_vocab_size),
    mhc_allele_vocab_size = as.integer(mhc_allele_vocab_size),
    mhc_class_embed_dim = as.integer(mhc_class_embed_dim),
    mhc_allele_embed_dim = as.integer(mhc_allele_embed_dim),
    temperature = temperature
  )
}

#' Initialize model with ESM embeddings from cache
#'
#' @param model V10 model from create_tcr_epitope_model_v10()
#' @param emb_cache Embedding cache from load_embedding_cache()
#' @param device Target device ('cuda' or 'cpu')
#' @return Model with ESM embeddings set
#' @export
initialize_model_esm <- function(model, emb_cache, device = "cuda") {

  message("Initializing model with ESM embeddings...")

  # Move model to device first
  model <- model$to(device)

  # Set ESM embeddings
  model$set_esm_embeddings(
    cdr3_alpha_emb = emb_cache$cdr3_alpha$embeddings,
    cdr3_beta_emb = emb_cache$cdr3_beta$embeddings,
    epitope_emb = emb_cache$epitope$embeddings
  )

  model
}

#' Print V10 model summary
#' @param model PyTorch V10 model object
#' @export
print_model_summary_v10 <- function(model) {

  py_run_string("
def _model_summary_v10(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)

    # ESM buffer sizes
    esm_size = 0
    if m.esm_cdr3_alpha is not None:
        esm_size += m.esm_cdr3_alpha.numel()
    if m.esm_cdr3_beta is not None:
        esm_size += m.esm_cdr3_beta.numel()
    if m.esm_epitope is not None:
        esm_size += m.esm_epitope.numel()

    return {
        'total_params': total,
        'trainable_params': trainable,
        'esm_buffer_elements': esm_size,
        'esm_initialized': m._esm_initialized
    }
")

  summary <- py$`_model_summary_v10`(model)

  cat("\n", strrep("=", 50), "\n", sep = "")
  cat("TCREpitopeModelV10 Summary\n")
  cat(strrep("=", 50), "\n", sep = "")
  cat(sprintf("Trainable parameters: %s\n", format(summary$trainable_params, big.mark = ",")))
  cat(sprintf("Total parameters: %s\n", format(summary$total_params, big.mark = ",")))
  cat(sprintf("ESM buffer elements: %s\n", format(summary$esm_buffer_elements, big.mark = ",")))
  cat(sprintf("ESM initialized: %s\n", summary$esm_initialized))
  cat(strrep("=", 50), "\n\n", sep = "")

  invisible(summary)
}
