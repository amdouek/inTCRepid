# ============================================================================
# TCR-Epitope Prediction Model Architecture (V9.1: MHC Integration)
# ============================================================================
#
# V7: Dual-chain TRA+TRB support
# V9.1: Added MHC class and allele embeddings integrated with epitope encoder
#
# ============================================================================

library(reticulate)

# ===== Amino Acid Properties for Embedding Initialization =====

#' Get Atchley factors as Python dictionary
#' @return Python dict mapping amino acids to 5-dimensional factor vectors
get_atchley_factors_py <- function() {

  atchley <- list(
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
  r_to_py(atchley)
}

#' Get BLOSUM62 matrix as Python-compatible structure
#' @return Python dict with 'matrix' (20x20) and 'aa_order' keys
get_blosum62_py <- function() {
  aa_order <- c("A","C","D","E","F","G","H","I","K","L",
                "M","N","P","Q","R","S","T","V","W","Y")

  blosum62 <- matrix(c(
    4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0,
    -1,9,-3,-4,-2,-3,-3,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,
    -2,-3,6,2,-3,-1,1,-1,-1,-3,-4,0,-3,-3,-1,0,-1,-4,-3,-3,
    -2,-4,2,5,-3,-2,0,-2,1,-3,-3,0,-2,-3,-1,0,-1,-3,-2,-2,
    -2,-2,-3,-3,6,-3,-1,0,-3,0,0,-3,0,3,-4,-2,-2,1,3,-1,
    0,-3,-1,-2,-3,6,-2,-4,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3,
    -2,-3,1,0,-1,-2,8,-3,-1,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3,
    -1,-1,-1,-2,0,-4,-3,4,-3,2,1,-3,1,0,-3,-2,-1,-3,-1,3,
    -1,-3,-1,1,-3,-2,-1,-3,5,-2,-2,0,-1,-3,-1,0,-1,-3,-2,-2,
    -1,-1,-3,-3,0,-4,-3,2,-2,4,2,-3,2,0,-3,-2,-1,-2,-1,1,
    -1,-1,-4,-3,0,-4,-3,1,-2,2,5,-2,2,0,-3,-2,-1,-1,-1,1,
    -1,-3,0,0,-3,-2,-1,-3,0,-3,-3,6,-2,-3,-2,0,0,-4,-2,-3,
    -1,-1,-3,-2,0,-3,-2,1,-1,2,2,-2,5,0,-2,-1,-1,-1,-1,1,
    -2,-2,-3,-3,3,-3,-1,0,-3,0,0,-3,0,6,-4,-2,-2,1,3,-1,
    -1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-3,-2,-2,-4,7,-1,-1,-4,-3,-2,
    1,-1,0,0,-2,0,-1,-2,0,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2,
    0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1,0,-1,-2,-1,1,5,-2,-2,0,
    -3,-2,-4,-3,1,-2,-2,-3,-3,-2,-1,-4,-1,1,-4,-3,-2,11,2,-3,
    -2,-2,-3,-2,3,-3,2,-1,-2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1,
    0,-1,-3,-2,-1,-3,-3,3,-2,1,1,-3,1,-1,-2,-2,0,-3,-1,4
  ), nrow = 20, ncol = 20, byrow = TRUE,
  dimnames = list(aa_order, aa_order))

  r_to_py(list(matrix = blosum62, aa_order = aa_order))
}

# ===== PyTorch Model Definition (V9.1: MHC Integration) =====

py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List

# --- Vocabulary ---
AA_VOCAB = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
AA_VOCAB['<PAD>'] = 0
AA_VOCAB['<UNK>'] = 21
VOCAB_SIZE = 22
AA_INDEX_TO_CHAR = {v: k for k, v in AA_VOCAB.items()}

# --- Embedding Initialization ---
def initialize_embeddings_with_atchley(embedding: nn.Embedding,
                                       atchley: Dict[str, List[float]],
                                       scale: float = 1.0) -> None:
    '''Initialize first 5 dims with Atchley factors; remainder random.'''
    with torch.no_grad():
        nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
        embedding.weight[0] = 0.0  # PAD

        all_factors = []
        for aa, idx in AA_VOCAB.items():
            if aa in atchley:
                factors = torch.tensor(atchley[aa], dtype=torch.float32) * scale
                all_factors.append(factors)
                n = min(5, embedding.embedding_dim)
                embedding.weight[idx, :n] = factors[:n]

        # UNK = mean of all AA factors
        if all_factors and embedding.embedding_dim >= 5:
            embedding.weight[AA_VOCAB['<UNK>'], :5] = torch.stack(all_factors).mean(0)

def compute_blosum_similarity_matrix(blosum_data: Dict) -> torch.Tensor:
    '''Convert BLOSUM62 to normalized [0,1] similarity matrix.'''
    mat = np.array(blosum_data['matrix'])
    aa_order = blosum_data['aa_order']

    # Normalize to [0, 1]
    norm = (mat - mat.min()) / (mat.max() - mat.min())

    sim = torch.zeros(VOCAB_SIZE, VOCAB_SIZE)
    for i, aa1 in enumerate(aa_order):
        for j, aa2 in enumerate(aa_order):
            if aa1 in AA_VOCAB and aa2 in AA_VOCAB:
                sim[AA_VOCAB[aa1], AA_VOCAB[aa2]] = norm[i, j]
    return sim

# --- Epitope Encoder (shared across model versions) ---
class EpitopeEncoder(nn.Module):
    '''CNN encoder for epitope sequences with masked pooling.'''

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 dropout: float = 0.3, atchley: Optional[Dict] = None,
                 use_plm: bool = False, plm_dim: Optional[int] = None):
        super().__init__()
        self.use_plm = use_plm
        self.output_dim = output_dim

        if use_plm:
            self.plm_proj = nn.Linear(plm_dim, embed_dim)
            self.embedding = None
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.plm_proj = None
            if atchley:
                initialize_embeddings_with_atchley(self.embedding, atchley)

        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, plm_emb: Optional[torch.Tensor] = None):
        # Embed
        if self.use_plm and plm_emb is not None:
            emb = self.plm_proj(plm_emb)
        else:
            emb = self.embedding(x)

        mask = (x != 0).float()
        emb_t = emb.transpose(1, 2)

        # Conv
        c1 = F.relu(self.bn1(self.conv1(emb_t)))
        c2 = F.relu(self.bn2(self.conv2(emb_t)))

        # Masked max + avg pooling
        mask_exp = mask.unsqueeze(1)
        c1_max = c1.masked_fill(mask_exp == 0, -1e9).max(-1)[0]
        c2_max = c2.masked_fill(mask_exp == 0, -1e9).max(-1)[0]
        lengths = mask.sum(-1, keepdim=True).clamp(min=1)
        c1_avg = (c1 * mask_exp).sum(-1) / lengths
        c2_avg = (c2 * mask_exp).sum(-1) / lengths

        combined = torch.cat([c1_max + c1_avg, c2_max + c2_avg], dim=-1)
        return self.fc(self.dropout(combined))

# --- MHC Encoder (V9.1) ---
class MHCEncoder(nn.Module):
    '''
    Encodes MHC information using hierarchical embeddings.

    - MHC Class: 4 tokens (PAD, MHCI, MHCII, UNK) -> 8-dim embedding
    - MHC Allele: ~116 tokens (including special tokens) -> 32-dim embedding
    - Output: Concatenated 40-dim representation
    '''

    def __init__(self,
                 mhc_class_vocab_size: int = 4,
                 mhc_allele_vocab_size: int = 116,
                 mhc_class_embed_dim: int = 8,
                 mhc_allele_embed_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()

        self.class_embed_dim = mhc_class_embed_dim
        self.allele_embed_dim = mhc_allele_embed_dim
        self.output_dim = mhc_class_embed_dim + mhc_allele_embed_dim

        # Embeddings with padding_idx=0
        self.class_embedding = nn.Embedding(
            mhc_class_vocab_size, mhc_class_embed_dim, padding_idx=0)
        self.allele_embedding = nn.Embedding(
            mhc_allele_vocab_size, mhc_allele_embed_dim, padding_idx=0)

        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.allele_embedding.weight, mean=0.0, std=0.1)
        self.class_embedding.weight.data[0] = 0.0  # PAD
        self.allele_embedding.weight.data[0] = 0.0  # PAD

    def forward(self, mhc_class_idx: torch.Tensor,
                mhc_allele_idx: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            mhc_class_idx: (batch,) tensor of MHC class indices
            mhc_allele_idx: (batch,) tensor of MHC allele indices
        Returns:
            (batch, 40) MHC embedding
        '''
        class_emb = self.class_embedding(mhc_class_idx)    # (batch, 8)
        allele_emb = self.allele_embedding(mhc_allele_idx)  # (batch, 32)

        combined = torch.cat([class_emb, allele_emb], dim=-1)  # (batch, 40)
        return self.dropout(combined)

# --- Epitope-MHC Encoder (V9.1) ---
class EpitopeMHCEncoder(nn.Module):
    '''
    Combined encoder for epitope sequence + MHC context.

    Epitope features (256-dim) are concatenated with MHC embedding (40-dim)
    and projected back to output_dim (256-dim).

    This captures the biological reality that epitope presentation is
    MHC-restricted.
    '''

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 dropout: float = 0.3, atchley: Optional[Dict] = None,
                 use_plm: bool = False, plm_dim: Optional[int] = None,
                 mhc_class_vocab_size: int = 4,
                 mhc_allele_vocab_size: int = 116,
                 mhc_class_embed_dim: int = 8,
                 mhc_allele_embed_dim: int = 32):
        super().__init__()

        self.output_dim = output_dim

        # Epitope encoder (produces output_dim features)
        self.epitope_encoder = EpitopeEncoder(
            vocab_size, embed_dim, hidden_dim, output_dim, dropout,
            atchley, use_plm, plm_dim)

        # MHC encoder
        self.mhc_encoder = MHCEncoder(
            mhc_class_vocab_size, mhc_allele_vocab_size,
            mhc_class_embed_dim, mhc_allele_embed_dim, dropout)

        mhc_dim = mhc_class_embed_dim + mhc_allele_embed_dim  # 40

        # Fusion: epitope_dim + mhc_dim -> output_dim
        self.fusion = nn.Sequential(
            nn.Linear(output_dim + mhc_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, epitope_idx: torch.Tensor,
                mhc_class_idx: torch.Tensor,
                mhc_allele_idx: torch.Tensor,
                plm_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Args:
            epitope_idx: (batch, seq_len) epitope token indices
            mhc_class_idx: (batch,) MHC class indices
            mhc_allele_idx: (batch,) MHC allele indices
            plm_emb: Optional PLM embeddings for epitope
        Returns:
            (batch, output_dim) Epitope-MHC embedding
        '''
        # Encode epitope
        epi_features = self.epitope_encoder(epitope_idx, plm_emb)  # (batch, 256)

        # Encode MHC
        mhc_features = self.mhc_encoder(mhc_class_idx, mhc_allele_idx)  # (batch, 40)

        # Fuse
        combined = torch.cat([epi_features, mhc_features], dim=-1)  # (batch, 296)
        return self.fusion(combined)  # (batch, 256)

    def encode_epitope_only(self, epitope_idx: torch.Tensor,
                            plm_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''Encode epitope without MHC (for backward compatibility).'''
        return self.epitope_encoder(epitope_idx, plm_emb)

# --- Dual Chain TCR Encoder (V7) ---
class DualChainTCREncoder(nn.Module):
    '''
    Dual-chain TCR encoder combining TRA (alpha) and TRB (beta).

    Each chain: CDR3 -> Conv+Attention -> concat V/J embeddings -> chain features
    Fusion: Concat/Gated/Attention -> output embedding
    '''

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 dropout: float = 0.3,
                 trb_v_size: int = 115, trb_j_size: int = 35,
                 tra_v_size: int = 187, tra_j_size: int = 78,
                 v_embed_dim: int = 32, j_embed_dim: int = 16,
                 fusion: str = 'concat',
                 atchley: Optional[Dict] = None,
                 use_plm: bool = False, plm_dim: Optional[int] = None):
        super().__init__()

        self.use_plm = use_plm
        self.fusion = fusion
        self.hidden_dim = hidden_dim

        # Feature dimensions
        cdr3_feat_dim = hidden_dim * 3
        chain_feat_dim = cdr3_feat_dim + v_embed_dim + j_embed_dim

        # Shared AA embedding
        if use_plm:
            self.plm_proj = nn.Linear(plm_dim, embed_dim)
            self.embedding = None
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.plm_proj = None
            if atchley:
                initialize_embeddings_with_atchley(self.embedding, atchley)

        # TRA components
        self.tra_conv1 = nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
        self.tra_conv2 = nn.Conv1d(embed_dim, hidden_dim, 5, padding=2)
        self.tra_conv3 = nn.Conv1d(embed_dim, hidden_dim, 7, padding=3)
        self.tra_bn1 = nn.BatchNorm1d(hidden_dim)
        self.tra_bn2 = nn.BatchNorm1d(hidden_dim)
        self.tra_bn3 = nn.BatchNorm1d(hidden_dim)
        self.tra_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1))
        self.tra_v_emb = nn.Embedding(tra_v_size, v_embed_dim, padding_idx=0)
        self.tra_j_emb = nn.Embedding(tra_j_size, j_embed_dim, padding_idx=0)

        # TRB components
        self.trb_conv1 = nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
        self.trb_conv2 = nn.Conv1d(embed_dim, hidden_dim, 5, padding=2)
        self.trb_conv3 = nn.Conv1d(embed_dim, hidden_dim, 7, padding=3)
        self.trb_bn1 = nn.BatchNorm1d(hidden_dim)
        self.trb_bn2 = nn.BatchNorm1d(hidden_dim)
        self.trb_bn3 = nn.BatchNorm1d(hidden_dim)
        self.trb_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1))
        self.trb_v_emb = nn.Embedding(trb_v_size, v_embed_dim, padding_idx=0)
        self.trb_j_emb = nn.Embedding(trb_j_size, j_embed_dim, padding_idx=0)

        # Fusion
        if fusion == 'concat':
            self.fusion_fc = nn.Linear(chain_feat_dim * 2, output_dim)
        elif fusion == 'gated':
            self.fusion_gate = nn.Sequential(nn.Linear(chain_feat_dim * 2, 2), nn.Softmax(dim=-1))
            self.fusion_fc = nn.Linear(chain_feat_dim, output_dim)
        elif fusion == 'attention':
            self.cross_attn = nn.MultiheadAttention(chain_feat_dim, 8, dropout, batch_first=True)
            self.fusion_fc = nn.Linear(chain_feat_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.chain_feat_dim = chain_feat_dim

    def _attn_pool(self, features: torch.Tensor, mask: torch.Tensor,
                   attn: nn.Module) -> torch.Tensor:
        scores = attn(features).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)

    def _encode_chain(self, cdr3, v_gene, j_gene, conv1, conv2, conv3,
                      bn1, bn2, bn3, attn, v_emb, j_emb, plm=None):
        # Embed CDR3
        if self.use_plm and plm is not None:
            emb = self.plm_proj(plm)
        else:
            emb = self.embedding(cdr3)

        mask = (cdr3 != 0).float()
        emb_t = emb.transpose(1, 2)

        # Conv + attention pool
        c1 = F.relu(bn1(conv1(emb_t))).transpose(1, 2)
        c2 = F.relu(bn2(conv2(emb_t))).transpose(1, 2)
        c3 = F.relu(bn3(conv3(emb_t))).transpose(1, 2)

        p1 = self._attn_pool(c1, mask, attn)
        p2 = self._attn_pool(c2, mask, attn)
        p3 = self._attn_pool(c3, mask, attn)

        # Concat CDR3 + V/J features
        return torch.cat([p1, p2, p3, v_emb(v_gene), j_emb(j_gene)], dim=-1)

    def forward(self, cdr3_a, cdr3_b, v_a, j_a, v_b, j_b,
                plm_a=None, plm_b=None):
        # Encode chains
        tra = self._encode_chain(cdr3_a, v_a, j_a,
            self.tra_conv1, self.tra_conv2, self.tra_conv3,
            self.tra_bn1, self.tra_bn2, self.tra_bn3,
            self.tra_attn, self.tra_v_emb, self.tra_j_emb, plm_a)

        trb = self._encode_chain(cdr3_b, v_b, j_b,
            self.trb_conv1, self.trb_conv2, self.trb_conv3,
            self.trb_bn1, self.trb_bn2, self.trb_bn3,
            self.trb_attn, self.trb_v_emb, self.trb_j_emb, plm_b)

        # Fuse
        if self.fusion == 'concat':
            out = self.fusion_fc(self.dropout(torch.cat([tra, trb], -1)))
        elif self.fusion == 'gated':
            combined = torch.cat([tra, trb], -1)
            g = self.fusion_gate(combined)
            out = self.fusion_fc(self.dropout(g[:,0:1]*tra + g[:,1:2]*trb))
        elif self.fusion == 'attention':
            seq = torch.stack([tra, trb], dim=1)
            attn_out, _ = self.cross_attn(seq, seq, seq)
            out = self.fusion_fc(self.dropout(attn_out.reshape(attn_out.size(0), -1)))

        return out

# --- Main Model V7 (Legacy, without MHC) ---
class TCREpitopeModelV7(nn.Module):
    '''TCR-epitope binding prediction with dual-chain (TRA+TRB) support.'''

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 cdr3_max_len: int = 25, epitope_max_len: int = 30,
                 dropout: float = 0.3,
                 trb_v_size: int = 115, trb_j_size: int = 35,
                 tra_v_size: int = 187, tra_j_size: int = 78,
                 v_embed_dim: int = 32, j_embed_dim: int = 16,
                 fusion: str = 'concat',
                 atchley: Optional[Dict] = None, blosum: Optional[Dict] = None,
                 use_plm: bool = False, plm_dim: Optional[int] = None,
                 temperature: float = 0.07):
        super().__init__()

        self.output_dim = output_dim
        self.use_plm = use_plm

        # TCR encoder (dual chain)
        self.tcr_encoder = DualChainTCREncoder(
            vocab_size, embed_dim, hidden_dim, output_dim, dropout,
            trb_v_size, trb_j_size, tra_v_size, tra_j_size,
            v_embed_dim, j_embed_dim, fusion, atchley, use_plm, plm_dim)

        # Epitope encoder
        self.epitope_encoder = EpitopeEncoder(
            vocab_size, embed_dim, hidden_dim, output_dim, dropout,
            atchley, use_plm, plm_dim)

        self.temperature = nn.Parameter(torch.ones(1) * temperature)

        # Optional BLOSUM regularization
        if blosum:
            self.register_buffer('blosum_sim', compute_blosum_similarity_matrix(blosum))
            self.use_blosum = True
        else:
            self.blosum_sim = None
            self.use_blosum = False

        # Log params
        total = sum(p.numel() for p in self.parameters())
        print(f'TCREpitopeModelV7: {total:,} parameters')

    def encode_tcr(self, cdr3_a, cdr3_b, v_a, j_a, v_b, j_b,
                   plm_a=None, plm_b=None):
        emb = self.tcr_encoder(cdr3_a, cdr3_b, v_a, j_a, v_b, j_b, plm_a, plm_b)
        return F.normalize(emb, p=2, dim=-1)

    def encode_epitope(self, epi_idx, plm=None):
        emb = self.epitope_encoder(epi_idx, plm)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, cdr3_a, cdr3_b, epitope, v_a, j_a, v_b, j_b,
                plm_a=None, plm_b=None, plm_epi=None):
        tcr = self.encode_tcr(cdr3_a, cdr3_b, v_a, j_a, v_b, j_b, plm_a, plm_b)
        epi = self.encode_epitope(epitope, plm_epi)
        sim = torch.mm(tcr, epi.t()) / self.temperature.clamp(min=0.01)
        return sim, tcr, epi

# --- Main Model V9.1 (With MHC) ---
class TCREpitopeModelV91(nn.Module):
    '''
    TCR-epitope binding prediction with:
    - Dual-chain TCR support (TRA+TRB)
    - MHC class and allele embeddings integrated with epitope encoder

    V9.1 changes:
    - Added MHCEncoder for class (4 tokens) and allele (~116 tokens)
    - EpitopeMHCEncoder combines epitope + MHC context
    - Forward pass accepts mhc_class_idx and mhc_allele_idx
    '''

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 cdr3_max_len: int = 25, epitope_max_len: int = 30,
                 dropout: float = 0.3,
                 trb_v_size: int = 115, trb_j_size: int = 35,
                 tra_v_size: int = 187, tra_j_size: int = 78,
                 v_embed_dim: int = 32, j_embed_dim: int = 16,
                 fusion: str = 'concat',
                 mhc_class_vocab_size: int = 4,
                 mhc_allele_vocab_size: int = 116,
                 mhc_class_embed_dim: int = 8,
                 mhc_allele_embed_dim: int = 32,
                 atchley: Optional[Dict] = None, blosum: Optional[Dict] = None,
                 use_plm: bool = False, plm_dim: Optional[int] = None,
                 temperature: float = 0.07):
        super().__init__()

        self.output_dim = output_dim
        self.use_plm = use_plm

        # TCR encoder (dual chain) - unchanged from V7
        self.tcr_encoder = DualChainTCREncoder(
            vocab_size, embed_dim, hidden_dim, output_dim, dropout,
            trb_v_size, trb_j_size, tra_v_size, tra_j_size,
            v_embed_dim, j_embed_dim, fusion, atchley, use_plm, plm_dim)

        # Epitope-MHC encoder (NEW in V9.1)
        self.epitope_mhc_encoder = EpitopeMHCEncoder(
            vocab_size, embed_dim, hidden_dim, output_dim, dropout,
            atchley, use_plm, plm_dim,
            mhc_class_vocab_size, mhc_allele_vocab_size,
            mhc_class_embed_dim, mhc_allele_embed_dim)

        self.temperature = nn.Parameter(torch.ones(1) * temperature)

        # Optional BLOSUM regularization
        if blosum:
            self.register_buffer('blosum_sim', compute_blosum_similarity_matrix(blosum))
            self.use_blosum = True
        else:
            self.blosum_sim = None
            self.use_blosum = False

        # Log params
        total = sum(p.numel() for p in self.parameters())
        print(f'TCREpitopeModelV91: {total:,} parameters')

        # Log MHC encoder params
        mhc_params = sum(p.numel() for p in self.epitope_mhc_encoder.mhc_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.epitope_mhc_encoder.fusion.parameters())
        print(f'  MHC encoder: {mhc_params:,} parameters')
        print(f'  Epitope-MHC fusion: {fusion_params:,} parameters')

    def encode_tcr(self, cdr3_a, cdr3_b, v_a, j_a, v_b, j_b,
                   plm_a=None, plm_b=None):
        emb = self.tcr_encoder(cdr3_a, cdr3_b, v_a, j_a, v_b, j_b, plm_a, plm_b)
        return F.normalize(emb, p=2, dim=-1)

    def encode_epitope_mhc(self, epi_idx, mhc_class_idx, mhc_allele_idx, plm=None):
        '''Encode epitope with MHC context.'''
        emb = self.epitope_mhc_encoder(epi_idx, mhc_class_idx, mhc_allele_idx, plm)
        return F.normalize(emb, p=2, dim=-1)

    def encode_epitope(self, epi_idx, plm=None):
        '''Encode epitope without MHC (for inference with unknown MHC).'''
        emb = self.epitope_mhc_encoder.encode_epitope_only(epi_idx, plm)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, cdr3_a, cdr3_b, epitope, v_a, j_a, v_b, j_b,
                mhc_class=None, mhc_allele=None,
                plm_a=None, plm_b=None, plm_epi=None):
        '''
        Forward pass with optional MHC information.

        Args:
            cdr3_a: (batch, seq_len) CDR3 alpha indices
            cdr3_b: (batch, seq_len) CDR3 beta indices
            epitope: (batch, seq_len) or (n_epitopes, seq_len) epitope indices
            v_a, j_a, v_b, j_b: (batch,) V/J gene indices
            mhc_class: (batch,) or (n_epitopes,) MHC class indices (optional)
            mhc_allele: (batch,) or (n_epitopes,) MHC allele indices (optional)
            plm_*: Optional PLM embeddings

        Returns:
            sim: (batch, n_epitopes) similarity matrix
            tcr: (batch, output_dim) TCR embeddings
            epi: (n_epitopes, output_dim) epitope embeddings
        '''
        tcr = self.encode_tcr(cdr3_a, cdr3_b, v_a, j_a, v_b, j_b, plm_a, plm_b)

        # Encode epitopes with or without MHC
        if mhc_class is not None and mhc_allele is not None:
            epi = self.encode_epitope_mhc(epitope, mhc_class, mhc_allele, plm_epi)
        else:
            epi = self.encode_epitope(epitope, plm_epi)

        sim = torch.mm(tcr, epi.t()) / self.temperature.clamp(min=0.01)
        return sim, tcr, epi

# --- Factory Functions ---

def create_model_v7(vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256,
                    output_dim=256, cdr3_max_len=25, epitope_max_len=30,
                    dropout=0.3, trb_v_size=115, trb_j_size=35,
                    tra_v_size=187, tra_j_size=78, v_embed_dim=32,
                    j_embed_dim=16, fusion='concat', atchley=None,
                    blosum=None, use_plm=False, plm_dim=None, temperature=0.07):
    '''Create V7 model (without MHC).'''
    return TCREpitopeModelV7(
        vocab_size, embed_dim, hidden_dim, output_dim, cdr3_max_len,
        epitope_max_len, dropout, trb_v_size, trb_j_size, tra_v_size,
        tra_j_size, v_embed_dim, j_embed_dim, fusion, atchley, blosum,
        use_plm, plm_dim, temperature)

def create_model_v91(vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256,
                     output_dim=256, cdr3_max_len=25, epitope_max_len=30,
                     dropout=0.3, trb_v_size=115, trb_j_size=35,
                     tra_v_size=187, tra_j_size=78, v_embed_dim=32,
                     j_embed_dim=16, fusion='concat',
                     mhc_class_vocab_size=4, mhc_allele_vocab_size=116,
                     mhc_class_embed_dim=8, mhc_allele_embed_dim=32,
                     atchley=None, blosum=None, use_plm=False, plm_dim=None,
                     temperature=0.07):
    '''Create V9.1 model (with MHC).'''
    return TCREpitopeModelV91(
        vocab_size, embed_dim, hidden_dim, output_dim, cdr3_max_len,
        epitope_max_len, dropout, trb_v_size, trb_j_size, tra_v_size,
        tra_j_size, v_embed_dim, j_embed_dim, fusion,
        mhc_class_vocab_size, mhc_allele_vocab_size,
        mhc_class_embed_dim, mhc_allele_embed_dim,
        atchley, blosum, use_plm, plm_dim, temperature)
")

# ===== R Wrapper Functions =====

#' Create TCR-Epitope model (V7: dual-chain TRA+TRB, without MHC)
#'
#' @param vocab_size AA vocabulary size (default 22)
#' @param embed_dim Token embedding dimension (default 128)
#' @param hidden_dim Hidden layer dimension (default 256)
#' @param output_dim Output embedding dimension (default 256)
#' @param cdr3_max_len Maximum CDR3 length (default 25)
#' @param epitope_max_len Maximum epitope length (default 30)
#' @param dropout Dropout rate (default 0.3)
#' @param trb_v_vocab_size TRB V gene vocabulary size
#' @param trb_j_vocab_size TRB J gene vocabulary size
#' @param tra_v_vocab_size TRA V gene vocabulary size
#' @param tra_j_vocab_size TRA J gene vocabulary size
#' @param v_embed_dim V gene embedding dimension (default 32)
#' @param j_embed_dim J gene embedding dimension (default 16)
#' @param fusion Chain fusion type: 'concat', 'gated', or 'attention'
#' @param use_atchley_init Use Atchley factor initialization (default TRUE)
#' @param use_blosum_reg Use BLOSUM62 regularization (default FALSE)
#' @param use_plm Use PLM embeddings (default FALSE)
#' @param plm_dim PLM embedding dimension (required if use_plm=TRUE)
#' @param temperature Initial temperature for similarity scaling
#' @return PyTorch model object
#' @export
create_tcr_epitope_model <- function(vocab_size = 22L,
                                     embed_dim = 128L,
                                     hidden_dim = 256L,
                                     output_dim = 256L,
                                     cdr3_max_len = 25L,
                                     epitope_max_len = 30L,
                                     dropout = 0.3,
                                     trb_v_vocab_size = 115L,
                                     trb_j_vocab_size = 35L,
                                     tra_v_vocab_size = 187L,
                                     tra_j_vocab_size = 78L,
                                     v_embed_dim = 32L,
                                     j_embed_dim = 16L,
                                     fusion = "concat",
                                     use_atchley_init = TRUE,
                                     use_blosum_reg = FALSE,
                                     use_plm = FALSE,
                                     plm_dim = NULL,
                                     temperature = 0.07) {

  atchley <- if (use_atchley_init && !use_plm) get_atchley_factors_py() else NULL
  blosum <- if (use_blosum_reg && !use_plm) get_blosum62_py() else NULL
  plm_dim_py <- if (!is.null(plm_dim)) as.integer(plm_dim) else NULL

  py$create_model_v7(
    vocab_size = as.integer(vocab_size),
    embed_dim = as.integer(embed_dim),
    hidden_dim = as.integer(hidden_dim),
    output_dim = as.integer(output_dim),
    cdr3_max_len = as.integer(cdr3_max_len),
    epitope_max_len = as.integer(epitope_max_len),
    dropout = dropout,
    trb_v_size = as.integer(trb_v_vocab_size),
    trb_j_size = as.integer(trb_j_vocab_size),
    tra_v_size = as.integer(tra_v_vocab_size),
    tra_j_size = as.integer(tra_j_vocab_size),
    v_embed_dim = as.integer(v_embed_dim),
    j_embed_dim = as.integer(j_embed_dim),
    fusion = fusion,
    atchley = atchley,
    blosum = blosum,
    use_plm = use_plm,
    plm_dim = plm_dim_py,
    temperature = temperature
  )
}

#' Create TCR-Epitope model V9.1 (with MHC integration)
#'
#' @param vocab_size AA vocabulary size (default 22)
#' @param embed_dim Token embedding dimension (default 128)
#' @param hidden_dim Hidden layer dimension (default 256)
#' @param output_dim Output embedding dimension (default 256)
#' @param cdr3_max_len Maximum CDR3 length (default 25)
#' @param epitope_max_len Maximum epitope length (default 30)
#' @param dropout Dropout rate (default 0.3)
#' @param trb_v_vocab_size TRB V gene vocabulary size
#' @param trb_j_vocab_size TRB J gene vocabulary size
#' @param tra_v_vocab_size TRA V gene vocabulary size
#' @param tra_j_vocab_size TRA J gene vocabulary size
#' @param v_embed_dim V gene embedding dimension (default 32)
#' @param j_embed_dim J gene embedding dimension (default 16)
#' @param fusion Chain fusion type: 'concat', 'gated', or 'attention'
#' @param mhc_class_vocab_size MHC class vocabulary size (default 4)
#' @param mhc_allele_vocab_size MHC allele vocabulary size
#' @param mhc_class_embed_dim MHC class embedding dimension (default 8)
#' @param mhc_allele_embed_dim MHC allele embedding dimension (default 32)
#' @param use_atchley_init Use Atchley factor initialization (default TRUE)
#' @param use_blosum_reg Use BLOSUM62 regularization (default FALSE)
#' @param use_plm Use PLM embeddings (default FALSE)
#' @param plm_dim PLM embedding dimension (required if use_plm=TRUE)
#' @param temperature Initial temperature for similarity scaling
#' @return PyTorch model object
#' @export
create_tcr_epitope_model_v91 <- function(vocab_size = 22L,
                                         embed_dim = 128L,
                                         hidden_dim = 256L,
                                         output_dim = 256L,
                                         cdr3_max_len = 25L,
                                         epitope_max_len = 30L,
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
                                         use_atchley_init = TRUE,
                                         use_blosum_reg = FALSE,
                                         use_plm = FALSE,
                                         plm_dim = NULL,
                                         temperature = 0.07) {

  atchley <- if (use_atchley_init && !use_plm) get_atchley_factors_py() else NULL
  blosum <- if (use_blosum_reg && !use_plm) get_blosum62_py() else NULL
  plm_dim_py <- if (!is.null(plm_dim)) as.integer(plm_dim) else NULL

  py$create_model_v91(
    vocab_size = as.integer(vocab_size),
    embed_dim = as.integer(embed_dim),
    hidden_dim = as.integer(hidden_dim),
    output_dim = as.integer(output_dim),
    cdr3_max_len = as.integer(cdr3_max_len),
    epitope_max_len = as.integer(epitope_max_len),
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
    atchley = atchley,
    blosum = blosum,
    use_plm = use_plm,
    plm_dim = plm_dim_py,
    temperature = temperature
  )
}

#' Print model summary
#' @param model PyTorch model object
#' @export
print_model_summary <- function(model) {
  py_run_string("
def _count_params(m):
    return sum(p.numel() for p in m.parameters())
def _get_model_class(m):
    return m.__class__.__name__
")
  n <- py$`_count_params`(model)
  model_class <- py$`_get_model_class`(model)
  cat(sprintf("\n%s\nTotal parameters: %s\n\n", model_class, format(n, big.mark = ",")))
}
