# model_architecture_legacy_v5-6.R
# Legacy TCR-Epitope model architectures (V5, V6) - DEPRECATED
# Preserved for backward compatibility and model loading
# Current development uses V7 (dual-chain) in model_architecture.R
# ============================================================================

library(reticulate)

# ============================================================================
# Amino Acid Properties (shared utilities)
# ============================================================================

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

# ============================================================================
# PyTorch Legacy Models (V5, V6)
# ============================================================================

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

# --- Embedding Initialization ---
def initialize_embeddings_with_atchley(embedding: nn.Embedding,
                                       atchley: Dict[str, List[float]],
                                       scale: float = 1.0) -> None:
    with torch.no_grad():
        nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
        embedding.weight[0] = 0.0
        all_factors = []
        for aa, idx in AA_VOCAB.items():
            if aa in atchley:
                factors = torch.tensor(atchley[aa], dtype=torch.float32) * scale
                all_factors.append(factors)
                n = min(5, embedding.embedding_dim)
                embedding.weight[idx, :n] = factors[:n]
        if all_factors and embedding.embedding_dim >= 5:
            embedding.weight[AA_VOCAB['<UNK>'], :5] = torch.stack(all_factors).mean(0)

def compute_blosum_similarity_matrix(blosum_data: Dict) -> torch.Tensor:
    mat = np.array(blosum_data['matrix'])
    aa_order = blosum_data['aa_order']
    norm = (mat - mat.min()) / (mat.max() - mat.min())
    sim = torch.zeros(VOCAB_SIZE, VOCAB_SIZE)
    for i, aa1 in enumerate(aa_order):
        for j, aa2 in enumerate(aa_order):
            if aa1 in AA_VOCAB and aa2 in AA_VOCAB:
                sim[AA_VOCAB[aa1], AA_VOCAB[aa2]] = norm[i, j]
    return sim

# =============================================================================
# V5 Architecture (TRB-only, no V/J genes)
# =============================================================================

class CDR3EncoderV5(nn.Module):
    '''CDR3 encoder with CNN + attention (no V/J gene integration).'''

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 max_length: int = 25, dropout: float = 0.3,
                 atchley: Optional[Dict] = None,
                 use_plm: bool = False, plm_dim: Optional[int] = None):
        super().__init__()
        self.use_plm = use_plm

        if use_plm:
            self.plm_proj = nn.Linear(plm_dim, embed_dim)
            self.embedding = None
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.plm_proj = None
            if atchley:
                initialize_embeddings_with_atchley(self.embedding, atchley)

        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, hidden_dim, 5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, hidden_dim, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1))
        self.fc = nn.Linear(hidden_dim * 3, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, plm_emb=None):
        if self.use_plm and plm_emb is not None:
            emb = self.plm_proj(plm_emb)
        else:
            emb = self.embedding(x)

        if mask is None:
            mask = (x != 0).float()

        emb_t = emb.transpose(1, 2)
        c1 = F.relu(self.bn1(self.conv1(emb_t))).transpose(1, 2)
        c2 = F.relu(self.bn2(self.conv2(emb_t))).transpose(1, 2)
        c3 = F.relu(self.bn3(self.conv3(emb_t))).transpose(1, 2)

        def attn_pool(feat, mask):
            scores = self.attention(feat).squeeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1)
            return torch.bmm(weights.unsqueeze(1), feat).squeeze(1)

        combined = torch.cat([attn_pool(c1, mask), attn_pool(c2, mask),
                              attn_pool(c3, mask)], dim=-1)
        return self.fc(self.dropout(combined))


class EpitopeEncoderV5(nn.Module):
    '''Epitope encoder with CNN + masked pooling.'''

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 max_length: int = 30, dropout: float = 0.3,
                 atchley: Optional[Dict] = None,
                 use_plm: bool = False, plm_dim: Optional[int] = None):
        super().__init__()
        self.use_plm = use_plm

        if use_plm:
            self.plm_proj = nn.Linear(plm_dim, embed_dim)
            self.embedding = None
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.plm_proj = None
            if atchley:
                initialize_embeddings_with_atchley(self.embedding, atchley)

        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, hidden_dim, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, plm_emb=None):
        if self.use_plm and plm_emb is not None:
            emb = self.plm_proj(plm_emb)
        else:
            emb = self.embedding(x)

        mask = (x != 0).float()
        emb_t = emb.transpose(1, 2)

        c1 = F.relu(self.bn1(self.conv1(emb_t)))
        c2 = F.relu(self.bn2(self.conv2(emb_t)))

        mask_exp = mask.unsqueeze(1)
        c1_max = c1.masked_fill(mask_exp == 0, -1e9).max(-1)[0]
        c2_max = c2.masked_fill(mask_exp == 0, -1e9).max(-1)[0]
        lengths = mask.sum(-1, keepdim=True).clamp(min=1)
        c1_avg = (c1 * mask_exp).sum(-1) / lengths
        c2_avg = (c2 * mask_exp).sum(-1) / lengths

        combined = torch.cat([c1_max + c1_avg, c2_max + c2_avg], dim=-1)
        return self.fc(self.dropout(combined))


class TCREpitopeModelV5(nn.Module):
    '''V5 model: TRB CDR3 only, no V/J gene integration.'''

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 cdr3_max_len: int = 25, epitope_max_len: int = 30,
                 dropout: float = 0.3, atchley: Optional[Dict] = None,
                 blosum: Optional[Dict] = None, use_plm: bool = False,
                 plm_dim: Optional[int] = None, temperature: float = 0.07):
        super().__init__()
        self.output_dim = output_dim
        self.use_plm = use_plm

        self.cdr3_encoder = CDR3EncoderV5(
            vocab_size, embed_dim, hidden_dim, output_dim, cdr3_max_len,
            dropout, atchley, use_plm, plm_dim)
        self.epitope_encoder = EpitopeEncoderV5(
            vocab_size, embed_dim, hidden_dim, output_dim, epitope_max_len,
            dropout, atchley, use_plm, plm_dim)

        self.temperature = nn.Parameter(torch.ones(1) * temperature)

        if blosum:
            self.register_buffer('blosum_sim', compute_blosum_similarity_matrix(blosum))
            self.use_blosum = True
        else:
            self.blosum_sim = None
            self.use_blosum = False

    def encode_cdr3(self, cdr3, plm=None):
        return F.normalize(self.cdr3_encoder(cdr3, plm_emb=plm), p=2, dim=-1)

    def encode_epitope(self, epi, plm=None):
        return F.normalize(self.epitope_encoder(epi, plm_emb=plm), p=2, dim=-1)

    def forward(self, cdr3, epitope, cdr3_plm=None, epi_plm=None):
        cdr3_emb = self.encode_cdr3(cdr3, cdr3_plm)
        epi_emb = self.encode_epitope(epitope, epi_plm)
        sim = torch.mm(cdr3_emb, epi_emb.t()) / self.temperature.clamp(min=0.01)
        return sim, cdr3_emb, epi_emb

    def compute_blosum_regularization_loss(self):
        if not self.use_blosum or self.use_plm:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        weights = self.cdr3_encoder.embedding.weight
        norm = F.normalize(weights, p=2, dim=-1)
        sim = torch.mm(norm, norm.t())
        valid = torch.zeros_like(self.blosum_sim)
        valid[1:21, 1:21] = 1.0
        diff = (sim - self.blosum_sim) * valid
        return (diff ** 2).sum() / valid.sum()


def create_model_v5(vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256,
                    output_dim=256, cdr3_max_len=25, epitope_max_len=30,
                    dropout=0.3, atchley=None, blosum=None,
                    use_plm=False, plm_dim=None, temperature=0.07):
    return TCREpitopeModelV5(
        vocab_size, embed_dim, hidden_dim, output_dim, cdr3_max_len,
        epitope_max_len, dropout, atchley, blosum, use_plm, plm_dim, temperature)


# =============================================================================
# V6 Architecture (TRB with V/J gene integration)
# =============================================================================

class CDR3EncoderV6(nn.Module):
    '''CDR3 encoder with V/J gene embeddings concatenated before projection.'''

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 max_length: int = 25, dropout: float = 0.3,
                 v_vocab_size: int = 115, j_vocab_size: int = 35,
                 v_embed_dim: int = 32, j_embed_dim: int = 16,
                 atchley: Optional[Dict] = None,
                 use_plm: bool = False, plm_dim: Optional[int] = None):
        super().__init__()
        self.use_plm = use_plm
        self.use_vj = v_vocab_size > 0 and j_vocab_size > 0

        if use_plm:
            self.plm_proj = nn.Linear(plm_dim, embed_dim)
            self.embedding = None
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.plm_proj = None
            if atchley:
                initialize_embeddings_with_atchley(self.embedding, atchley)

        if self.use_vj:
            self.v_embedding = nn.Embedding(v_vocab_size, v_embed_dim, padding_idx=0)
            self.j_embedding = nn.Embedding(j_vocab_size, j_embed_dim, padding_idx=0)
        else:
            self.v_embedding = None
            self.j_embedding = None
            v_embed_dim = 0
            j_embed_dim = 0

        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, hidden_dim, 5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, hidden_dim, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1))

        cdr3_dim = hidden_dim * 3
        self.fc = nn.Linear(cdr3_dim + v_embed_dim + j_embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, v_gene=None, j_gene=None, mask=None, plm_emb=None):
        if self.use_plm and plm_emb is not None:
            emb = self.plm_proj(plm_emb)
        else:
            emb = self.embedding(x)

        if mask is None:
            mask = (x != 0).float()

        emb_t = emb.transpose(1, 2)
        c1 = F.relu(self.bn1(self.conv1(emb_t))).transpose(1, 2)
        c2 = F.relu(self.bn2(self.conv2(emb_t))).transpose(1, 2)
        c3 = F.relu(self.bn3(self.conv3(emb_t))).transpose(1, 2)

        def attn_pool(feat, mask):
            scores = self.attention(feat).squeeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1)
            return torch.bmm(weights.unsqueeze(1), feat).squeeze(1)

        cdr3_feat = torch.cat([attn_pool(c1, mask), attn_pool(c2, mask),
                               attn_pool(c3, mask)], dim=-1)

        if self.use_vj and v_gene is not None and j_gene is not None:
            combined = torch.cat([cdr3_feat, self.v_embedding(v_gene),
                                  self.j_embedding(j_gene)], dim=-1)
        else:
            combined = cdr3_feat

        return self.fc(self.dropout(combined))


class TCREpitopeModelV6(nn.Module):
    '''V6 model: TRB CDR3 with V/J gene integration.'''

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 cdr3_max_len: int = 25, epitope_max_len: int = 30,
                 dropout: float = 0.3,
                 v_vocab_size: int = 115, j_vocab_size: int = 35,
                 v_embed_dim: int = 32, j_embed_dim: int = 16,
                 atchley: Optional[Dict] = None, blosum: Optional[Dict] = None,
                 use_plm: bool = False, plm_dim: Optional[int] = None,
                 temperature: float = 0.07):
        super().__init__()
        self.output_dim = output_dim
        self.use_plm = use_plm
        self.use_vj = v_vocab_size > 0 and j_vocab_size > 0

        self.cdr3_encoder = CDR3EncoderV6(
            vocab_size, embed_dim, hidden_dim, output_dim, cdr3_max_len,
            dropout, v_vocab_size, j_vocab_size, v_embed_dim, j_embed_dim,
            atchley, use_plm, plm_dim)
        self.epitope_encoder = EpitopeEncoderV5(
            vocab_size, embed_dim, hidden_dim, output_dim, epitope_max_len,
            dropout, atchley, use_plm, plm_dim)

        self.temperature = nn.Parameter(torch.ones(1) * temperature)

        if blosum:
            self.register_buffer('blosum_sim', compute_blosum_similarity_matrix(blosum))
            self.use_blosum = True
        else:
            self.blosum_sim = None
            self.use_blosum = False

    def encode_cdr3(self, cdr3, v_gene=None, j_gene=None, plm=None):
        emb = self.cdr3_encoder(cdr3, v_gene=v_gene, j_gene=j_gene, plm_emb=plm)
        return F.normalize(emb, p=2, dim=-1)

    def encode_epitope(self, epi, plm=None):
        return F.normalize(self.epitope_encoder(epi, plm_emb=plm), p=2, dim=-1)

    def forward(self, cdr3, epitope, v_gene=None, j_gene=None,
                cdr3_plm=None, epi_plm=None):
        cdr3_emb = self.encode_cdr3(cdr3, v_gene, j_gene, cdr3_plm)
        epi_emb = self.encode_epitope(epitope, epi_plm)
        sim = torch.mm(cdr3_emb, epi_emb.t()) / self.temperature.clamp(min=0.01)
        return sim, cdr3_emb, epi_emb

    def compute_blosum_regularization_loss(self):
        if not self.use_blosum or self.use_plm:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        weights = self.cdr3_encoder.embedding.weight
        norm = F.normalize(weights, p=2, dim=-1)
        sim = torch.mm(norm, norm.t())
        valid = torch.zeros_like(self.blosum_sim)
        valid[1:21, 1:21] = 1.0
        diff = (sim - self.blosum_sim) * valid
        return (diff ** 2).sum() / valid.sum()


def create_model_v6(vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256,
                    output_dim=256, cdr3_max_len=25, epitope_max_len=30,
                    dropout=0.3, v_vocab_size=115, j_vocab_size=35,
                    v_embed_dim=32, j_embed_dim=16, atchley=None,
                    blosum=None, use_plm=False, plm_dim=None, temperature=0.07):
    return TCREpitopeModelV6(
        vocab_size, embed_dim, hidden_dim, output_dim, cdr3_max_len,
        epitope_max_len, dropout, v_vocab_size, j_vocab_size,
        v_embed_dim, j_embed_dim, atchley, blosum, use_plm, plm_dim, temperature)
")

# ============================================================================
# R Wrapper Functions (Legacy)
# ============================================================================

#' Create TCR-Epitope model V5 (TRB-only, no V/J genes) - DEPRECATED
#' @param vocab_size AA vocabulary size (default 22)
#' @param embed_dim Token embedding dimension (default 128)
#' @param hidden_dim Hidden layer dimension (default 256)
#' @param output_dim Output embedding dimension (default 256)
#' @param cdr3_max_len Maximum CDR3 length (default 25)
#' @param epitope_max_len Maximum epitope length (default 30)
#' @param dropout Dropout rate (default 0.3)
#' @param use_atchley_init Use Atchley factor initialization (default TRUE)
#' @param use_blosum_reg Use BLOSUM62 regularization (default FALSE)
#' @param use_plm Use PLM embeddings (default FALSE)
#' @param plm_dim PLM embedding dimension
#' @param temperature Initial temperature for similarity scaling
#' @return PyTorch model object
create_tcr_epitope_model_v5 <- function(vocab_size = 22L,
                                        embed_dim = 128L,
                                        hidden_dim = 256L,
                                        output_dim = 256L,
                                        cdr3_max_len = 25L,
                                        epitope_max_len = 30L,
                                        dropout = 0.3,
                                        use_atchley_init = TRUE,
                                        use_blosum_reg = FALSE,
                                        use_plm = FALSE,
                                        plm_dim = NULL,
                                        temperature = 0.07) {

  .Deprecated("create_tcr_epitope_model",
              msg = "V5 is deprecated. Use V7 (dual-chain) from model_architecture.R")

  atchley <- if (use_atchley_init && !use_plm) get_atchley_factors_py() else NULL
  blosum <- if (use_blosum_reg && !use_plm) get_blosum62_py() else NULL
  plm_dim_py <- if (!is.null(plm_dim)) as.integer(plm_dim) else NULL

  py$create_model_v5(
    vocab_size = as.integer(vocab_size),
    embed_dim = as.integer(embed_dim),
    hidden_dim = as.integer(hidden_dim),
    output_dim = as.integer(output_dim),
    cdr3_max_len = as.integer(cdr3_max_len),
    epitope_max_len = as.integer(epitope_max_len),
    dropout = dropout,
    atchley = atchley,
    blosum = blosum,
    use_plm = use_plm,
    plm_dim = plm_dim_py,
    temperature = temperature
  )
}

#' Create TCR-Epitope model V6 (TRB with V/J genes) - DEPRECATED
#' @param vocab_size AA vocabulary size (default 22)
#' @param embed_dim Token embedding dimension (default 128)
#' @param hidden_dim Hidden layer dimension (default 256)
#' @param output_dim Output embedding dimension (default 256)
#' @param cdr3_max_len Maximum CDR3 length (default 25)
#' @param epitope_max_len Maximum epitope length (default 30)
#' @param dropout Dropout rate (default 0.3)
#' @param v_gene_vocab_size V gene vocabulary size (default 115)
#' @param j_gene_vocab_size J gene vocabulary size (default 35)
#' @param v_embed_dim V gene embedding dimension (default 32)
#' @param j_embed_dim J gene embedding dimension (default 16)
#' @param use_atchley_init Use Atchley factor initialization (default TRUE)
#' @param use_blosum_reg Use BLOSUM62 regularization (default FALSE)
#' @param use_plm Use PLM embeddings (default FALSE)
#' @param plm_dim PLM embedding dimension
#' @param temperature Initial temperature for similarity scaling
#' @return PyTorch model object
create_tcr_epitope_model_v6 <- function(vocab_size = 22L,
                                        embed_dim = 128L,
                                        hidden_dim = 256L,
                                        output_dim = 256L,
                                        cdr3_max_len = 25L,
                                        epitope_max_len = 30L,
                                        dropout = 0.3,
                                        v_gene_vocab_size = 115L,
                                        j_gene_vocab_size = 35L,
                                        v_embed_dim = 32L,
                                        j_embed_dim = 16L,
                                        use_atchley_init = TRUE,
                                        use_blosum_reg = FALSE,
                                        use_plm = FALSE,
                                        plm_dim = NULL,
                                        temperature = 0.07) {

  .Deprecated("create_tcr_epitope_model",
              msg = "V6 is deprecated. Use V7 (dual-chain) from model_architecture.R")

  atchley <- if (use_atchley_init && !use_plm) get_atchley_factors_py() else NULL
  blosum <- if (use_blosum_reg && !use_plm) get_blosum62_py() else NULL
  plm_dim_py <- if (!is.null(plm_dim)) as.integer(plm_dim) else NULL

  py$create_model_v6(
    vocab_size = as.integer(vocab_size),
    embed_dim = as.integer(embed_dim),
    hidden_dim = as.integer(hidden_dim),
    output_dim = as.integer(output_dim),
    cdr3_max_len = as.integer(cdr3_max_len),
    epitope_max_len = as.integer(epitope_max_len),
    dropout = dropout,
    v_vocab_size = as.integer(v_gene_vocab_size),
    j_vocab_size = as.integer(j_gene_vocab_size),
    v_embed_dim = as.integer(v_embed_dim),
    j_embed_dim = as.integer(j_embed_dim),
    atchley = atchley,
    blosum = blosum,
    use_plm = use_plm,
    plm_dim = plm_dim_py,
    temperature = temperature
  )
}

#' Get model version from model object
#' @param model PyTorch model object
#' @return Character: 'v5', 'v6', or 'v7'
get_model_version <- function(model) {
  cls <- class(model)[1]
  if (grepl("V7", cls)) "v7"
  else if (grepl("V6", cls)) "v6"
  else if (grepl("V5", cls)) "v5"
  else "unknown"
}
