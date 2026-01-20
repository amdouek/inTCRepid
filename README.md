inTCRepid
================

<!-- badges: start -->

[![](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

<!-- badges: end -->

## inTCRepid

inTCRepid (an eye-rollingly forced contrivance of ‘TCR’, ‘epitope’ and
‘intrepid’, because I needed a package name that wasn’t liable to be
used by someone doing serious work) is an R package implementing a deep
learning framework for predicting linear peptide epitope sequences from
TCR CDR3 sequences. It uses data from
[VDJdb](https://github.com/antigenomics/vdjdb-db) ([Goncharov et
al. 2022, *Nat Methods*](https://doi.org/10.1038/s41592-022-01578-0))
and [IEDB](https://www.iedb.org/) ([Vita et al. 2024, *Nucleic Acids
Research*](https://doi.org/10.1093/nar/gkae1092)) and
[McPAS-TCR](https://friedmanlab.weizmann.ac.il/McPAS-TCR/) ([Tickotsky
et al. 2017.
*Bioinformatics*](https://academic.oup.com/bioinformatics/article/33/18/2924/3803440))
for testing, training, and validation.

**PLEASE NOTE:** This is a <u>curiosity-first pet project and a learning
exercise</u>, and *definitely* shouldn’t be taken seriously as a
reliable inference model for epitope prediction. This is a major, active
field of study in immunoinformatics, and far more advanced and better
tools exist.

If you are looking to perform real predictive analyses for TCRs and/or
epitopes, you may wish to consider any of the following
**non-exhaustive** list of resources and tools:

- [CATCR](https://doi.org/10.1093/bib/bbae210) - to predict TCR
  sequences for novel epitopes;

- [ePytope-TCR](https://doi.org/10.1016/j.xgen.2025.100946) - a
  “simplified interface to TCR-epitope predictors”;

- [ERGO-II](https://github.com/IdoSpringer/ERGO-II) - for prediction of
  TCR and epitope peptide binding;

- [tcrdist3](https://github.com/kmayerb/tcrdist3) - distance measures
  for TCR repertoire analysis;

- [NetTCR-2.0](https://services.healthtech.dtu.dk/services/NetTCR-2.0/) -
  to predict binding probability between TCR CDR3 and MHC-I- bound to
  HLA\*02:01;

- [ImRex](https://github.com/pmoris/ImRex) - unseen epitope TCR
  interaction prediction.

That said, anyone is welcome to play around with inTCRepid - I am always
thrilled to collaborate (especially with people who are actually working
in this space!) - just please keep in mind that this model is
experimental, liable to change whenever I feel like doing so, and the
model outputs should *absolutely not* be treated as anything resembling
biological ground truth without actual validation.

### Model overview

inTCRepid uses a dual-encoder architecture with:

- **Dual chain TCR encoding** (CDR3$\alpha$ + CDR3$\beta$) with V/J
  germline gene embeddings;

- **CNN-based sequence encoders** with multi-scale convolutions (kernel
  sizes 3, 5, and 7);

- **Attention-weighted pooling** for variable-length CDR3 sequences;

- **MHC class/allele** **integration** for epitope-MHC context;

- **Contrastive learning** with cosine similarity and temperature
  scaling;

- **Transfer learning** from human to mouse TCR data with experience
  replay; and

- **Post-hoc calibration** via temperature scaling.

## Installation

``` r
devtools::install_github("amdouek/inTCRepid")
```

## Quick start

### Environment initialisation

Each new R session requires explicit python env setup:

``` r
Sys.setenv(RETICULATE_PYTHON = normalizePath(
  file.path(Sys.getenv("LOCALAPPDATA"), 
            "r-miniconda/envs/tcr_epitope/python.exe"),
  winslash = "/"
))
library(reticulate)
py_config() # Verify environment
```

### Source all scripts (needed until proper package architecture is configured)

``` r
source("R/source_all.R")
```

### Architecture

(Relevant for models V0-V4)

![](images/clipboard-1363855137.jpeg)

### Minimal model execution - IN PROGRESS (epitope inference not yet working)

``` r
library(inTCRepid)
library(reticulate)
library(tidyverse)

# 1. Load pretrained model version
model <- load_model_package(package_dir = "models",
                            package_name = "tcr_epitope_mouse_v0")

# 2. Build epitope reference
epitope_reference <- build_epitope_reference(trainer = model$trainer,
                                             unique_epitopes = model$epitope_reference,
                                             epitope_max_length = 30L)
# 3. Predict epitopes for known CDR3 sequences
query_cdr3 <- c("CASSIRSSYEQYF", "CASSLGQAYEQYF")

predictions <- predict_epitopes(
  query_cdr3 = query_cdr3,
  trainer = model$trainer,
  epitope_reference = epitope_reference,
  top_k = 5
)
```

# Model version history

### Version summary

<table style="width:97%;">
<colgroup>
<col style="width: 7%" />
<col style="width: 36%" />
<col style="width: 17%" />
<col style="width: 12%" />
<col style="width: 13%" />
<col style="width: 9%" />
</colgroup>
<thead>
<tr>
<th>Version</th>
<th>Key Changes</th>
<th>Data sources</th>
<th>Train samples</th>
<th>Epitope Classes</th>
<th>Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td>V0</td>
<td>Mouse-only baseline (VDJdb)</td>
<td>VDJdb</td>
<td>~600</td>
<td>~30</td>
<td>~284,000</td>
</tr>
<tr>
<td>V1</td>
<td>Transfer learning (human + mouse);<br />
4:1 weighting by VDJdb score</td>
<td>VDJdb</td>
<td>~50,000</td>
<td>~100</td>
<td>~284,000</td>
</tr>
<tr>
<td>V2</td>
<td>Aggressive 16:1 score weighting</td>
<td>VDJdb</td>
<td>~50,000</td>
<td>~100</td>
<td>~284,000</td>
</tr>
<tr>
<td>V3</td>
<td>Label smoothing</td>
<td>VDJdb</td>
<td>~50,000</td>
<td>~100</td>
<td>~284,000</td>
</tr>
<tr>
<td>V4</td>
<td>IEDB integration</td>
<td>VDJdb + IEDB</td>
<td>~220,000</td>
<td>~1,940</td>
<td>~1.1M</td>
</tr>
<tr>
<td>V5</td>
<td>Increased capacity; focal loss; Atchley init</td>
<td>VDJdb + IEDB</td>
<td>~220,000</td>
<td>~1,940</td>
<td>~1.1M</td>
</tr>
<tr>
<td>V5.1</td>
<td>EWC implementation</td>
<td>VDJdb + IEDB</td>
<td>~220,000</td>
<td>~1,940</td>
<td>~1.1M</td>
</tr>
<tr>
<td>V6</td>
<td>TRB V/J gene embeddings; 3% experience replay</td>
<td>VDJdb</td>
<td>~70,000</td>
<td>~1,754</td>
<td>~1.14M</td>
</tr>
<tr>
<td>V7</td>
<td>Dual chain (TRA +TRB) integration</td>
<td>VDJdb</td>
<td>~99,000</td>
<td>~1,754</td>
<td>~2.1M</td>
</tr>
<tr>
<td>V8</td>
<td>V7 + IEDB</td>
<td>VDJdb + IEDB</td>
<td>~164,000</td>
<td>3,338</td>
<td>~1.88M</td>
</tr>
<tr>
<td>V8.1</td>
<td>Curated IEDB filtering</td>
<td>VDJdb + IEDB</td>
<td>~69,000</td>
<td>2,862</td>
<td>~1.88M</td>
</tr>
<tr>
<td>V9</td>
<td>McPAS-TCR integration</td>
<td>VDJdb + IEDB + McPAS</td>
<td>~167,000</td>
<td>3,380</td>
<td>~1.88M</td>
</tr>
<tr>
<td>V9.1</td>
<td>MHC class/allele embeddings</td>
<td>VDJdb + IEDB + McPAS</td>
<td>~166,000</td>
<td>3,362</td>
<td>~2.03M</td>
</tr>
<tr>
<td>V9.1.1</td>
<td>Hyperparameter optimisation</td>
<td>VDJdb + IEDB + McPAS</td>
<td>~166,000</td>
<td>3,362</td>
<td>~2.03M</td>
</tr>
<tr>
<td>V10</td>
<td>ESM-2 PLM embeddings (mean-pooled)</td>
<td>VDJdb + IEDB + McPAS</td>
<td>~166,000</td>
<td>3,362</td>
<td>~2.5M</td>
</tr>
</tbody>
</table>

### Performance comparison

#### Phase 1 (Combined Human + Mouse)

| Version | Accuracy | Top-5 | Top-10 | Human Acc | Mouse Acc | ECE (raw → cal) | Temp |
|---------|----------|-------|--------|-----------|-----------|-----------------|------|
| V0      | 89%      | 99%   | \-     | \-        | 89%       | \-              | \-   |
| V1      | \-       | \-    | \-     | \-        | 83.9%     | 0.312           |      |
| V2      | \-       | ~95%  | \-     | \-        | 76.3%     | 0.275→0.110     | 1.71 |
| V3      | \-       | \-    | \-     | \-        | \-        | \-              | \-   |
| V4      | 18.4%    | 37.7% | 49.5%  | 18%       | 30.8%     | 0.646→0.353     | 1.34 |
| V5      | 20.8%    | 41.4% | 52.5%  | 20.5%     | 28.9%     | 0.606→0.316     | 1.49 |
| V5.1    | 21.9%    | 43.0% | 53.2%  | 21.7%     | 30.6%     | 0.510→0.267     | 1.36 |
| V6      | 46.2%    | 58.4% | 63.7%  | 46.8%     | 31.6%     | 0.072           | 1.20 |
| V7      | 56.5%    | 71.9% | 76.2%  | 56.9%     | 46.6%     |                 |      |
| V8      | 39.1%    | 55.9% | 62.7%  |           |           | 0.044→0.061     | 1.06 |
| V8.1    | 23.5%    | 33.7% | 49.8%  |           |           | 0.056→0.050     | 1.11 |
| V9      | 33.9%    | 49.5% | 57.0%  |           |           | 0.069→0.032     | 1.18 |
| V9.1    | 32.1%    | 46.7% | 54.1%  |           |           | 0.020→0.007     | 1.49 |
| V9.1.1  | 34.1%    | 49.7% | 57.0%  |           |           | 0.039→0.008     | 1.64 |
| V10     | 33.2%    | 48.2% | 54.8%  |           |           |                 |      |

#### Phase 2 (Mouse Fine-Tuning)

| Version | Mouse Acc | Mouse Top-5 | Mouse Top-10 | Human Retention | Forgetting rate |
|---------|-----------|-------------|--------------|-----------------|-----------------|
| V4      | 45.4%     |             | 81.2%        | ~3%             | ~83%            |
| V5      | 46.2%     |             | 81.7%        | 3.1%            | ~85%            |
| V5.1    | 47.1%     |             | 81.6%        | 4.5%            | ~79%            |
| V6      | 46.2%     |             | 63.7%        | 46.2%           | 0%              |
| V7      | 51.9%     |             |              | 52.1%           | 4.8%            |
| V8      | 62.6%     |             | 79.8%        |                 |                 |
| V8.1    | 48.4%     |             | 59.3%        |                 |                 |
| V9      | 50.1%     |             | 67.6%        |                 |                 |
| V9.1    | 47.3%     |             | 62.3%        |                 |                 |
| V9.1.1  | 49.6%     |             | 65.3%        |                 |                 |
| V10     | 48.9%     | 63.2%       |              | ~5.4%           | 5.4%            |

#### Pairing effect (V7+)

| Version | Paired (TRA + TRB) | Unpaired (TRB only) | Delta  |
|---------|--------------------|---------------------|--------|
| V7      | 66.6%              | 32.5%               | +34.1% |
| V8      |                    |                     |        |
| V9.1.1  |                    |                     |        |
| V10     | 52.0%              | 24.6%               | +27.4% |

### Architecture summary

| Component | V0-V4 | V5-6 | V7-V9.1.1 | V10 |
|----|----|----|----|----|
| Token embed dim | 64 | 128 | 128 | ESM-2 (640→128) |
| Hidden dim | 128 | 256 | 256 | 256 |
| Output dim | 128 | 256 | 256 | 256 |
| CDR3 encoder | CNN (k=3,5,7) | CNN (k=3,5,7) | CNN (k=3,5,7) | MLP |
| CDR3 pooling | Attention | Attention | Attention | Mean (ESM) |
| Epitope encoder | CNN (k=3,5,7) | CNN (k=3,5,7) | CNN (k=3,5,7) | MLP |
| TRA chains | ✗ | ✗ | ✓ | ✓ |
| V/J embeddings | ✗ | V6: TRB-only | TRA + TRB | TRA+TRB |
| MHC embeddings | ✗ | ✗ | V9.1+ | ✓ |
| Embed init | Random | Atchley | Atchley | ESM-2 (frozen) |
| Loss function | Cross-entropy | Focal (gamma = 2) | Focal (gamma = 2) | Focal (gamma = 2) |

## Model changelog

#### V10

- Integrated ESM-2 PLM (esm2_t30_150M_UR50D, 640-dim)

- Pre-computed embeddings for CDR3a, CDR3b, and epitope sequences (HDF5
  cache)

- Replaced CNN encoders with MLP projections for mean-pooled ESM
  embeddings

  - No improvement over V9.1.1; mean-pooling may lose positional info
    critical for short CDR3 sequences

#### V9.1/V9.1.1

- Added MHC class (4 tokens) and allele (~116 tokens) embeddings

- Epitope-MHC fusion layer captures presentation context

- **V9.1.1** - hyperparameter optimisation for MHC-integrated
  architecture

  - Current best-performing learned-embedded model

#### V9

- Integrated McPAS-TCR data (+~3K entries after deduplication)

#### V8/V8.1

- Added IEDB to V7 dual-chain architecture, and curated IEDB more
  strictly

#### V7

- Dual-chain TCR encoding (CDR3$\alpha$ + CDR3$\beta$)

- TRA V/J gene vocabularies (187 V, 78 J)

- Chain fusion options (concat, gated, attention)

- Paired chains predicted far better than single-chain

#### V6

- TRB V/J germline gene embeddings (115 V, 35 J genes)

- 3% stratified experience replay to mitigate forgetting

#### V5/V5.1

- 4x parameter increase (284K → 1.1M)

- Focal loss (default $\gamma$ =2.0) for class imbalance

- Atchley factor embedding initialisation

- V5.1: EWC regularisation (though had minimal effect)

#### V4

- Integrated IEDB (+190K samples, +1,800 epitope classes)

- Performance drop due to increased task difficulty

#### V0-V3

- Baseline development on VDJdb

- Implemented human → mouse transfer learning

- Score-based sample weighting (16:1 for high-confidence entries)

# Technical stack

- **Language:** R with Python via reticulate

- **Deep learning:** PyTorch 2.5.1 (CUDA 12.1)

- **PLM:** HuggingFace transformers (ESM-2)

- **Python env:** Conda (r-miniconda/envs/tcr_epitope)

- **Hardware:** NVIDIA RTX 2000 Ada (8 GB VRAM)
