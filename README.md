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
Research*](https://doi.org/10.1093/nar/gkae1092)) for testing, training,
and validation.

**PLEASE NOTE:** This is a <u>curiosity-first pet project and a learning
exercise</u>, and *definitely* shouldn’t be taken seriously as a
reliable inference model for epitope prediction. This is a major, active
field of study in immunoinformatics, and far more advanced and better
tools exist. If you are looking to perform real predictive analyses for
TCRs and/or epitopes, you may wish to consider any of the following
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

- **CNN-based sequence encoders** with multi-scale convolutions (kernel
  sizes 3, 5, and 7),

- **Attention-weighted pooling** for CDR3 sequences of variable length,

- **Contrastive learning** with cosine similarity and temperature
  scaling,

- **Transfer learning** from human to mouse TCR data,

- **Post-hoc calibration** for reliable probability estimates.

### Architecture

(Relevant for models V0-V4)

![](images/clipboard-1363855137.jpeg)

## Installation

``` r
devtools::install_github("amdouek/inTCRepid")
```

## Quick start

### Environment initialisation

Each new R session requires you to explicitly set `RETICULATE_PYTHON` to
the `tcr_epitope` python env:

``` r
Sys.setenv(RETICULATE_PYTHON = normalizePath(
  file.path(Sys.getenv("LOCALAPPDATA"), 
            "r-miniconda/envs/tcr_epitope/python.exe"),
  winslash = "/"
))

# Verify environment
py_config()
```

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

# Model version performance logs

### Versioning overview

<table style="width:97%;">
<colgroup>
<col style="width: 8%" />
<col style="width: 41%" />
<col style="width: 12%" />
<col style="width: 13%" />
<col style="width: 10%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr>
<th>Version</th>
<th>Key Changes</th>
<th>Training Data</th>
<th>Epitope Classes</th>
<th>Parameters</th>
<th>Status</th>
</tr>
</thead>
<tbody>
<tr>
<td>V0</td>
<td>Mouse-only baseline (VDJdb)</td>
<td>~600</td>
<td>~30</td>
<td>~284,000</td>
<td>Superseded</td>
</tr>
<tr>
<td>V1</td>
<td>Implementation of transfer learning (human + mouse)<br />
Exponential weighting by VDJdb score (4:1)</td>
<td>~50,000</td>
<td>~100</td>
<td>~284,000</td>
<td>Superseded</td>
</tr>
<tr>
<td>V2</td>
<td>Aggressive weighting (16:1)</td>
<td>~50,000</td>
<td>~100</td>
<td>~284,000</td>
<td>Superseded</td>
</tr>
<tr>
<td>V2 + cal</td>
<td>Implemented temperature calibration (T = 1.71)</td>
<td>~50,000</td>
<td>~100</td>
<td>~284,000</td>
<td>Legacy best</td>
</tr>
<tr>
<td>V3a</td>
<td>Label smoothing (<span class="math inline"><em>ϵ</em></span> =
0.1)</td>
<td>~50,000</td>
<td>~100</td>
<td>~284,000</td>
<td>Rejected</td>
</tr>
<tr>
<td>V3b</td>
<td>Label smoothing (<span class="math inline"><em>ϵ</em></span> =
0.04)</td>
<td>~50,000</td>
<td>~100</td>
<td>~284,000</td>
<td>Rejected</td>
</tr>
<tr>
<td>V4</td>
<td>Integrated data from IEDB (+190K)</td>
<td>~220,000</td>
<td>~1,940</td>
<td>~1,100,000</td>
<td>Superseded</td>
</tr>
<tr>
<td>V5</td>
<td>Increased capacity; focal loss; Atchley init</td>
<td>~220,000</td>
<td>~1,940</td>
<td>~1,100,000</td>
<td>Superseded</td>
</tr>
<tr>
<td>V5.1</td>
<td>EWC</td>
<td>~220,000</td>
<td>~1,940</td>
<td>~1,100,000</td>
<td>Superseded</td>
</tr>
</tbody>
</table>

### Version architecture

| Version | Token Embed Dim | Hidden Dim | Output Dim | Params | CDR3 encoder convs | Epitope encoder convs | Attention | Pooling (CDR3) | Pooling (epitope) | Embed init | Loss func |
|----|----|----|----|----|----|----|----|----|----|----|----|
| V0-V4 | 64 | 128 | 128 | ~284K | k=3,5,7 | k=3,5 | Single-head | Attn-weighted | Max+Avg | Random | Cross-entropy |
| V5-V5.1 | 128 | 256 | 256 | ~1.1M | k=3,5,7 | k=3,5 | Single-head | Attn-weighted | Max+Avg | Atchley (first 5 dims) | Focal |

### Phase 1 Performance (V0-3, VDJdb only)

| Version | Training data | Score 3 Accuracy | Overall Accuracy | Top-5 Accuracy | Mouse Accuracy | ECE (raw) | ECE (cal) | Temp | Notes |
|----|----|----|----|----|----|----|----|----|----|
| V0 | ~600 | 89% | 89% | 99% | 89% |  |  |  | Tiny dataset, severe overfit |
| V1 | ~50K | 72.5% |  |  | 83.9% | 0.312 |  |  | Baseline for transfer learning |
| V2 | ~50K | 80.8% |  | ~95% | 76.3% | 0.275 |  |  |  |
| V2 + cal | ~50K | 80.8% |  | ~95% | 76.3% |  | 0.110 | 1.71 | Best pre-IEDB |
| V3a | ~50K | 69.8% |  |  |  | ~0 |  | ~1 | Rejected |
| V3b | ~50K | 86.9% |  |  |  |  |  |  | Catastrophic forgetting in Phase 2 |

### Phase 1 Performance (V4-5, VDJdb + IEDB)

<table>

<colgroup>

<col style="width: 19%" />

<col style="width: 9%" />

<col style="width: 7%" />

<col style="width: 7%" />

<col style="width: 9%" />

<col style="width: 7%" />

<col style="width: 8%" />

<col style="width: 7%" />

<col style="width: 8%" />

<col style="width: 6%" />

<col style="width: 5%" />

<col style="width: 8%" />

<col style="width: 12%" />

<col style="width: 7%" />

</colgroup>

<thead>

<tr>

<th>

<p>

Version
</p>

</th>

<th>

<p>

Overall Acc
</p>

</th>

<th>

<p>

Top-3 Acc
</p>

</th>

<th>

<p>

Top-5 Acc
</p>

</th>

<th>

<p>

Top-10 Acc
</p>

</th>

<th>

<p>

Score 0 Acc
</p>

</th>

<th>

<p>

Score 1 Acc
</p>

</th>

<th>

<p>

Score 2 Acc
</p>

</th>

<th>

<p>

Score 3 Acc
</p>

</th>

<th>

<p>

Hum Acc
</p>

</th>

<th>

<p>

Ms Acc
</p>

</th>

<th>

<p>

Macro F1
</p>

</th>

<th>

<p>

Weighted F1
</p>

</th>

<th>

<p>

Val Loss
</p>

</th>

</tr>

</thead>

<tbody>

<tr>

<td>

<p>

V4
</p>

<p>

Tr: 154,800; Ep: 1,940; Par: ~284K
</p>

</td>

<td>

<p>

18.4%
</p>

</td>

<td>

<p>

30.6%
</p>

</td>

<td>

<p>

37.7%
</p>

</td>

<td>

<p>

49.5%
</p>

</td>

<td>

<p>

11.4%
</p>

</td>

<td>

<p>

4.7%
</p>

</td>

<td>

<p>

3.0%
</p>

</td>

<td>

<p>

23.2%
</p>

</td>

<td>

<p>

18.0%
</p>

</td>

<td>

<p>

30.8%
</p>

</td>

<td>

<p>

0.011
</p>

</td>

<td>

<p>

0.151
</p>

</td>

<td>

<p>

2.89
</p>

</td>

</tr>

<tr>

<td>

<p>

V5
</p>

<p>

Tr: 154,800; Ep: 1,940; Par: ~1.1M
</p>

</td>

<td>

<p>

20.8%
</p>

</td>

<td>

<p>

34.3%
</p>

</td>

<td>

<p>

41.4%
</p>

</td>

<td>

<p>

52.5%
</p>

</td>

<td>

<p>

22.7%
</p>

</td>

<td>

<p>

5.5%
</p>

</td>

<td>

<p>

3.1%
</p>

</td>

<td>

<p>

22.8%
</p>

</td>

<td>

<p>

20.5%
</p>

</td>

<td>

<p>

28.9%
</p>

</td>

<td>

<p>

0.016
</p>

</td>

<td>

<p>

0.190
</p>

</td>

<td>

<p>

0.2
</p>

</td>

</tr>

<tr>

<td>

<p>

V5.1
</p>

<p>

Tr: 154,800; Ep: 1,940; Par: ~1.1M
</p>

</td>

<td>

<p>

21.9%
</p>

</td>

<td>

<p>

36.0%
</p>

</td>

<td>

<p>

43.0%
</p>

</td>

<td>

<p>

53.2%
</p>

</td>

<td>

<p>

23.9%
</p>

</td>

<td>

<p>

4.7%
</p>

</td>

<td>

<p>

3.6%
</p>

</td>

<td>

<p>

24.1%
</p>

</td>

<td>

<p>

21.7%
</p>

</td>

<td>

<p>

30.6%
</p>

</td>

<td>

<p>

0.018
</p>

</td>

<td>

<p>

0.2
</p>

</td>

<td>

<p>

2.43
</p>

</td>

</tr>

</tbody>

</table>

### Phase 2 Performance (Mouse Fine-Tuning)

#### Mouse test set eval

<table>

<colgroup>

<col style="width: 8%" />

<col style="width: 12%" />

<col style="width: 7%" />

<col style="width: 7%" />

<col style="width: 9%" />

<col style="width: 7%" />

<col style="width: 8%" />

<col style="width: 7%" />

<col style="width: 8%" />

<col style="width: 8%" />

<col style="width: 12%" />

<col style="width: 7%" />

</colgroup>

<thead>

<tr>

<th>

<p>

Version
</p>

</th>

<th>

<p>

Overall Acc
</p>

</th>

<th>

<p>

Top-3 Acc
</p>

</th>

<th>

<p>

Top-5 Acc
</p>

</th>

<th>

<p>

Top-10 Acc
</p>

</th>

<th>

<p>

Score 0 Acc
</p>

</th>

<th>

<p>

Score 1 Acc
</p>

</th>

<th>

<p>

Score 2 Acc
</p>

</th>

<th>

<p>

Score 3 Acc
</p>

</th>

<th>

<p>

Macro F1
</p>

</th>

<th>

<p>

Weighted F1
</p>

</th>

<th>

<p>

Val Loss
</p>

</th>

</tr>

</thead>

<tbody>

<tr>

<td>

<p>

V4
</p>

<p>

Tr: 4,079
</p>

</td>

<td>

<p>

45.4%
</p>

</td>

<td>

<p>

66%
</p>

</td>

<td>

<p>

72.1%
</p>

</td>

<td>

<p>

81.2%
</p>

</td>

<td>

<p>

0%
</p>

</td>

<td>

<p>

\-
</p>

</td>

<td>

<p>

19.2%
</p>

</td>

<td>

<p>

52.8%
</p>

</td>

<td>

<p>

0.117
</p>

</td>

<td>

<p>

0.401
</p>

</td>

<td>

<p>

2.02
</p>

</td>

</tr>

<tr>

<td>

<p>

V5
</p>

<p>

Tr: 4,079
</p>

</td>

<td>

<p>

46.2%
</p>

</td>

<td>

<p>

62.6%
</p>

</td>

<td>

<p>

71.6%
</p>

</td>

<td>

<p>

81.7%
</p>

</td>

<td>

<p>

0%
</p>

</td>

<td>

<p>

\-
</p>

</td>

<td>

<p>

12.8%
</p>

</td>

<td>

<p>

54.5%
</p>

</td>

<td>

<p>

0.159
</p>

</td>

<td>

<p>

0.425
</p>

</td>

<td>

<p>

1.36
</p>

</td>

</tr>

<tr>

<td>

<p>

V5.1
</p>

<p>

Tr: 4,079
</p>

</td>

<td>

<p>

47.1%
</p>

</td>

<td>

<p>

64.1%
</p>

</td>

<td>

<p>

72.1%
</p>

</td>

<td>

<p>

81.6%
</p>

</td>

<td>

<p>

\-
</p>

</td>

<td>

<p>

\-
</p>

</td>

<td>

<p>

17.9%
</p>

</td>

<td>

<p>

55%
</p>

</td>

<td>

<p>

0.169
</p>

</td>

<td>

<p>

0.427
</p>

</td>

<td>

<p>

1.3
</p>

</td>

</tr>

</tbody>

</table>

#### Full test set eval (forgetting assessment)

| Version | Overall Acc | Top-3 Acc | Top-5 Acc | Top-10 Acc | Score 0 Acc | Score 1 Acc | Score 2 Acc | Score 3 Acc | Hum Acc (post-P2) | Ms Acc (post-P2) | Macro F1 | Weighted F1 |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| V4 | 4.7% | 12.5% | 16.6% | 23.5% | 0.2% | 2.2% | 0.9% | 6.9% | ~3% | 52.8% | 0.006 | 0.049 |
| V5 | 4.2% | 20.4% | 26.2% | 35.5% | 0.6% | 2.9% | 0.7% | 5.9% | 3.1% | 46.2% | 0.01 | 0.043 |
| V5.1 | 5.6% | 22.5% | 30.4% | 41.0% | 2.1% | 3.3% | 0.8% | 7.5% | 4.5% | 47.1% | 0.01 | 0.065 |

#### Catastrophic forgetting

| Version | Hum Acc (P1) | Hum Acc (P2) | Abs Forgetting | Rel Forgetting | EWC $\lambda$ | EWC effect |
|----|----|----|----|----|----|----|
| V4 | 18% | ~3% | ~15% | ~83% | N/A | N/A |
| V5 | 20.5% | 3.1% | 17.4% | 85% | N/A | N/A |
| V5.1 | 21.7% | 4.5% | 17.2% | 79% | 1000 | Minimal |

### Calibration metrics

| Version | Opt Temp | ECE (pre) | ECE (post) | ECE improvement | MCE (pre) | MCE (post) | Brier (pre) | Brier (post) |
|----|----|----|----|----|----|----|----|----|
| V2+cal | 1.71 | 0.275 | 0.11 | 60% |  |  |  |  |
| V4 | 1.34 | 0.646 | 0.353 | 45% | 0.868 | 0.672 | 1.461 | 1.142 |
| V5 | 1.4926 | 0.606 | 0.3155 | 47.9% | 0.9336 | 0.9365 | 1.4136 | 1.1097 |
| V5.1 | 1.3565 | 0.5097 | 0.2674 | 47.5% | 0.7711 | 0.4792 | 1.2674 | 1.0537 |

## Model changelog

#### V0

- Baseline (VDJdb mouse-only), tiny dataset

- Severe overfitting, not generalisable

#### V1

- Implemented transfer learning (pretraining on human VDJdb data, then
  fine-tuned on mouse VDJdb)

- Implemented exponential (4:1) score weighting to increase effect of
  TCR-epitope pairs with experimental validation (VDJdb score 3).

#### V2

- Made score weighting more aggressive (16:1), leading to +8.3% Score 3
  accuracy compared to V1

#### V2+cal

- Implemented post-hoc temperature scaling. Optimal temp for V2 was
  1.71, improving ECE from 0.275 to 0.110

#### V3a/b

- Implemented label smoothing with $\epsilon = 0.1$ (a) and
  $\epsilon = 0.04$ (b).

- Caused accuracy drop and catastrophic forgetting in phase 2.

#### V4

- Integrated data from IEDB (+190K samples, +1800 epitope classes)

- Accuracy dropped due to increased task difficulty.

#### V5

- Increased capacity from 284K to 1.1M

- Implemented Atchley factor embedding initialisation

- Replaced cross-entropy with focal loss (default $\gamma = 2.0$)

- Parameterised label smoothing for future testing iterations

- Added hooks for EWC regularisation

- Added optional BLOSUM regularisation

- Implemented PLM embedding hooks for future ESM-2/ProtBERT integration

#### V5.1

- Implemented EWC for state transfer between P1 and P2 trainers
