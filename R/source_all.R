# source_all.R - Load all TCR-epitope model functions (V10)
# Source in order of dependencies

# --- Environment ---
source("R/env_setup_config.R")

# --- Data Acquisition ---
source("R/vdjdb_download.R")
source("R/vdjdb_preprocessing.R")
source("R/iedb_preprocessing.R")
source("R/mcpastcr_preprocessing.R")

# --- Encoding (order matters) ---
source("R/sequence_encoding.R")
source("R/vj_gene_encoding.R")
source("R/mhc_encoding.R")              # Must be before paired_chain_preprocessing
source("R/paired_chain_preprocessing.R") # Uses MHC encoding functions
source("R/esm_embeddings.R")            # ESM-2 embedding infrastructure

# --- Model (order matters) ---
source("R/model_architecture_v10.R")    # V10 model classes
source("R/model_evaluation.R")          # Must be before training (provides metrics)
source("R/model_training_v10.R")        # Uses evaluation functions

# --- Inference & Persistence ---
source("R/inference.R")
source("R/model_persistence.R")

# --- Calibration ---
source("R/calibration.R")
source("R/postcalibration_analysis.R")

message("All TCR-epitope model scripts loaded (V10).")
