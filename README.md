# scLAMP-SR

State-resolved mixture decoding for paired single-cell multi-omics (RNA–ATAC and RNA–ADT).

scLAMP-SR trains modality-specific and shared latent variables, then alternates those latents as inputs to the same mixture decoder. Component families can be chosen by omics layer; this repository implements the paper configuration with zero-inflated negative binomial (ZINB) and zero-inflated Poisson (ZIP) components and gene- and state-dependent mixture weights.

**License:** MIT  
**Contact:** 34520241151650@stu.xmu.edu.cn

This repository contains training, preprocessing, evaluation, and baseline-wrapper code. It does **not** ship standardized `.h5ad` files, train/validation/test splits, model checkpoints, figure CSVs, or manuscript LaTeX.

## Installation

```bash
conda create -n sclamp-sr python=3.8
conda activate sclamp-sr
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The paper analyses used Python 3.8.16, PyTorch 2.4.0 with CUDA 12.4, NumPy 1.24.3, pandas 2.0.3, SciPy 1.10.1, scikit-learn 1.3.2, AnnData 0.9.2, Scanpy 1.9.8, epiScanpy 0.4.0, scvi-tools 0.13.0, and UMAP-learn 0.5.7.

## Data

Download the public source datasets (do not commit them here):

| Dataset | Assay | Source |
| --- | --- | --- |
| GLUE PBMC10k | RNA–ATAC | 10x Genomics PBMC Multiome demonstration data as released with [GLUE](https://doi.org/10.1038/s41587-022-01284-4) |
| Human Cortex | RNA–ATAC | GEO SuperSeries [GSE204684](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE204684) (RNA: GSE204683; ATAC: GSE204682) |
| SLN111 | RNA–ADT | GEO [GSE150599](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150599) |

Standardize local copies, then validate pairing:

```bash
python scripts/standardize_glue_pbmc_pair.py \
  --rna-h5ad /path/to/glue_pbmc10k_rna.h5ad \
  --atac-h5ad /path/to/glue_pbmc10k_atac.h5ad \
  --output-dir dataset/scLAMP_SR_ready/GLUE_PBMC10k

python scripts/standardize_human_cortex_pair.py \
  --raw-dir /path/to/GSE204684/raw \
  --annotation /path/to/human_cortex.h5ad \
  --output-dir dataset/scLAMP_SR_ready/HUMAN_CORTEX_CP

python scripts/convert_scmdc_h5_to_scapi.py \
  --dataset_dir /path/to/citeseq_h5 \
  --output_root dataset/scLAMP_SR_ready \
  --datasets SLN111

python scripts/validate_scapi_ready_pairs.py \
  --root dataset/scLAMP_SR_ready \
  --datasets GLUE_PBMC10k HUMAN_CORTEX_CP SLN111
```

Human Cortex analyses in the paper used the 14,336 nuclei labelled cortical plate. Held-out probability experiments used PBMC and SLN111 only.

## Train scLAMP-SR

Paper settings: 200 epochs, 50 warmup epochs, latent dimension 32, batch size 64, prototype/pseudo-state count 12, auxiliary state CE weight 0.02 from epoch 100 with a 30-epoch ramp, soft-target max probability 0.90.

```bash
python models/scAPI_shared_feature_lambda_continuous_prior_gentle_zi_state_contrast_residual.py \
  --seed 0 \
  --output_dir results/PBMC \
  --run_dir_name seed_0 \
  --rna_h5ad dataset/scLAMP_SR_ready/GLUE_PBMC10k/rna.h5ad \
  --modality2_h5ad dataset/scLAMP_SR_ready/GLUE_PBMC10k/atac.h5ad \
  --modality2_name ATAC \
  --modality2_input_transform tfidf \
  --num_epochs 200 \
  --warmup_epochs 50 \
  --latent_dim 32 \
  --batch_size 64 \
  --early_stopping_patience 0 \
  --latent_state_ce_weight 0.02 \
  --latent_state_soft_target_max_prob 0.90 \
  --latent_state_ramp_epochs 30 \
  --latent_state_start_epoch 100 \
  --prototype_scale 0.8 \
  --prototype_count 12 \
  --state_contrast_clusters 12
```

For SLN111, set `--modality2_name ADT` and `--modality2_input_transform clr`. Repeat `--seed 0 1 2 3 4` for the integration benchmark.

Export encoder means and evaluate pairing/structure metrics:

```bash
python scripts/export_sclamp_embeddings_from_run.py --run-dir results/PBMC/seed_0

python scripts/evaluate_integration_embeddings.py \
  --reference-rna dataset/scLAMP_SR_ready/GLUE_PBMC10k/rna.h5ad \
  --output-dir results/PBMC/evaluation \
  --dataset-name PBMC \
  --embedding scLAMP-SR 0 results/PBMC/seed_0/embeddings.npz
```

## Held-out probability, missing modality, and CD14 analysis

Five mixture-weight variants (`fixed`, `global`, `feature`, `continuous_prior`, `state_resolved`) share the encoder, parameter heads, ZINB/ZIP components, split, and training budget:

```bash
python experiments/run_heldout_probability_ablation.py \
  --dataset_name GLUE_PBMC10k \
  --rna_h5ad dataset/scLAMP_SR_ready/GLUE_PBMC10k/rna.h5ad \
  --modality2_h5ad dataset/scLAMP_SR_ready/GLUE_PBMC10k/atac.h5ad \
  --modality2_name ATAC \
  --modality2_transform tfidf \
  --method state_resolved \
  --seed 0 \
  --output_dir results/heldout/GLUE_PBMC10k
```

The ablation script writes splits locally; they are not stored in this repository. Frozen-latent decoder calibration:

```bash
python experiments/run_precomputed_latent_decoder_calibration.py --help
```

Missing-modality evaluation (pass the held-out checkpoint directory and split file you generated):

```bash
python scripts/evaluate_pbmc_missing_modality_pilot.py \
  --checkpoint-dir path/to/state_resolved/seed_0 \
  --split-path path/to/splits/GLUE_PBMC10k_seed_0.npz \
  --output-dir results/missing_modality/seed_0
```

CD14 program analysis:

```bash
python scripts/analyze_pbmc_cd14_multimodal_states.py --help
```

## Baseline wrappers

```bash
python scripts/run_multivi_pbmc.py --rna-h5ad ... --atac-h5ad ... --output-dir results/MultiVI/seed_0
python scripts/run_cobolt_pbmc.py --rna-h5ad ... --atac-h5ad ... --output-dir results/Cobolt/seed_0
python scripts/run_totalvi_experiment1.py --rna-h5ad ... --adt-h5ad ... --output-dir results/totalVI/seed_0
python scripts/run_midas_experiment1.py --rna-h5ad ... --modality2-h5ad ... --modality2-name ATAC \
  --dataset-name PBMC --output-dir results/MIDAS/seed_0 --midas-root third_party/midas
```

See `third_party/README.md` for MIDAS.

## Tests

```bash
python -m unittest tests.test_count_distributions
```

## Repository layout

```text
models/         scLAMP-SR model and the ZINB/ZIP decoder family
experiments/    held-out likelihood ablation and frozen-latent calibration
scripts/        preprocessing, evaluation, and baseline wrappers
tests/          negative-binomial parameterization checks
```

## Citation

Hu Z, Yang F, Zeng F, Huang S, Yang Z. scLAMP-SR: state-resolved mixture decoding for paired single-cell multi-omics.
