# HE Neutrophil-Centered Spatial Feature Pipeline

This Markdown notebook provides a compact, GitHub-readable description of the de-identified pipeline.

This workflow is adapted from the [sc-MTOP framework](https://github.com/fuscc-deep-path/sc_MTOP). It assumes the same general environment and dependency requirements as sc-MTOP, including HoVer-Net inference, OpenSlide support, GPU-based WSI processing, and graph-based spatial feature extraction.

## 1. Workflow

```text
H&E-stained WSI
    ↓
HoVer-Net inference
    ↓
JSON output
    ↓
Cell graph construction
    ↓
Neutrophil-centered spatial feature integration
```

## 2. Final feature scope

The public version keeps only neutrophil-centered interaction features:

```text
Neu-tumor
Neu-neu
Neu-immune
Neu-stromal
```

## 3. Basic usage

```bash
python he_neutrophil_spatial_pipeline.py \
  --base-dir /path/to/project_root \
  --input-wsi-dir /path/to/project_root/svs \
  --hover-dir /path/to/project_root/Hover \
  --modelA-path /path/to/hovernet_fast_pannuke_type_tf2pytorch.tar \
  --modelB-path /path/to/hovernet_fast_monusac_type_tf2pytorch.tar \
  --type-info-path /path/to/type_info.json \
  --gpu 0
```

## 4. Feature integration only

```bash
python he_neutrophil_spatial_pipeline.py \
  --base-dir /path/to/project_root \
  --summary-only
```

## 5. Missing-value rules

```text
minEdgeLength / meanEdgeLength: NA -> 100
```

## 6. Output

```text
HE_neutrophil_centered_features.csv
```

This file contains one row per sample and only the final neutrophil-centered features.
