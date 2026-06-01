# HE Neutrophil-Centered Spatial Feature Pipeline

This Markdown notebook provides a compact, GitHub-readable description of the de-identified pipeline.

## 1. Workflow

```text
H&E-stained WSI
    ↓
HoVer-Net inference
    ↓
Classic JSON + neutrophil-focused JSON
    ↓
Merged cell-level JSON
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

The term `immune` replaces the original lymphocyte-oriented label. Macrophage-related final features are excluded. Stromal-related features are consistently named `stromal`.

## 3. Basic usage

```bash
python he_neutrophil_spatial_pipeline.py \
  --base-dir /path/to/project_root \
  --input-wsi-dir /path/to/project_root/svs \
  --hover-dir /path/to/project_root/Hover \
  --classic-model-path /path/to/hovernet_fast_pannuke_type_tf2pytorch.tar \
  --neutrophil-model-path /path/to/hovernet_fast_monusac_type_tf2pytorch.tar \
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
Nsubgraph: NA -> 1
Degrees: NA -> 0
```

## 6. Output

```text
HE_neutrophil_centered_features.csv
```

This file contains one row per sample and only the final neutrophil-centered features.
