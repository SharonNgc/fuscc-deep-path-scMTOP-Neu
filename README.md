# HE Neutrophil-Centered Spatial Feature Pipeline

This repository provides a de-identified Python pipeline for extracting neutrophil-centered spatial features from H&E-stained whole-slide images (WSIs).

This work is adapted from the [sc-MTOP framework](https://github.com/fuscc-deep-path/sc_MTOP). The same general software environment, dependency structure, and runtime assumptions as sc-MTOP are expected, including HoVer-Net-based nuclear segmentation/classification, OpenSlide support, GPU-based WSI inference, and graph-based spatial feature extraction. Please refer to the original sc-MTOP repository for baseline installation and environment requirements.

The adapted workflow combines HoVer-Net-based cell detection/classification, merged cell-level JSON outputs, cell-graph construction, and final feature integration.

## Overview

The pipeline contains four major stages:

1. **F1: WSI cell detection and classification**  
   Run a classic HoVer-Net model and a neutrophil-focused HoVer-Net model on each WSI.

2. **F2: JSON merging**  
   Merge the two cell-level JSON outputs and remove overlapping duplicated detections.

3. **F3: Cell graph construction**  
   Build a spatial graph of detected cells and export per-cell graph features.

4. **F4: Feature integration**  
   Export a sample-level neutrophil-centered feature matrix.

The final output is restricted to the following neutrophil-centered interaction groups:

- `Neu-tumor`
- `Neu-neu`
- `Neu-immune`
- `Neu-stromal`

Here, `immune` is used as the standardized term for the original lymphocyte-related label, and `stromal` is used consistently instead of connective. Macrophage-related and residual non-target cell features are intentionally excluded from the final output.

## Expected folder structure

```text
project_root/
├── svs/
│   ├── sample_001.svs
│   ├── sample_002.svs
│   └── ...
├── Hover/
│   └── ...
├── WSIGraph_Alter_ly.py
├── utils_xml.py
└── he_neutrophil_spatial_pipeline.py
```

## Required external components

This script assumes the same baseline dependencies as sc-MTOP and additionally expects the following project-specific components to be available locally:

- HoVer-Net inference code
- HoVer-Net model weights
- `WSIGraph_Alter_ly.py`
- `utils_xml.py`
- OpenSlide, if required by your platform

Python packages used by the script include:

```bash
pip install numpy pandas torch shapely rtree orjson
```

## Run the full pipeline

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

On Windows, you may also need:

```bash
  --openslide-bin C:/path/to/openslide/bin
```

## Run feature integration only

If F3 graph feature folders already exist, run:

```bash
python he_neutrophil_spatial_pipeline.py \
  --base-dir /path/to/project_root \
  --summary-only
```

## Output

The final output file is:

```text
HE_neutrophil_centered_features.csv
```

The feature table includes:

```text
Sample
Neu-tumor_minEdgeLength
Neu-tumor_meanEdgeLength
Neu-tumor_Nsubgraph
Neu-tumor_Degrees
Neu-neu_minEdgeLength
Neu-neu_meanEdgeLength
Neu-neu_Nsubgraph
Neu-neu_Degrees
Neu-immune_minEdgeLength
Neu-immune_meanEdgeLength
Neu-immune_Nsubgraph
Neu-immune_Degrees
Neu-stromal_minEdgeLength
Neu-stromal_meanEdgeLength
Neu-stromal_Nsubgraph
Neu-stromal_Degrees
```

## Missing-value handling

For graph-derived variables, missing values are handled as follows:

- `minEdgeLength` and `meanEdgeLength`: `NA -> 100`
- `Nsubgraph`: `NA -> 1`
- `Degrees`: `NA -> 0`

These rules are applied before calculating sample-level mean feature values.

## Notes

This public version removes local private paths, personal file names, and nonessential cohort-specific references. The labels are standardized to tumor, immune, neutrophil, and stromal categories for clearer reuse.
