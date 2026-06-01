#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HE neutrophil-centered spatial feature pipeline for whole-slide images.

This script is a de-identified, GitHub-ready version of a local HE analysis
pipeline. It runs HoVer-Net-based WSI inference, merges classic cell detection
and neutrophil-focused detection outputs, constructs a cell graph, and exports
neutrophil-centered spatial interaction features.

Final exported interaction variables are restricted to:
    - Neu-tumor
    - Neu-neu
    - Neu-immune
    - Neu-stromal

Notes
-----
1. This script assumes that the HoVer-Net repository and required graph utility
   modules are available locally.
2. Replace placeholder paths by command-line arguments.
3. "immune" is used as the standardized term for the original lymphocyte label.
4. Macrophage-related final features are intentionally excluded.
5. "stromal" is used consistently instead of "connective".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from multiprocessing import Process, freeze_support
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import orjson
import pandas as pd
import torch
from rtree import index
from shapely.geometry import Polygon


# =============================================================================
# 1. Global definitions
# =============================================================================

VALID_WSI_EXTS = [".svs", ".ndpi", ".tif", ".tiff", ".mrxs"]

# Internal cell-type codes used by the downstream graph module.
CELL_TYPE_CODE = {
    "tumor": 1,      # T
    "immune": 2,    # originally lymph/L; standardized here as immune
    "macro": 3,     # excluded from the final feature table
    "neu": 4,       # neutrophils
    "stromal": 5,   # standardized from connective/stroma
    "other": 6,
}

FINAL_INTERACTION_SPECS = [
    # output_group, dataframe_key, graph_prefix
    ("Neu-tumor", "N", "Graph_T-N"),
    ("Neu-neu", "N", "Graph_N-N"),
    ("Neu-immune", "N", "Graph_L-N"),
    ("Neu-stromal", "N", "Graph_N-S"),
]

GRAPH_METRICS = [
    "minEdgeLength",
    "meanEdgeLength",
    "Nsubgraph",
    "Degrees",
]


# =============================================================================
# 2. Generic utilities
# =============================================================================

def ensure_dir(path: str | Path) -> None:
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_sample_name_from_wsi(wsi_path: str | Path) -> str:
    """Return the sample name by removing the WSI file extension."""
    return Path(wsi_path).stem


def get_all_wsi_files(input_wsi_dir: str | Path) -> List[Path]:
    """Collect all supported WSI files from the input directory."""
    input_wsi_dir = Path(input_wsi_dir)
    files = [
        p for p in input_wsi_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_WSI_EXTS
    ]
    return sorted(files)


def read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    """Read a CSV file; return an empty DataFrame if the file is missing or empty."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def prepare_single_wsi_input(
    wsi_path: str | Path,
    sample_name: str,
    single_wsi_tmp_dir: str | Path,
) -> Path:
    """
    Create a temporary single-WSI input folder.

    HoVer-Net WSI inference is run on one slide at a time in this pipeline.
    A hard link is preferred to avoid copying large WSI files; if it fails,
    the file is copied.
    """
    wsi_path = Path(wsi_path)
    single_input_dir = Path(single_wsi_tmp_dir) / sample_name

    if single_input_dir.exists():
        shutil.rmtree(single_input_dir)
    ensure_dir(single_input_dir)

    dst = single_input_dir / wsi_path.name

    try:
        os.link(wsi_path, dst)
    except Exception:
        shutil.copy2(wsi_path, dst)

    return single_input_dir


def normalize_bbox(cell_info: Dict) -> Dict:
    """
    Convert bbox from [minx, miny, maxx, maxy] to [[minx, miny], [maxx, maxy]].

    If bbox is already in the expected two-point format, it is left unchanged.
    """
    bbox = cell_info.get("bbox", None)

    if (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(x, (int, float)) for x in bbox)
    ):
        minx, miny, maxx, maxy = bbox
        cell_info["bbox"] = [[minx, miny], [maxx, maxy]]

    return cell_info


# =============================================================================
# 3. Environment preparation
# =============================================================================

def prepare_environment(
    base_dir: str | Path,
    input_wsi_dir: str | Path,
    hover_dir: str | Path,
    openslide_bin: Optional[str | Path],
    output_dirs: Iterable[str | Path],
) -> None:
    """Prepare DLL paths, Python paths, and output folders."""
    base_dir = Path(base_dir)
    input_wsi_dir = Path(input_wsi_dir)
    hover_dir = Path(hover_dir)

    if openslide_bin:
        openslide_bin = Path(openslide_bin)
        if openslide_bin.exists():
            os.add_dll_directory(str(openslide_bin))
        else:
            print(f"Warning: OpenSlide binary folder does not exist: {openslide_bin}")

    if str(hover_dir) not in sys.path:
        sys.path.append(str(hover_dir))

    ensure_dir(base_dir)

    for d in output_dirs:
        ensure_dir(d)

    if not input_wsi_dir.exists():
        raise FileNotFoundError(f"Input WSI folder does not exist: {input_wsi_dir}")


# =============================================================================
# 4. F1: HoVer-Net WSI inference
# =============================================================================

def run_hovernet_wsi(
    input_dir: str | Path,
    output_dir: str | Path,
    model_path: str | Path,
    nr_types: int,
    hover_dir: str | Path,
    openslide_bin: Optional[str | Path] = None,
    type_info_path: Optional[str | Path] = None,
    gpu: str = "0",
    model_mode: str = "fast",
    nr_inference_workers: int = 8,
    nr_post_proc_workers: int = 0,
    batch_size: int = 16,
    cache_path: str | Path = "cache_hovernet",
    proc_mag: int = 40,
    ambiguous_size: int = 128,
    chunk_shape: int = 4096,
    tile_shape: int = 2048,
    save_thumb: bool = True,
    save_mask: bool = True,
) -> None:
    """Run HoVer-Net WSI inference on all WSI files in a given folder."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_path = Path(model_path)
    hover_dir = Path(hover_dir)
    cache_path = Path(cache_path)

    if openslide_bin:
        openslide_bin = Path(openslide_bin)
        if openslide_bin.exists():
            os.add_dll_directory(str(openslide_bin))

    if str(hover_dir) not in sys.path:
        sys.path.append(str(hover_dir))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    ensure_dir(output_dir)
    ensure_dir(cache_path)

    print("=" * 90)
    print("Start HoVer-Net WSI inference")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model path      : {model_path}")
    print(f"Number of types : {nr_types}")
    print(f"GPU             : {gpu}")
    print("=" * 90)

    nr_gpus = torch.cuda.device_count()
    if nr_gpus == 0:
        raise RuntimeError(
            f"No CUDA GPU detected. Please check CUDA_VISIBLE_DEVICES={gpu}."
        )

    method_args = {
        "method": {
            "model_args": {
                "nr_types": int(nr_types) if int(nr_types) > 0 else None,
                "mode": model_mode,
            },
            "model_path": str(model_path),
        },
        "type_info_path": None if not type_info_path else str(type_info_path),
    }

    run_args = {
        "batch_size": int(batch_size) * nr_gpus,
        "nr_inference_workers": int(nr_inference_workers),
        "nr_post_proc_workers": int(nr_post_proc_workers),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "presplit_dir": None,
        "input_mask_dir": "",
        "cache_path": str(cache_path),
        "proc_mag": int(proc_mag),
        "ambiguous_size": int(ambiguous_size),
        "chunk_shape": int(chunk_shape),
        "tile_shape": int(tile_shape),
        "save_thumb": save_thumb,
        "save_mask": save_mask,
    }

    if model_mode == "fast":
        run_args["patch_input_shape"] = 256
        run_args["patch_output_shape"] = 164
    else:
        run_args["patch_input_shape"] = 270
        run_args["patch_output_shape"] = 80

    from Hover.infer.wsi import InferManager

    infer = InferManager(**method_args)
    infer.process_wsi_list(run_args)

    torch.cuda.empty_cache()

    print("=" * 90)
    print(f"Finished HoVer-Net WSI inference: {output_dir}")
    print("=" * 90)


# =============================================================================
# 5. F2: Merge classic and neutrophil-focused JSON outputs
# =============================================================================

def merge_one_json(
    classic_json_path: str | Path,
    neutrophil_json_path: str | Path,
    output_json_path: str | Path,
    overlap_threshold: float = 0.05,
) -> int:
    """
    Merge a classic HoVer-Net JSON with a neutrophil-focused HoVer-Net JSON.

    The classic model is used as the base. Immune/neutrophil/stromal-related
    detections from the neutrophil-focused model can replace overlapping base
    detections when the overlap ratio exceeds ``overlap_threshold``.

    The output keeps cell types 1-6 for graph-module compatibility. Macrophage-
    related final features are excluded later.
    """
    classic_json_path = Path(classic_json_path)
    neutrophil_json_path = Path(neutrophil_json_path)
    output_json_path = Path(output_json_path)

    with classic_json_path.open("rb") as f:
        ori_data = orjson.loads(f.read())

    with neutrophil_json_path.open("rb") as f:
        neu_data = orjson.loads(f.read())

    ori_ids = [int(cell_id) for cell_id in ori_data["nuc"].keys()]
    max_ori_id = max(ori_ids) if ori_ids else 0
    id_offset = max_ori_id + 1

    merged_data = {"mag": ori_data.get("mag", None), "nuc": {}}

    # Re-map classic model labels following the original implementation.
    type_mapping_ori = {
        3: CELL_TYPE_CODE["stromal"],
        4: CELL_TYPE_CODE["other"],
    }

    ori_nuc = {}
    for cell_id_str, cell_info in ori_data["nuc"].items():
        cell_id = int(cell_id_str)
        cell_type = cell_info.get("type", None)
        if cell_type in type_mapping_ori:
            cell_info["type"] = type_mapping_ori[cell_type]
        ori_nuc[cell_id] = cell_info

    ori_data["nuc"] = ori_nuc

    ori_idx = index.Index()
    ori_cells = {}

    for cell_id, cell_info in ori_data["nuc"].items():
        contour = cell_info.get("contour", None)
        if not (isinstance(contour, list) and len(contour) >= 3):
            continue
        try:
            polygon = Polygon(contour)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue
            minx, miny, maxx, maxy = polygon.bounds
            if minx <= maxx and miny <= maxy:
                ori_cells[cell_id] = {"cell_info": cell_info, "polygon": polygon}
                ori_idx.insert(cell_id, (minx, miny, maxx, maxy))
        except Exception:
            continue

    for cell_id, cell_info in ori_data["nuc"].items():
        merged_data["nuc"][cell_id] = cell_info

    # Types 2/3/4 are retained for compatibility; macrophage final features are excluded.
    for neu_id_str, neu_cell_info in neu_data["nuc"].items():
        neu_type = neu_cell_info.get("type", None)
        if neu_type not in [2, 3, 4]:
            continue

        contour = neu_cell_info.get("contour", None)
        if not (isinstance(contour, list) and len(contour) >= 3):
            continue

        try:
            neu_polygon = Polygon(contour)
            if not neu_polygon.is_valid:
                neu_polygon = neu_polygon.buffer(0)
            if neu_polygon.is_empty:
                continue

            neu_area = neu_polygon.area
            if neu_area <= 0:
                continue

            minx, miny, maxx, maxy = neu_polygon.bounds
            possible_overlap_ids = list(ori_idx.intersection((minx, miny, maxx, maxy)))

            for ori_id in possible_overlap_ids:
                ori_polygon = ori_cells[ori_id]["polygon"]
                if neu_polygon.intersects(ori_polygon):
                    intersection_area = neu_polygon.intersection(ori_polygon).area
                    overlap_ratio = intersection_area / neu_area
                    if overlap_ratio > overlap_threshold and ori_id in merged_data["nuc"]:
                        del merged_data["nuc"][ori_id]

            new_neu_id = int(neu_id_str) + id_offset
            merged_data["nuc"][new_neu_id] = neu_cell_info

        except Exception:
            continue

    final_nuc = {}
    new_cell_id = 1
    for _, cell_info in merged_data["nuc"].items():
        cell_type = cell_info.get("type", None)
        if cell_type is None:
            continue
        if 1 <= cell_type <= 6:
            cell_info = normalize_bbox(cell_info)
            final_nuc[str(new_cell_id)] = cell_info
            new_cell_id += 1

    merged_data["nuc"] = final_nuc
    ensure_dir(output_json_path.parent)

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f)

    return len(final_nuc)


# =============================================================================
# 6. F3: Cell graph construction and per-class feature export
# =============================================================================

def run_one_graph_feature_extraction(
    json_path: str | Path,
    wsi_path: str | Path,
    output_path: str | Path,
    xml_path: Optional[str | Path] = None,
    distance_threshold: int = 100,
    graph_level: int = 0,
    k_neighbors: int = 5,
) -> None:
    """
    Build a cell graph and export per-class feature files.

    Output files:
        <sample>_Feats_T.csv  : tumor cells
        <sample>_Feats_I.csv  : immune cells, originally lymph/L
        <sample>_Feats_N.csv  : neutrophils
        <sample>_Feats_S.csv  : stromal cells
        <sample>_Feats_O.csv  : other cells
        <sample>_Edges.csv    : graph edges

    Macrophage-specific feature files are not exported in this public version.
    """
    from WSIGraph_Alter_ly import constructGraphFromDict
    from utils_xml import get_windows

    json_path = Path(json_path)
    wsi_path = Path(wsi_path)
    output_path = Path(output_path)
    xml_path = Path(xml_path) if xml_path else None
    sample_name = json_path.stem

    with json_path.open("r", encoding="utf-8") as fp:
        print(f"{' Loading merged JSON ':=^90s}")
        nucleus_info = json.load(fp)

    global_graph, edge_info = constructGraphFromDict(
        str(wsi_path), nucleus_info, distance_threshold, k_neighbors, graph_level
    )

    vertex_dataframe = global_graph.get_vertex_dataframe()

    if xml_path is not None:
        centroid = np.array(vertex_dataframe["Centroid"].tolist())
        window_bbox = np.array(get_windows(str(xml_path)))
        index_mat = np.zeros((len(centroid), len(window_bbox)), dtype=np.bool_)
        for i in range(len(window_bbox)):
            index_mat[:, i] = (
                (window_bbox[i, 0, 0] < centroid[:, 0])
                & (centroid[:, 0] < window_bbox[i, 1, 0])
            ) & (
                (window_bbox[i, 0, 1] < centroid[:, 1])
                & (centroid[:, 1] < window_bbox[i, 1, 1])
            )
        index_x, _ = np.where(index_mat)
        vertex_dataframe = vertex_dataframe.iloc[index_x]

    output_cell_types = {
        "T": [CELL_TYPE_CODE["tumor"]],
        "I": [CELL_TYPE_CODE["immune"]],
        "N": [CELL_TYPE_CODE["neu"]],
        "S": [CELL_TYPE_CODE["stromal"]],
        "O": [CELL_TYPE_CODE["other"]],
    }

    graph_pair_tokens_by_output = {
        "T": ["T"],
        "I": ["L"],
        "N": ["N"],
        "S": ["S"],
        "O": ["O"],
    }

    col_dist = defaultdict(list)
    for feat_name in vertex_dataframe.columns.values:
        if feat_name == "Contour":
            continue
        if "Graph" not in feat_name:
            for output_label in output_cell_types.keys():
                col_dist[output_label].append(feat_name)
        else:
            try:
                graph_pair = feat_name.split("_")[1]
            except Exception:
                continue
            for output_label, graph_tokens in graph_pair_tokens_by_output.items():
                if any(token in graph_pair for token in graph_tokens):
                    col_dist[output_label].append(feat_name)

    output_folder = output_path / sample_name
    ensure_dir(output_folder)

    for output_label, type_ids in output_cell_types.items():
        vertex_csvfile = output_folder / f"{sample_name}_Feats_{output_label}.csv"
        save_index = vertex_dataframe["CellType"].isin(type_ids).values
        selected_columns = col_dist.get(
            output_label,
            [c for c in ["CellType", "Centroid"] if c in vertex_dataframe.columns],
        )
        vertex_dataframe.iloc[save_index].to_csv(
            vertex_csvfile, index=False, columns=selected_columns
        )

    edge_csvfile = output_folder / f"{sample_name}_Edges.csv"
    edge_info.to_csv(edge_csvfile, index=False)


# =============================================================================
# 7. Final neutrophil-centered feature integration
# =============================================================================

def clean_graph_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    sample_name: str,
    df_label: str,
) -> pd.DataFrame:
    """
    Convert graph columns to numeric and impute missing values.

    Imputation rules:
        - minEdgeLength / meanEdgeLength: NA -> 100
        - Nsubgraph: NA -> 1
        - Degrees: NA -> 0
    """
    if df.empty:
        return df

    for col in columns:
        if col not in df.columns:
            print(f"Warning: column {col} not found in {df_label} for sample {sample_name}")
            continue
        df[col] = df[col].astype(str).str.replace("'", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if "minEdgeLength" in col or "meanEdgeLength" in col:
            df[col] = df[col].fillna(100)
        elif "Nsubgraph" in col:
            df[col] = df[col].fillna(1)
        elif "Degrees" in col:
            df[col] = df[col].fillna(0)
    return df


def safe_numeric_mean(df: pd.DataFrame, col: str, sample_name: str, df_label: str) -> float:
    """Return the numeric mean of a column with robust conversion."""
    if df.empty or col not in df.columns:
        print(f"Warning: cannot summarize missing column {col} in {df_label} for sample {sample_name}")
        return np.nan
    s = df[col]
    if s.dtype == object:
        s = s.astype(str).str.replace("'", "", regex=False)
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean())


def summarize_one_sample(sample_folder: str, f3_dir: str | Path) -> Dict[str, float | str]:
    """
    Summarize one sample into neutrophil-centered spatial interaction features.

    Final output variables are restricted to:
        Neu-tumor_<metric>
        Neu-neu_<metric>
        Neu-immune_<metric>
        Neu-stromal_<metric>
    """
    f3_dir = Path(f3_dir)
    sample_path = f3_dir / sample_folder
    feats_n_path = sample_path / f"{sample_folder}_Feats_N.csv"
    df_n = read_csv_or_empty(feats_n_path)

    required_cols = []
    for _, _, graph_prefix in FINAL_INTERACTION_SPECS:
        for metric in GRAPH_METRICS:
            required_cols.append(f"{graph_prefix}_{metric}")

    df_n = clean_graph_columns(df_n, required_cols, sample_folder, "Feats_N")

    result = {"Sample": sample_folder}
    for output_group, _, graph_prefix in FINAL_INTERACTION_SPECS:
        for metric in GRAPH_METRICS:
            output_col = f"{output_group}_{metric}"
            graph_col = f"{graph_prefix}_{metric}"
            result[output_col] = safe_numeric_mean(df_n, graph_col, sample_folder, "Feats_N")
    return result


def run_final_summary(f3_dir: str | Path, final_output_csv: str | Path) -> pd.DataFrame:
    """Generate the final neutrophil-centered spatial feature matrix."""
    f3_dir = Path(f3_dir)
    final_output_csv = Path(final_output_csv)

    print("=" * 90)
    print("Start final neutrophil-centered feature summary")
    print(f"F3 folder : {f3_dir}")
    print(f"Output CSV: {final_output_csv}")
    print("=" * 90)

    sample_folders = [p.name for p in f3_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(sample_folders) == 0:
        raise RuntimeError(f"No sample folders found in: {f3_dir}")

    features = []
    for sample_folder in sorted(sample_folders):
        print(f"Summarizing sample: {sample_folder}")
        features.append(summarize_one_sample(sample_folder, f3_dir))

    features_df = pd.DataFrame(features)
    ensure_dir(final_output_csv.parent)
    features_df.to_csv(final_output_csv, index=False)

    print("=" * 90)
    print(f"Final summary finished: {final_output_csv}")
    print(f"Total samples: {len(features_df)}")
    print("=" * 90)
    return features_df


# =============================================================================
# 8. One-sample F1-F2-F3 workflow
# =============================================================================

def sample_f2_ready(sample_name: str, f2_dir: str | Path) -> bool:
    """Check whether the merged JSON exists for one sample."""
    return (Path(f2_dir) / f"{sample_name}.json").exists()


def sample_f3_ready(sample_name: str, f3_dir: str | Path) -> bool:
    """Check whether F3 feature files exist for one sample."""
    sample_f3_dir = Path(f3_dir) / sample_name
    required_files = [
        sample_f3_dir / f"{sample_name}_Feats_T.csv",
        sample_f3_dir / f"{sample_name}_Feats_I.csv",
        sample_f3_dir / f"{sample_name}_Feats_N.csv",
        sample_f3_dir / f"{sample_name}_Feats_S.csv",
        sample_f3_dir / f"{sample_name}_Edges.csv",
    ]
    return all(p.exists() for p in required_files)


def run_one_sample_f1_f2_f3(wsi_path: str | Path, config: argparse.Namespace) -> Dict[str, str]:
    """Run F1-F2-F3 for one WSI sample."""
    wsi_path = Path(wsi_path)
    sample_name = get_sample_name_from_wsi(wsi_path)

    f1_classic_dir = Path(config.base_dir) / "F1_classic"
    f1_neu_dir = Path(config.base_dir) / "F1_neutrophil"
    f1_classic_json_dir = f1_classic_dir / "json"
    f1_neu_json_dir = f1_neu_dir / "json"
    f2_dir = Path(config.base_dir) / "F2_merged_json"
    f3_dir = Path(config.base_dir) / "F3_graph_features"
    single_wsi_tmp_dir = Path(config.base_dir) / "_single_wsi_input"

    print("=" * 90)
    print(f"Start sample pipeline: {sample_name}")
    print(f"WSI: {wsi_path}")
    print("=" * 90)

    if sample_f3_ready(sample_name, f3_dir):
        print(f"Skip {sample_name}: F3 outputs already exist.")
        return {"Sample": sample_name, "Status": "Skipped_F3_exists", "Message": "F3 outputs already exist."}

    single_input_dir = prepare_single_wsi_input(wsi_path, sample_name, single_wsi_tmp_dir)
    classic_json_path = f1_classic_json_dir / f"{sample_name}.json"
    neu_json_path = f1_neu_json_dir / f"{sample_name}.json"

    p_classic = Process(
        target=run_hovernet_wsi,
        kwargs={
            "input_dir": single_input_dir,
            "output_dir": f1_classic_dir,
            "model_path": config.classic_model_path,
            "nr_types": config.classic_nr_types,
            "hover_dir": config.hover_dir,
            "openslide_bin": config.openslide_bin,
            "type_info_path": config.type_info_path,
            "gpu": config.gpu,
            "cache_path": Path(config.base_dir) / "_cache_classic",
            "nr_inference_workers": config.nr_inference_workers,
            "nr_post_proc_workers": config.nr_post_proc_workers,
            "batch_size": config.batch_size,
            "proc_mag": config.proc_mag,
            "ambiguous_size": config.ambiguous_size,
            "chunk_shape": config.chunk_shape,
            "tile_shape": config.tile_shape,
            "save_thumb": config.save_thumb,
            "save_mask": config.save_mask,
        },
    )

    p_neu = Process(
        target=run_hovernet_wsi,
        kwargs={
            "input_dir": single_input_dir,
            "output_dir": f1_neu_dir,
            "model_path": config.neutrophil_model_path,
            "nr_types": config.neutrophil_nr_types,
            "hover_dir": config.hover_dir,
            "openslide_bin": config.openslide_bin,
            "type_info_path": config.type_info_path,
            "gpu": config.gpu,
            "cache_path": Path(config.base_dir) / "_cache_neutrophil",
            "nr_inference_workers": config.nr_inference_workers,
            "nr_post_proc_workers": config.nr_post_proc_workers,
            "batch_size": config.batch_size,
            "proc_mag": config.proc_mag,
            "ambiguous_size": config.ambiguous_size,
            "chunk_shape": config.chunk_shape,
            "tile_shape": config.tile_shape,
            "save_thumb": config.save_thumb,
            "save_mask": config.save_mask,
        },
    )

    print(f"Running classic and neutrophil-focused models in parallel: {sample_name}")
    p_classic.start()
    p_neu.start()
    p_classic.join()
    p_neu.join()

    print(f"Classic model exit code     : {p_classic.exitcode}")
    print(f"Neutrophil model exit code  : {p_neu.exitcode}")

    if not classic_json_path.exists():
        return {"Sample": sample_name, "Status": "Failed_F1_classic_json_missing", "Message": f"Classic JSON missing: {classic_json_path}"}
    if not neu_json_path.exists():
        return {"Sample": sample_name, "Status": "Failed_F1_neutrophil_json_missing", "Message": f"Neutrophil JSON missing: {neu_json_path}"}

    try:
        if single_input_dir.exists():
            shutil.rmtree(single_input_dir)
            print(f"Deleted temporary single-WSI input folder: {single_input_dir}")
    except Exception as e:
        print(f"Warning: failed to delete temporary folder {single_input_dir}: {e}")

    try:
        output_json_path = f2_dir / f"{sample_name}.json"
        n_cells = merge_one_json(classic_json_path, neu_json_path, output_json_path)
        print(f"F2 finished for {sample_name}: merged cells = {n_cells}")
    except Exception as e:
        return {"Sample": sample_name, "Status": "Failed_F2_merge", "Message": str(e)}

    if not sample_f2_ready(sample_name, f2_dir):
        return {"Sample": sample_name, "Status": "Failed_F2_json_missing", "Message": "F2 merged JSON was not generated."}

    try:
        run_one_graph_feature_extraction(
            json_path=f2_dir / f"{sample_name}.json",
            wsi_path=wsi_path,
            output_path=f3_dir,
            xml_path=None,
            distance_threshold=config.distance_threshold,
            graph_level=config.graph_level,
            k_neighbors=config.k_neighbors,
        )
    except Exception as e:
        return {"Sample": sample_name, "Status": "Failed_F3_graph", "Message": str(e)}

    if not sample_f3_ready(sample_name, f3_dir):
        return {"Sample": sample_name, "Status": "Failed_F3_incomplete", "Message": "F3 feature files were not completely generated."}

    print("=" * 90)
    print(f"Finished sample pipeline: {sample_name}")
    print("=" * 90)
    return {"Sample": sample_name, "Status": "Success", "Message": "F1-F3 completed."}


# =============================================================================
# 9. Command-line interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="De-identified HE WSI pipeline for neutrophil-centered spatial feature extraction."
    )
    parser.add_argument("--base-dir", required=True, help="Root analysis directory. Outputs will be created here.")
    parser.add_argument("--input-wsi-dir", default=None, help="Input WSI directory. Default: <base-dir>/svs")
    parser.add_argument("--hover-dir", default="Hover", help="Local HoVer-Net repository folder.")
    parser.add_argument("--openslide-bin", default=None, help="Optional OpenSlide binary folder, mainly required on Windows.")
    parser.add_argument("--classic-model-path", default="Hover/hovernet_fast_pannuke_type_tf2pytorch.tar")
    parser.add_argument("--neutrophil-model-path", default="Hover/hovernet_fast_monusac_type_tf2pytorch.tar")
    parser.add_argument("--type-info-path", default="Hover/type_info.json")
    parser.add_argument("--classic-nr-types", type=int, default=6)
    parser.add_argument("--neutrophil-nr-types", type=int, default=5)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nr-inference-workers", type=int, default=8)
    parser.add_argument("--nr-post-proc-workers", type=int, default=0)
    parser.add_argument("--proc-mag", type=int, default=40)
    parser.add_argument("--ambiguous-size", type=int, default=128)
    parser.add_argument("--chunk-shape", type=int, default=4096)
    parser.add_argument("--tile-shape", type=int, default=2048)
    parser.add_argument("--save-thumb", action="store_true", default=True)
    parser.add_argument("--save-mask", action="store_true", default=True)
    parser.add_argument("--distance-threshold", type=int, default=100)
    parser.add_argument("--graph-level", type=int, default=0)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--summary-only", action="store_true", help="Skip F1-F3 and only summarize existing F3 features.")
    return parser.parse_args()


def main() -> None:
    """Run the full pipeline or the summary-only workflow."""
    config = parse_args()
    base_dir = Path(config.base_dir)
    input_wsi_dir = Path(config.input_wsi_dir) if config.input_wsi_dir else base_dir / "svs"

    f1_classic_dir = base_dir / "F1_classic"
    f1_neu_dir = base_dir / "F1_neutrophil"
    f2_dir = base_dir / "F2_merged_json"
    f3_dir = base_dir / "F3_graph_features"
    single_wsi_tmp_dir = base_dir / "_single_wsi_input"
    final_output_csv = base_dir / "HE_neutrophil_centered_features.csv"
    status_csv = base_dir / "HE_pipeline_sample_status.csv"

    output_dirs = [f1_classic_dir, f1_neu_dir, f2_dir, f3_dir, single_wsi_tmp_dir]

    prepare_environment(
        base_dir=base_dir,
        input_wsi_dir=input_wsi_dir,
        hover_dir=config.hover_dir,
        openslide_bin=config.openslide_bin,
        output_dirs=output_dirs,
    )

    if config.summary_only:
        run_final_summary(f3_dir=f3_dir, final_output_csv=final_output_csv)
        return

    print("=" * 90)
    print("HE neutrophil-centered WSI pipeline started")
    print(f"Base directory      : {base_dir}")
    print(f"Input WSI directory : {input_wsi_dir}")
    print(f"F1 classic output   : {f1_classic_dir}")
    print(f"F1 neutrophil output: {f1_neu_dir}")
    print(f"F2 merged JSON      : {f2_dir}")
    print(f"F3 graph features   : {f3_dir}")
    print(f"Final CSV           : {final_output_csv}")
    print(f"Status CSV          : {status_csv}")
    print("=" * 90)

    wsi_files = get_all_wsi_files(input_wsi_dir)
    if len(wsi_files) == 0:
        raise RuntimeError(f"No WSI files found in: {input_wsi_dir}")

    sample_status = []
    for idx, wsi_path in enumerate(wsi_files, start=1):
        sample_name = get_sample_name_from_wsi(wsi_path)
        print("=" * 90)
        print(f"Running sample {idx}/{len(wsi_files)}: {sample_name}")
        print("=" * 90)
        status = run_one_sample_f1_f2_f3(wsi_path, config)
        sample_status.append(status)
        pd.DataFrame(sample_status).to_csv(status_csv, index=False)
        print(f"Status : {status.get('Status')}")
        print(f"Message: {status.get('Message')}")

    run_final_summary(f3_dir=f3_dir, final_output_csv=final_output_csv)
    print("=" * 90)
    print("All available samples finished.")
    print(f"Final output: {final_output_csv}")
    print(f"Status table: {status_csv}")
    print("=" * 90)


if __name__ == "__main__":
    freeze_support()
    main()
