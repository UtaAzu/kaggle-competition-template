"""
eda.py

Paste this file into a Kaggle notebook cell or run directly as a CLI script.
It auto-detects Kaggle vs Local/Codespaces. Returns input_dir, is_kaggle.

Usage examples (Kaggle on Notebook):
!python /kaggle/working/eda.py --input-dir /kaggle/input/<dataset-slug> --reduced

Usage in Notebook (import and call):
from eda import run_eda
res = run_eda(input_dir, mask_dir, output_dir='/kaggle/working/eda', reduced=True, sample=100, show_plots=True)

"""

import os
import sys
import json
import time
import argparse
import math
import random
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, entropy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------- Utilities ----------

def detect_environment(explicit_input_dir=None):
    """Auto-detect Kaggle vs Local/Codespaces. Returns input_dir, is_kaggle."""
    if explicit_input_dir:
        return Path(explicit_input_dir), os.path.exists(explicit_input_dir)

    # Kaggle detection: environment var
    if os.getenv('KAGGLE_KERNEL_RUN_TYPE') is not None:
        base = Path('/kaggle/input')
        # heuristically select the first dataset directory under /kaggle/input
        candidates = [d for d in base.iterdir() if d.is_dir()]
        if len(candidates) > 0:
            return candidates[0], True
    # fallback local/codespaces
    return Path('./'), False


def make_output_dir(exp_id='eda', output_dir=None):
    if output_dir:
        out = Path(output_dir)
    else:
        ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        out = Path(f"experiments/{exp_id}-{ts}-artifacts")
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_images(input_dir):
    ext = ('*.png', '*.jpg', '*.jpeg', '*.tif')
    files = []
    for e in ext:
        files += list(input_dir.glob(f"**/{e}"))
    return sorted(set(files))


def list_masks(input_dir):
    # masks: .npy or .png etc.
    ext = ('*.npy','*.png','*.jpg')
    files = []
    for e in ext:
        files += list(input_dir.glob(f"**/{e}"))
    return sorted(set(files))


def image_stats(path):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    lum = (0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2])
    return {
        'path': str(path),
        'width': int(w), 'height': int(h), 'area': int(w*h),
        'aspect': float(w/h),
        'mean_r': float(arr[...,0].mean()), 'mean_g': float(arr[...,1].mean()), 'mean_b': float(arr[...,2].mean()),
        'std_r': float(arr[...,0].std()), 'std_g': float(arr[...,1].std()), 'std_b': float(arr[...,2].std()),
        'brightness': float(lum.mean()), 'contrast': float(lum.std()),
        'format': path.suffix.lower()
    }


def mask_stats(mask_path, image_area=None):
    if mask_path.suffix.lower() == '.npy':
        mask = np.load(mask_path)
    else:
        mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        return { 'mask_path': str(mask_path), 'mask_area': 0, 'mask_ratio': 0.0 }
    if mask.ndim == 3:
        mask = mask.max(axis=0)
    mask_bin = (mask > 0).astype(np.uint8)
    area = int(mask_bin.sum())
    if area == 0:
        return { 'mask_path': str(mask_path), 'mask_area': 0, 'mask_ratio': 0.0 }
    h, w = mask.shape[:2]
    area_total = image_area or (w*h)
    x,y,wb,hb = cv2.boundingRect(mask_bin.astype(np.uint8))
    bbox_area = int(wb*hb)
    ncomp = int(np.max(cv2.connectedComponents(mask_bin)[1]))
    return {
        'mask_path': str(mask_path), 'mask_area': area, 'mask_ratio': float(area/area_total),
        'bbox_area': bbox_area, 'bbox_ratio': float(bbox_area/area_total), 'n_components': ncomp
    }


def compute_bg_features(img_path, mask_path=None):
    """Compute background (non-mask) features for image-level analysis."""
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    area_total = h * w
    if mask_path and Path(mask_path).exists():
        if Path(mask_path).suffix == '.npy':
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(str(mask_path), 0)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
    if mask is None:
        mask = np.zeros((h, w), dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask.max(axis=0)
    mask_bin = (mask > 0).astype(np.uint8)
    bg_mask = (1 - mask_bin).astype(np.uint8)
    bg_area = int(bg_mask.sum())
    bg_ratio = float(bg_area / area_total) if area_total > 0 else 0.0
    # border coverage: fraction of background pixels on image borders
    edges = np.zeros_like(bg_mask)
    edges[0, :] = 1; edges[-1, :] = 1; edges[:, 0] = 1; edges[:, -1] = 1
    border_bg = float((bg_mask & edges).sum() / area_total) if area_total > 0 else 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    bg_lap_var = float(lap[bg_mask == 1].var()) if bg_area > 0 else 0.0
    # color means on background
    bg_pixels = img[bg_mask == 1]
    if len(bg_pixels) > 0:
        mean_r = float(bg_pixels[:, 0].mean() / 255.0)
        mean_g = float(bg_pixels[:, 1].mean() / 255.0)
        mean_b = float(bg_pixels[:, 2].mean() / 255.0)
    else:
        mean_r = mean_g = mean_b = 0.0
    # entropy of background color distribution (grayscale histogram)
    if bg_area > 0:
        bg_hist, _ = np.histogram(gray[bg_mask == 1], bins=64, range=(0, 255), density=True)
        bg_hist = bg_hist + 1e-9
        bg_entropy = float(entropy(bg_hist))
    else:
        bg_entropy = 0.0
    return {
        'image_path': str(img_path), 'bg_area': bg_area, 'bg_ratio': bg_ratio,
        'border_bg': border_bg, 'bg_lap_var': bg_lap_var,
        'bg_mean_r': mean_r, 'bg_mean_g': mean_g, 'bg_mean_b': mean_b,
        'bg_entropy': bg_entropy, 'width': w, 'height': h
    }


def save_json(obj, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------- Step 1: Overview ----------

def step_overview(train_df=None, test_df=None, images=None, masks=None, outdir=Path('.')):
    rows = {}
    rows['n_train_rows'] = int(len(train_df)) if train_df is not None else None
    rows['n_test_rows'] = int(len(test_df)) if test_df is not None else None
    rows['n_images'] = len(images) if images is not None else None
    rows['n_masks'] = len(masks) if masks is not None else None
    # additional stats
    if train_df is not None:
        rows['train_columns'] = train_df.shape[1]
        rows['train_memory_mb'] = int(train_df.memory_usage(deep=True).sum() / (1024*1024))
    p = outdir / 'overview.json'
    save_json(rows, p)
    print('Saved overview:', p)
    return rows

# ---------- Step 2: Target Analysis ----------

def step_target(train_df, outdir=Path('.')):
    if train_df is None or 'annotation' not in train_df.columns:
        print('No target column found; skip target analysis')
        return {}
    ann = train_df['annotation']
    cnt = ann.value_counts(dropna=False).to_dict()
    # If segmentation, compute authentic vs forged
    is_authentic = ann == 'authentic'
    prop_auth = float(is_authentic.mean())
    res = {'value_counts': cnt, 'prop_authentic': prop_auth}
    save_json(res, outdir/'target_summary.json')
    return res

# ---------- Step 3: Data Quality ----------

def step_quality(train_df, outdir=Path('.')):
    res = {}
    if train_df is not None:
        missing = train_df.isnull().mean().sort_values(ascending=False).to_dict()
        # df.nunique can raise TypeError when some cells are unhashable
        # (lists/dicts). Provide a robust fallback converting to hashable types.
        def safe_nunique(df):
            try:
                return df.nunique(dropna=False).sort_values(ascending=False).to_dict()
            except TypeError:
                counts = {}
                for c in df.columns:
                    try:
                        counts[c] = int(df[c].nunique(dropna=False))
                    except TypeError:
                        def make_hashable(x):
                            if isinstance(x, list):
                                return tuple(x)
                            if isinstance(x, dict):
                                return tuple(sorted(x.items()))
                            return x
                        uniq = {make_hashable(x) for x in df[c].values}
                        counts[c] = int(len(uniq))
                return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

        nunique = safe_nunique(train_df)
        res['missing'] = missing
        res['nunique'] = nunique
        save_json(res, outdir/'data_quality.json')
    return res

# ---------- Step 4: Univariate ----------

def step_univariate(train_df, target_col=None, outdir=Path('.')):
    if train_df is None:
        return {}
    numeric = train_df.select_dtypes(include=[np.number])
    top_feats = []
    corr = None
    if target_col and target_col in train_df.columns:
        try:
            corr = numeric.corrwith(train_df[target_col].astype(float), method='pearson').abs().sort_values(ascending=False)
            top_feats = corr.head(5).index.tolist()
        except Exception:
            top_feats = []
    # Simple plots
    for c in numeric.columns:
        fig, ax = plt.subplots(figsize=(6,3)); sns.histplot(numeric[c].dropna(), ax=ax, kde=False); ax.set_title(c)
        p = outdir/f'feature_hist_{c}.png'; fig.savefig(p); plt.close(fig)
    save_json({'top_feats': top_feats}, outdir/'univariate.json')
    return {'top_feats': top_feats, 'corr_target': corr.to_dict() if corr is not None else {}}

# ---------- Step 5: Bivariate ----------

def step_bivariate(train_df, outdir=Path('.')):
    if train_df is None:
        return {}
    numeric = train_df.select_dtypes(include=[np.number])
    corrmat = numeric.corr()
    fig, ax = plt.subplots(figsize=(8,6)); sns.heatmap(corrmat, vmin=-1, vmax=1, cmap='RdBu_r'); fig.savefig(outdir/'correlation_heatmap.png'); plt.close(fig)
    # high-corr pairs
    pairs = []
    abs_corr = corrmat.abs()
    for i in range(len(abs_corr.columns)):
        for j in range(i+1, len(abs_corr.columns)):
            v = abs_corr.iloc[i, j]
            if v > 0.85:
                pairs.append((abs_corr.columns[i], abs_corr.columns[j], float(v)))
    save_json({'high_corr_pairs': pairs}, outdir/'bivariate.json')
    return {'high_corr_pairs': pairs}

# ---------- Step 6: Time Series ----------

def step_time_series(train_df, outdir=Path('.')):
    if train_df is None:
        return {}
    time_cols = [c for c in train_df.columns if 'time' in c.lower() or 'date' in c.lower()]
    res = {}
    for c in time_cols:
        try:
            s = pd.to_datetime(train_df[c], errors='coerce')
            if s.notnull().any():
                fig, ax = plt.subplots(figsize=(8,3)); s.dropna().dt.date.value_counts().sort_index().plot(ax=ax); fig.savefig(outdir/f'time_series_{c}.png'); plt.close(fig)
                res[c] = 'ok'
        except Exception:
            res[c] = 'failed'
    save_json(res, outdir/'time_series.json')
    return res

# ---------- Step 7: Distribution Shift ----------

def step_distribution_shift(train_df, test_df, outdir=Path('.')):
    if train_df is None or test_df is None:
        print('Train or test df missing; skip distribution shift')
        return {}
    numeric = set(train_df.select_dtypes(include=[np.number]).columns) & set(test_df.select_dtypes(include=[np.number]).columns)
    res = {}
    for f in numeric:
        a = train_df[f].dropna().values
        b = test_df[f].dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        ks = float(ks_2samp(a, b).statistic)
        # JS divergence
        pa, _ = np.histogram(a, bins=50, density=True); pb, _ = np.histogram(b, bins=50, density=True)
        pa = pa + 1e-9; pb = pb + 1e-9; pa/=pa.sum(); pb/=pb.sum()
        m = 0.5*(pa+pb)
        js = float(0.5*(entropy(pa, m) + entropy(pb, m)))
        res[f] = {'ks': ks, 'js': js}
    save_json(res, outdir/'distribution_shift.json')
    return res

# ---------- Image/Mask EDA ----------

def step_image_mask(images, masks, outdir=Path('.'), sample=50):
    # images, masks are arrays of Path
    out = {}
    if images:
        sample_images = random.sample(images, min(len(images), sample))
        img_rows = [image_stats(p) for p in sample_images]
        df = pd.DataFrame(img_rows)
        df.to_csv(outdir/'images_sample_summary.csv', index=False)
        out['images_sample_summary'] = str(outdir/'images_sample_summary.csv')
    if masks:
        sample_masks = random.sample(masks, min(len(masks), sample))
        mask_rows = [mask_stats(p) for p in sample_masks]
        dfm = pd.DataFrame(mask_rows)
        dfm.to_csv(outdir/'masks_sample_summary.csv', index=False)
        out['masks_sample_summary'] = str(outdir/'masks_sample_summary.csv')
    # overlay some samples
    try:
        for i, m in enumerate(sample_masks[:6]):
            mask = np.load(m) if m.suffix == '.npy' else cv2.imread(str(m),0)
            base = Path(m).stem
            img_path = next((x for x in images if Path(x).stem==base), None)
            if img_path:
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                if mask.ndim==3: mask = mask.max(axis=0)
                fig, ax = plt.subplots(figsize=(6,4)); ax.imshow(img); ax.imshow(mask, alpha=0.4, cmap='jet'); ax.axis('off')
                fig.savefig(outdir/f'overlay_{base}.png'); plt.close(fig)
    except Exception:
        pass
    return out


def step_background_features(images, masks, outdir=Path('.')):
    """Compute background features for all images and save to CSV."""
    if not images:
        return {}
    print("Computing background features for images... (this may take a while)")
    rows = []
    mask_map = {Path(m).stem: m for m in masks}
    for p in tqdm(sorted(images), desc="bg features"):
        stem = Path(p).stem
        mpath = mask_map.get(stem, None)
        rows.append(compute_bg_features(p, mpath))
    df = pd.DataFrame(rows)
    p = outdir / 'images_bg_features.csv'
    df.to_csv(p, index=False)
    return {'images_bg_features': str(p)}


def step_bg_clustering(outdir=Path('.'), bg_features_csv=None, n_clusters=4):
    """Cluster background features and save the mapping and representative images."""
    if not bg_features_csv or not Path(bg_features_csv).exists():
        return {}
    df = pd.read_csv(bg_features_csv)
    feat_cols = ['bg_ratio', 'border_bg', 'bg_lap_var', 'bg_entropy', 'bg_mean_r', 'bg_mean_g', 'bg_mean_b']
    # Clean NaNs
    df[feat_cols] = df[feat_cols].fillna(0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feat_cols].values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['bg_cluster'] = kmeans.fit_predict(X)
    p = outdir / 'bg_cluster_map.csv'
    df.to_csv(p, index=False)
    # Save cluster distributions and representative images
    stats = df.groupby('bg_cluster')['bg_ratio'].describe().to_dict()
    save_json({'cluster_stats': stats}, outdir / 'bg_clusters_summary.json')
    # save representative images: choose top few by closeness to cluster center
    centers = kmeans.cluster_centers_
    df['dist_to_center'] = np.linalg.norm(X - centers[df['bg_cluster']], axis=1)
    rep_dir = outdir / 'bg_cluster_examples'
    rep_dir.mkdir(parents=True, exist_ok=True)
    for c in range(n_clusters):
        sub = df[df['bg_cluster'] == c].sort_values('dist_to_center')
        for i, row in sub.head(6).iterrows():
            src = Path(row['image_path'])
            if src.exists():
                dst = rep_dir / f'cluster{c}_{Path(src).stem}.png'
                img = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # persist scaler params and centers for mapping test images
    np.save(outdir / 'bg_cluster_centers.npy', centers)
    np.save(outdir / 'bg_scaler_mean.npy', scaler.mean_)
    np.save(outdir / 'bg_scaler_scale.npy', scaler.scale_)
    return {'bg_cluster_map': str(p), 'bg_cluster_examples': str(rep_dir)}


def build_target_df_from_masks(images, masks):
    """Create a small DataFrame with 'case_id' and 'annotation' using mask presence.

    This helps image-only competitions where no train.csv is provided.
    """
    mask_map = {}
    for m in masks:
        key = Path(m).stem
        mask_map.setdefault(key, []).append(str(m))
    rows = []
    for img in images:
        base = Path(img).stem
        has_mask = base in mask_map
        rows.append({
            'case_id': base,
            'image_path': str(img),
            'annotation': 'authentic' if not has_mask else 'forged',
            'n_masks': len(mask_map.get(base, [])),
            'mask_paths': mask_map.get(base, [])
        })
    return pd.DataFrame(rows)


def step_image_univariate(images, outdir=Path('.')):
    """Compute image-level numeric feature distributions when no CSV label file exists.

    This serves as a fallback for the univariate step in purely image competitions.
    """
    if not images:
        return {}
    rows = [image_stats(p) for p in images]
    df = pd.DataFrame(rows)
    df.to_csv(outdir/'images_full_summary.csv', index=False)
    stats = df.describe().to_dict()
    numeric = df.select_dtypes(include=[np.number])
    # Correlation with brightness
    corr = numeric.corr()['brightness'].abs().sort_values(ascending=False).to_dict()
    # save histograms for top numeric columns only (avoid too many files)
    for c in numeric.columns[:12]:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.histplot(numeric[c].dropna(), ax=ax, kde=False, bins=40)
        ax.set_title(f'image_{c}')
        fig.savefig(outdir/f'image_feature_hist_{c}.png')
        plt.close(fig)
    res = {'image_stats': stats, 'image_corr_with_brightness': corr}
    save_json(res, outdir/'univariate_image.json')
    return res

# ---------- Main run_eda ----------

def run_eda(input_dir=None, mask_dir=None, sample=None, reduced=True, output_dir=None, show_plots=False):
    input_dir, is_kaggle = detect_environment(input_dir)
    input_dir = Path(input_dir)
    outdir = make_output_dir('eda', output_dir)
    print('Input dir:', input_dir, 'is_kaggle:', is_kaggle)

    # try to load train/test csvs
    train_df = None; test_df = None
    for f in ('train.csv','train.csv.gz','train.zip','sample_train.csv'):
        p = input_dir/f
        if p.exists():
            try:
                train_df = pd.read_csv(p)
                print('Loaded train', p)
                break
            except Exception:
                continue
    for f in ('test.csv','sample_test.csv'):
        p = input_dir/f
        if p.exists():
            try:
                test_df = pd.read_csv(p)
                print('Loaded test', p)
                break
            except Exception:
                continue

    # image and mask directories
    # common layout in competition
    img_dir = input_dir/'train_images'
    test_img_dir = input_dir/'test_images'
    mask_dir = Path(mask_dir) if mask_dir else input_dir/'train_masks'

    images = list_images(img_dir) if img_dir.exists() else []
    test_images = list_images(test_img_dir) if test_img_dir.exists() else []
    masks = list_masks(mask_dir) if mask_dir.exists() else []

    # sample reduction
    if reduced and sample:
        if len(images)>sample:
            images = random.sample(images, sample)
        if len(masks)>sample:
            masks = random.sample(masks, sample)

    # If no train.csv but masks/images exist, infer a pseudo train_df
    if train_df is None and images:
        print("No train.csv found — infer annotation from mask presence")
        train_df = build_target_df_from_masks(images, masks)

    # Step 1
    overview = step_overview(train_df, test_df, images, masks, outdir)

    # Step 2
    target_info = step_target(train_df, outdir)

    # Step 3
    dq = step_quality(train_df, outdir)

    # Step 4
    # If there are no numeric columns in train_df (common in image-only dataset),
    # fall back to image-level univariate features
    if train_df is None or not any(np.issubdtype(dt, np.number) for dt in train_df.dtypes):
        uni = step_image_univariate(images, outdir=outdir)
    else:
        uni = step_univariate(train_df, target_col='annotation' if train_df is not None and 'annotation' in train_df.columns else None, outdir=outdir)

    # Step 5
    bi = step_bivariate(train_df, outdir)

    # Step 6
    ts = step_time_series(train_df, outdir)

    # Step 7
    shift = step_distribution_shift(train_df, test_df, outdir)

    # Image/Mask
    im = step_image_mask(images, masks, outdir, sample=50)
    # Background features and clustering
    bg = step_background_features(images, masks, outdir)
    bg_csv = Path(bg.get('images_bg_features')) if isinstance(bg, dict) and bg.get('images_bg_features') else None
    if bg_csv and bg_csv.exists():
        cl = step_bg_clustering(outdir, str(bg_csv), n_clusters=4)
    else:
        cl = {}
    # compute cluster distribution for train (if available)
    try:
        if cl and 'bg_cluster_map' in cl and Path(cl['bg_cluster_map']).exists():
            dfc = pd.read_csv(cl['bg_cluster_map'])
            train_cluster_dist = dfc['bg_cluster'].value_counts(normalize=True).to_dict()
        else:
            train_cluster_dist = {}
        # map test images to clusters if scaler/centers exist
        centers_path = outdir / 'bg_cluster_centers.npy'
        if centers_path.exists() and len(test_images)>0:
            mean = np.load(outdir / 'bg_scaler_mean.npy')
            scale = np.load(outdir / 'bg_scaler_scale.npy')
            centers = np.load(centers_path)
            feat_cols = ['bg_ratio', 'border_bg', 'bg_lap_var', 'bg_entropy', 'bg_mean_r', 'bg_mean_g', 'bg_mean_b']
            test_bg_rows = [compute_bg_features(p, None) for p in test_images]
            dft = pd.DataFrame(test_bg_rows)
            dft[feat_cols] = dft[feat_cols].fillna(0.0)
            X_test = (dft[feat_cols].values - mean) / scale
            dists = np.linalg.norm(X_test[:, None, :] - centers[None, :, :], axis=2)
            cluster_assign = dists.argmin(axis=1)
            test_cluster_dist = pd.Series(cluster_assign).value_counts(normalize=True).to_dict()
        else:
            test_cluster_dist = {}
    except Exception as e:
        print('Error computing cluster distributions:', e)
        train_cluster_dist, test_cluster_dist = {}, {}

    summary = {
        'overview': overview,
        'target': target_info,
        'quality': dq,
        'univariate': uni,
        'bivariate': bi,
        'time_series': ts,
        'distribution_shift': shift,
        'image_mask': im
    }
    # Add background cluster distribution info if available
    summary['bg'] = {
        'bg_features': bg,
        'bg_cluster': cl,
        'train_cluster_dist': train_cluster_dist,
        'test_cluster_dist': test_cluster_dist
    }

    save_json(summary, outdir/'eda_summary.json')
    print('EDA complete. Artifacts saved to:', outdir)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run EDA for dataset. Auto-detect Kaggle vs local.')
    parser.add_argument('--input-dir', default=None)
    parser.add_argument('--mask-dir', default=None)
    parser.add_argument('--sample', default=200, type=int)
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    # In interactive environments (Jupyter/Colab) the kernel injects its own
    # argv entries such as "-f /root/.local/.../kernel-xxx.json" which
    # cause argparse to fail. Use parse_known_args() to ignore unexpected args
    # when users paste/run this file from a notebook cell.
    args, _ = parser.parse_known_args()

    random.seed(args.seed); np.random.seed(args.seed)
    run_eda(input_dir=args.input_dir, mask_dir=args.mask_dir, sample=args.sample, reduced=args.reduced, output_dir=args.output_dir, show_plots=not args.no_plots)
