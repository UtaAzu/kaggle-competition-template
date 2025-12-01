import os
import argparse
import cv2
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Mask & Metric Extraction
# -----------------------------

def split_composite(img: np.ndarray):
    """Split composite visualization into panels.
    Expected layouts:
      5 panels: Original | GT | Pred | FP | FN
      3 panels (fallback): Original | GT | Pred
    Returns dict of panels as BGR arrays (for color channel extraction).
    """
    h, w = img.shape[:2]
    panels = {}
    # Try 5-panel first
    if w % 5 == 0:
        pw = w // 5
        names = ['original', 'gt', 'pred', 'fp', 'fn']
        for i, name in enumerate(names):
            panel = img[:, i*pw:(i+1)*pw]
            panels[name] = panel  # Keep BGR format
        panels['layout'] = 5
    elif w % 3 == 0:
        pw = w // 3
        names = ['original', 'gt', 'pred']
        for i, name in enumerate(names):
            panel = img[:, i*pw:(i+1)*pw]
            panels[name] = panel  # Keep BGR format
        panels['layout'] = 3
    else:
        # Attempt approximate equal partition (rounding) for robustness
        # This helps if thin separator lines cause slight width drift.
        candidate = None
        for n in (5, 3):
            approx_pw = w / n
            # within 2% relative tolerance
            if abs(round(approx_pw) - approx_pw) / approx_pw < 0.02:
                candidate = n
                break
        if candidate:
            n = candidate
            pw = round(w / n)
            names = ['original', 'gt', 'pred'] + (['fp', 'fn'] if n == 5 else [])
            for i, name in enumerate(names):
                xs = i*pw
                xe = (i+1)*pw if i < n-1 else w
                panel = img[:, xs:xe]
                panels[name] = panel  # Keep BGR format
            panels['layout'] = n
        else:
            raise ValueError(f"Unrecognized composite layout width={w}. Expected divisible by 5 or 3.")
    return panels


def binarize(panel: np.ndarray, thresh: int = 10):
    """Convert grayscale panel to binary mask: pixels > thresh become 1 (uint8 0/1)."""
    return (panel > thresh).astype(np.uint8)


def extract_masks_from_color_panels(panels, binarize_thresh=100):
    """Extract GT/Pred masks from colored overlay panels.
    GT typically uses green overlay (BGR channel 1).
    Pred typically uses red overlay (BGR channel 2).
    Returns: (gt_mask, pred_mask) as binary uint8 arrays.
    """
    layout = panels['layout']
    
    if layout == 5:
        # GT panel: extract green channel (index 1 in BGR)
        gt_panel = panels['gt']
        pred_panel = panels['pred']
    else:  # layout 3
        gt_panel = panels['gt']
        pred_panel = panels['pred']
    
    # Extract masks from color channels
    # GT: green channel (BGR[1])
    gt_mask = (gt_panel[:, :, 1] > binarize_thresh).astype(np.uint8)
    # Pred: red channel (BGR[2])
    pred_mask = (pred_panel[:, :, 2] > binarize_thresh).astype(np.uint8)
    
    return gt_mask, pred_mask


def compute_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray):
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())
    intersection = int((gt_mask & pred_mask).sum())
    fp_area = pred_area - intersection
    fn_area = gt_area - intersection
    precision = intersection / pred_area if pred_area > 0 else 0.0
    recall = intersection / gt_area if gt_area > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        'gt_area': gt_area,
        'pred_area': pred_area,
        'intersection': intersection,
        'fp_area': fp_area,
        'fn_area': fn_area,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def boundary_mask(mask: np.ndarray, k: int = 3):
    if mask.sum() == 0:
        return np.zeros_like(mask)
    kernel = np.ones((k, k), np.uint8)
    er = cv2.erode(mask, kernel, iterations=1)
    dil = cv2.dilate(mask, kernel, iterations=1)
    # Boundary pixels: dilated - eroded (or mask - eroded)
    bnd = ((dil - er) > 0).astype(np.uint8)
    return bnd


def classify_pattern(row, params):
    gt_area = row['gt_area']
    pred_area = row['pred_area']
    fp_area = row['fp_area']
    fn_area = row['fn_area']
    f1 = row['f1']
    # Derived ratios
    fp_ratio_pred = fp_area / pred_area if pred_area > 0 else 0.0
    fn_ratio_gt = fn_area / gt_area if gt_area > 0 else 0.0

    # Complete miss
    if gt_area > 0 and pred_area == 0:
        return 'complete_miss'

    # Small area FN
    if gt_area > 0 and (gt_area / row['total_pixels']) < params.small_gt_frac and fn_ratio_gt > params.small_area_fn_min_fn_frac:
        return 'small_area_fn'

    # FP dominant
    if pred_area > 0 and fp_ratio_pred > params.fp_dominant_ratio:
        return 'fp_dominant'

    # Partial detection (some intersection, some fn)
    if row['intersection'] > 0 and fn_area > 0 and f1 < 0.9:
        return 'partial_detection'

    # Good (high quality)
    if f1 >= 0.9:
        return 'good'

    # Fallback generic error
    return 'other_error'


def refine_boundary_classification(df, panels_dict, params):
    # Add boundary_fn classification where FN concentrated on boundaries
    boundary_flags = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc='Boundary pass'):
        image_id = r['image_id']
        entry = panels_dict.get(image_id)
        if entry is None:
            boundary_flags.append(False)
            continue
        gt_mask = entry['gt_mask']
        # Prefer logical operations (1 - pred) to avoid bitwise invert side-effects on uint8.
        fn_mask = (gt_mask & (1 - entry['pred_mask']))
        if fn_mask.sum() == 0:
            boundary_flags.append(False)
            continue
        bmask = boundary_mask(gt_mask, k=3)
        boundary_fn_ratio = (fn_mask & bmask).sum() / fn_mask.sum()
        boundary_flags.append(boundary_fn_ratio >= params.min_fn_boundary_ratio)
    df['boundary_fn_flag'] = boundary_flags
    df.loc[df['boundary_fn_flag'], 'pattern'] = 'boundary_fn'
    return df

# -----------------------------
# Parameter Dataclass Substitute
# -----------------------------
class Params:
    def __init__(self, min_fn_boundary_ratio, fp_dominant_ratio, small_gt_frac, small_area_fn_min_fn_frac):
        self.min_fn_boundary_ratio = min_fn_boundary_ratio
        self.fp_dominant_ratio = fp_dominant_ratio
        self.small_gt_frac = small_gt_frac
        self.small_area_fn_min_fn_frac = small_area_fn_min_fn_frac

# -----------------------------
# Analysis Runner
# -----------------------------

def analyze_visuals(visuals_dir: str, output_dir: str, params: Params, limit: int = None, binarize_thresh: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    records = []
    panels_cache = {}
    if not os.path.exists(visuals_dir):
        raise FileNotFoundError(f"Visuals directory does not exist: {visuals_dir}")
    image_files = [f for f in os.listdir(visuals_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    image_files.sort()
    if len(image_files) == 0:
        print(f"No image files found in {visuals_dir}. Nothing to analyze.")
        return
    if limit:
        image_files = image_files[:limit]

    skipped = 0
    for fname in tqdm(image_files, desc='Analyzing composites'):
        fpath = os.path.join(visuals_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        try:
            panels = split_composite(img)
        except Exception as e:
            print(f"[WARN] Skip {fname}: {e}")
            continue
        layout = panels['layout']
        # Extract masks using color channel extraction
        gt_mask, pred_mask = extract_masks_from_color_panels(panels, binarize_thresh)
        # Ensure masks have matching shape
        if gt_mask.shape != pred_mask.shape:
            print(f"[WARN] Skipping {fname}: GT and Pred mask shapes differ {gt_mask.shape} vs {pred_mask.shape}")
            skipped += 1
            continue
        h, w = gt_mask.shape
        total_pixels = h * w
        metrics = compute_metrics(gt_mask, pred_mask)
        row = {
            'image_id': os.path.splitext(fname)[0],
            'layout': layout,
            'total_pixels': total_pixels,
            **metrics,
        }
        panels_cache[row['image_id']] = {'gt_mask': gt_mask, 'pred_mask': pred_mask}
        records.append(row)

    if not records:
        print('No valid composite images found.')
        return

    df = pd.DataFrame(records)
    df['pattern'] = df.apply(lambda r: classify_pattern(r, params), axis=1)
    # Boundary refinement step
    df = refine_boundary_classification(df, panels_cache, params)

    # Aggregate stats
    pattern_counts = df['pattern'].value_counts().to_dict()
    avg_metrics = df[['precision', 'recall', 'f1']].mean().to_dict()

    # Save CSV
    csv_path = os.path.join(output_dir, 'analysis_results.csv')
    df.to_csv(csv_path, index=False)

    # Pattern markdowns
    for pattern, subset in df.groupby('pattern'):
        md_path = os.path.join(output_dir, f'pattern_{pattern}.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Pattern: {pattern}\n\n")
            f.write(f"Count: {len(subset)}\n\n")
            f.write("Examples (up to 20):\n")
            for img_id in subset['image_id'].head(20):
                f.write(f"- {img_id}\n")
            f.write("\nMetric Averages (subset):\n")
            f.write(subset[['precision', 'recall', 'f1']].mean().to_string())

    # Summary markdown
    summary_path = os.path.join(output_dir, 'visual_summary_report.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('# Visual Analysis Summary\n\n')
        f.write('## Global Metrics\n')
        for k, v in avg_metrics.items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write('\n## Pattern Counts\n')
        for k, v in pattern_counts.items():
            f.write(f"- {k}: {v}\n")
        f.write('\n## Worst 20 by F1\n')
        worst = df.sort_values('f1').head(20)
        for _, r in worst.iterrows():
            f.write(f"- {r['image_id']} f1={r['f1']:.4f} pattern={r['pattern']}\n")
        f.write('\n## Best 20 by F1\n')
        best = df.sort_values('f1', ascending=False).head(20)
        for _, r in best.iterrows():
            f.write(f"- {r['image_id']} f1={r['f1']:.4f} pattern={r['pattern']}\n")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    df['f1'].hist(ax=axes[0, 0], bins=30, color='steelblue')
    axes[0, 0].set_title('F1 Distribution')
    axes[0, 0].set_xlabel('F1')
    axes[0, 0].set_ylabel('Count')

    df['precision'].hist(ax=axes[0, 1], bins=30, color='darkorange')
    axes[0, 1].set_title('Precision Distribution')

    df['recall'].hist(ax=axes[1, 0], bins=30, color='seagreen')
    axes[1, 0].set_title('Recall Distribution')

    # Pattern counts bar chart
    pc_items = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in pc_items]
    counts = [x[1] for x in pc_items]
    axes[1, 1].bar(labels, counts, color='mediumpurple')
    axes[1, 1].set_title('Pattern Counts')
    axes[1, 1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots.png')
    fig.savefig(plot_path)
    plt.close(fig)

    print(f"Saved analysis artifacts to {output_dir}")
    if skipped > 0:
        print(f"Skipped {skipped} images due to malformed panels / shape mismatch.")


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze composite visualization panels.')
    parser.add_argument('--visuals-dir', type=str, default=None, help='Directory with composite images (provide path; default None)')
    parser.add_argument('--output-dir', type=str, default='/kaggle/working', help='Directory to write analysis outputs')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on number of images to analyze')
    parser.add_argument('--min-fn-boundary-ratio', type=float, default=0.6, help='Boundary FN ratio threshold')
    parser.add_argument('--fp-dominant-ratio', type=float, default=0.6, help='FP dominance ratio threshold (fp/pred)')
    parser.add_argument('--small-gt-frac', type=float, default=0.01, help='GT area fraction threshold for small_area_fn')
    parser.add_argument('--small-area-fn-min-fn-frac', type=float, default=0.5, help='Minimum fn/gt fraction for small_area_fn')
    parser.add_argument('--binarize-thresh', type=int, default=10, help='Threshold value for binarization (0-255)')
    # Use parse_known_args so this script can be run from Jupyter/Kaggle notebooks
    # which may inject kernel-specific arguments like '-f <kernel-file>'.
    # Clean out known kernel args that some notebook frontends inject (e.g. -f <file>)
    # This helps when running the script via `%run` or from within the interactive kernel.
    argv = list(sys.argv)
    cleaned = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a == '-f':
            # skip file path following -f
            skip_next = True
            continue
        if a.startswith('--ipykernel') or a.startswith('--profile'):
            continue
        cleaned.append(a)
    # parsed from cleaned argv; drop program name
    args, _unknown = parser.parse_known_args(cleaned[1:])
    return args


def load_prob_debug_summary(csv_path: str):
    """Load probability debug summary CSV in a robust way.
    Ensures the key columns exist (gt_inside_mean / gt_outside_mean) even if older debug CSVs lack them.
    Returns a pandas.DataFrame with the expected columns present.
    """
    df = pd.read_csv(csv_path)
    # Guarantee expected columns exist, fill with NaN if missing
    expected = ['case_id', 'min', 'max', 'mean', 'median', 'percentiles', 'gt_inside_mean', 'gt_outside_mean']
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    # Ensure percentiles column is parsed as string or list - leave as-is
    return df


def show_prob_debug_summary(csv_path: str):
    """Utility to display a human-friendly summary of prob-debug CSV.
    Uses load_prob_debug_summary to be resilient to missing fields.
    """
    df = load_prob_debug_summary(csv_path)
    display_df = df.head(10)
    print(display_df.to_string(index=False))
    print('\n-- Basic statistics --')
    # Only compute describe() on numeric columns we know are safe
    cols = ['min', 'max', 'mean', 'median', 'gt_inside_mean', 'gt_outside_mean']
    num_cols = [c for c in cols if c in df.columns]
    if num_cols:
        print(df[num_cols].describe().T)
    else:
        print('No numeric prob-debug columns found to describe.')
    return df


def main():
    args = parse_args()
    params = Params(
        min_fn_boundary_ratio=args.min_fn_boundary_ratio,
        fp_dominant_ratio=args.fp_dominant_ratio,
        small_gt_frac=args.small_gt_frac,
        small_area_fn_min_fn_frac=args.small_area_fn_min_fn_frac,
    )
    analyze_visuals(
        visuals_dir=args.visuals_dir,
        output_dir=args.output_dir,
        params=params,
        limit=args.limit,
        binarize_thresh=args.binarize_thresh,
    )

if __name__ == '__main__':
    main()
