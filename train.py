"""
====================================================================================================
EXP004Tv3 — Ensemble Inference: DINOv2 (Transformer) + DeepLabV3+ (CNN)
====================================================================================================

Purpose:
    - Combine DINOv2 (global semantic features) with DeepLabV3+ (dense prediction)
    - Weighted ensemble blending: W_DINO=0.6, W_DEEPLAB=0.4
    - Generate RLE-encoded submission for Kaggle competition
    - Break through the 0.330 LB score barrier with chemical reaction of two models

Architecture:
    - Model 1: DINOv2 (Vision Transformer) + Tiny Decoder (768 → 1 channel)
    - Model 2: DeepLabV3+ (ResNet34 backbone) - pre-trained from EXP003T
    - Ensemble: Weighted average of probability maps
    - TTA: 3-direction averaging (Base, H-flip, V-flip)

Output:
    - submission.csv (RLE-encoded forgery annotations)
    - val_oof.csv, val_metrics.json (validation evaluation with GT masks)
    - visuals/ (4-panel comparisons: Original, GT, DINO prob, DeepLab prob, Ensemble prob, overlays)

Strategy Quote:
    "いきなり『DINOv2 + DeepLabV3+』の共演（アンサンブル）で行くわよ！" — ポンポさん 🎬
    DINOv2単体（LB 0.318）とDeepLab単体（est. 0.29〜0.31）の化学反応で0.330の壁をぶち破る！

Date: 2025-12-01
Author: UtaAzu (Director: ポンポさん 🎬)
Version: Ensemble v3 (Stronger baseline towards 0.330+)
"""

import os
import sys
import cv2
import json
import math
import argparse
import random
import shutil
import torch
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel, BitImageProcessor

# Suppress noisy logs from TF/XLA and protobuf incompat warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
warnings.filterwarnings("ignore", category=UserWarning)

# ====================================================================================================
# Config
# ====================================================================================================
class Config:
    # Weights for Ensemble (Total = 1.0)
    # Strategy: DINOv2はLB 0.318の実績あるスター、DeepLabV3+は補完役
    W_DINO = 0.6
    W_DEEPLAB = 0.4
    
    # Paths (can be set via environment variables or CLI)
    TEST_DIR = os.environ.get('TEST_DIR', None)
    SAMPLE_SUB = os.environ.get('SAMPLE_SUB', None)
    
    # DINOv2 Paths
    DINO_CONFIG_PATH = os.environ.get('DINO_CONFIG_PATH', None)
    DINO_WEIGHT_PATH = os.environ.get('DINO_WEIGHT_PATH', None)
    
    # DeepLabV3+ Paths (EXP003T trained model)
    DEEPLAB_WEIGHT_PATH = os.environ.get('DEEPLAB_WEIGHT_PATH', None)
    
    # Settings
    IMG_SIZE = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Post-processing (Tuned for ensemble stability)
    ALPHA_GRAD = 0.30
    THRESHOLD_COEF = 0.30
    AREA_THRESHOLD = 400
    MEAN_THRESHOLD = 0.30
    KERNEL_CLOSE = 5
    KERNEL_OPEN = 3

CFG = Config()


def _parse_and_apply_args_to_cfg():
    """Apply CLI args to CFG (only when run as a script)."""
    parser = argparse.ArgumentParser(prog='train.py (generic template)', description='Train / inference script template')
    parser.add_argument('--test-dir', type=str, default=None, help='Path to test images')
    parser.add_argument('--sample-sub', type=str, default=None, help='Path to sample_submission.csv')
    parser.add_argument('--dino-config', type=str, default=None, help='DINO config path')
    parser.add_argument('--dino-weight', type=str, default=None, help='DINO weight file')
    parser.add_argument('--deeplab-weight', type=str, default=None, help='DeepLabV3+ weight file')
    # parse only if running as script (avoid parsing on import)
    args, _ = parser.parse_known_args()
    if args.test_dir:
        CFG.TEST_DIR = args.test_dir
    if args.sample_sub:
        CFG.SAMPLE_SUB = args.sample_sub
    if args.dino_config:
        CFG.DINO_CONFIG_PATH = args.dino_config
    if args.dino_weight:
        CFG.DINO_WEIGHT_PATH = args.dino_weight
    if args.deeplab_weight:
        CFG.DEEPLAB_WEIGHT_PATH = args.deeplab_weight


if __name__ == '__main__':
    _parse_and_apply_args_to_cfg()

# ====================================================================================================
# Model 1: DINOv2 Definitions
# ====================================================================================================
class DinoTinyDecoder(nn.Module):
    def __init__(self, in_ch=768, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_ch, 1)
        )
    def forward(self, f, size):
        return self.net(F.interpolate(f, size=size, mode="bilinear", align_corners=False))

class DinoSegmenter(nn.Module):
    def __init__(self, encoder, processor):
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.seg_head = DinoTinyDecoder(768, 1)
        
    def forward(self, img_np):
        # Preprocess inside forward for simplicity
        imgs = (img_np * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(self.encoder.device)
        with torch.no_grad():
            feats = self.encoder(**inputs).last_hidden_state
        
        B, N, C = feats.shape
        fmap = feats[:, 1:, :].permute(0, 2, 1)
        s = int(math.sqrt(N - 1))
        fmap = fmap.reshape(B, C, s, s)
        return self.seg_head(fmap, (CFG.IMG_SIZE, CFG.IMG_SIZE))

def load_dino_model():
    print("🦕 Loading DINOv2...")
    try:
        processor = AutoImageProcessor.from_pretrained(CFG.DINO_CONFIG_PATH, local_files_only=True, use_fast=False)
    except:
        # Robust fallback
        processor = BitImageProcessor(
            do_resize=True, size={"shortest_edge": 224}, do_center_crop=True, 
            crop_size={"height": 224, "width": 224}, do_rescale=True, rescale_factor=1/255,
            do_normalize=True, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
        )
    
    encoder = AutoModel.from_pretrained(CFG.DINO_CONFIG_PATH, local_files_only=True).eval()
    model = DinoSegmenter(encoder, processor)
    
    # Load weights
    if os.path.exists(CFG.DINO_WEIGHT_PATH):
        state = torch.load(CFG.DINO_WEIGHT_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("✅ DINOv2 Weights Loaded.")
    else:
        print(f"❌ DINO Weights not found: {CFG.DINO_WEIGHT_PATH}")
    
    return model.to(CFG.DEVICE).eval()

# ====================================================================================================
# Model 2: DeepLabV3+ (ResNet34)
# ====================================================================================================
# Note: segmentation_models_pytorch should be pre-installed in Kaggle environment
# If not, we'll attempt to import and fail gracefully

def load_deeplab_model():
    """
    Load DeepLabV3+ ResNet34 pre-trained model from EXP003T.
    The model was trained with SMP, so we load it directly.
    
    WHL Installation:
    - Optionally install SMP and timm from local WHL files when provided via env vars.
    - Falls back to pip install if WHL not available
    """
    print("🔬 Loading DeepLabV3+ (ResNet34)...")
    
    # Step 1: Install SMP from WHL if available
    import subprocess
    import sys
    
    # Allow user to provide local WHL files via env vars; otherwise fall back to pip
    smp_whl_path = os.environ.get('SMP_WHL_PATH', None)
    timm_whl_path = os.environ.get('TIMM_WHL_PATH', None)
    
    try:
        if smp_whl_path and timm_whl_path and os.path.exists(smp_whl_path) and os.path.exists(timm_whl_path):
            print("   📦 Installing SMP and timm from WHL files...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", smp_whl_path, timm_whl_path, "--no-deps", "-q"])
            print("   ✅ SMP and timm WHL installed successfully.")
        else:
            print(f"   ⚠️ WHL files not found or not provided, attempting pip install...")
            print(f"      SMP WHL: {smp_whl_path if smp_whl_path else 'not-provided'}")
            print(f"      timm WHL: {timm_whl_path if timm_whl_path else 'not-provided'}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "segmentation-models-pytorch", "timm", "-q"])
            print("   ✅ SMP and timm installed via pip.")
    except Exception as e:
        print(f"   ❌ Failed to install SMP/timm: {e}")
        return None
    
    # Step 2: Import SMP
    try:
        from segmentation_models_pytorch import DeepLabV3Plus
    except ImportError as e:
        print(f"❌ Failed to import segmentation_models_pytorch after install: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 3: Create and load model
    try:
        # Create model with no pre-trained weights on encoder
        # (we'll load the full model weights from checkpoint)
        model = DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        
        if os.path.exists(CFG.DEEPLAB_WEIGHT_PATH):
            # Load pre-trained weights from EXP003T
            state = torch.load(CFG.DEEPLAB_WEIGHT_PATH, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print("✅ DeepLabV3+ Weights Loaded from EXP004T (DLv3+).")
        else:
            print(f"⚠️ DeepLabV3+ Weights not found: {CFG.DEEPLAB_WEIGHT_PATH}")
            print("   Proceeding with random initialization (will likely produce poor results)")
        
        return model.to(CFG.DEVICE).eval()
    except Exception as e:
        print(f"❌ Failed to load DeepLabV3+: {e}")
        import traceback
        traceback.print_exc()
        return None

# ====================================================================================================
# Helper Functions
# ====================================================================================================

def get_prob_map(model, img_tensor, model_type="dino"):
    """Get probability map from model"""
    with torch.no_grad():
        if model_type == "dino":
            logits = model(img_tensor)
        else:  # deeplab
            logits = model(img_tensor)
            
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return prob

def tta_predict(model, pil_img, model_type="dino"):
    """3-way TTA: Original, H-Flip, V-Flip"""
    img_np = np.array(pil_img.resize((CFG.IMG_SIZE, CFG.IMG_SIZE)))
    
    if model_type == "deeplab":
        # DeepLabV3+ expects standard ImageNet normalization
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0).to(CFG.DEVICE)
    else:  # dino
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        img_t = img_t.unsqueeze(0).to(CFG.DEVICE)

    # 1. Original
    p1 = get_prob_map(model, img_t, model_type)
    
    # 2. H-Flip
    img_h = torch.flip(img_t, [3])
    p2 = get_prob_map(model, img_h, model_type)
    p2 = np.fliplr(p2)
    
    # 3. V-Flip
    img_v = torch.flip(img_t, [2])
    p3 = get_prob_map(model, img_v, model_type)
    p3 = np.flipud(p3)
    
    return (p1 + p2 + p3) / 3.0

def post_process(prob_map):
    """
    Post-process probability map:
    1. Sobel gradient enhancement
    2. Adaptive threshold
    3. Morphological operations
    """
    gx = cv2.Sobel(prob_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob_map, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    norm_mag = mag / (mag.max() + 1e-6)
    
    enhanced = (1 - CFG.ALPHA_GRAD) * prob_map + CFG.ALPHA_GRAD * norm_mag
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Thresholding
    thr = enhanced.mean() + CFG.THRESHOLD_COEF * enhanced.std()
    mask = (enhanced > thr).astype(np.uint8)
    
    # Morphology
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CFG.KERNEL_CLOSE, CFG.KERNEL_CLOSE))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CFG.KERNEL_OPEN, CFG.KERNEL_OPEN))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    
    return mask, enhanced


def compute_f1(gt_mask, pred_mask):
    """Compute F1 score between binary masks (0/1). Both arrays must be same shape."""
    if gt_mask is None:
        return 0.0
    gt = (gt_mask > 0).astype(np.uint8).flatten()
    pr = (pred_mask > 0).astype(np.uint8).flatten()
    tp = int(((gt == 1) & (pr == 1)).sum())
    fp = int(((gt == 0) & (pr == 1)).sum())
    fn = int(((gt == 1) & (pr == 0)).sum())
    if tp + fp + fn == 0:
        # Both empty -> perfect (F1=1)
        return 1.0
    if tp == 0:
        return 0.0
    f1 = 2 * tp / float(2 * tp + fp + fn)
    return float(f1)


def overlay_mask_on_image_pil(pil_img: Image.Image, mask_np: np.ndarray, color=(255, 0, 0), alpha=0.5):
    """Overlay a binary mask (numpy 0/1) on a PIL image and return a new PIL image."""
    if mask_np is None or mask_np.sum() == 0:
        return pil_img.copy()
    rgb = pil_img.convert('RGBA')
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((mask_np * 255).astype('uint8')).convert('L')
    colored = Image.new('RGBA', pil_img.size, color + (int(255 * alpha),))
    overlay.paste(colored, (0, 0), mask_img)
    composite = Image.alpha_composite(rgb, overlay).convert('RGB')
    return composite


def load_gt_mask_from_dir(mask_dir, case_id):
    """Try to load GT mask from mask_dir. Support .npy, .png, .jpg."""
    candidates = [
        os.path.join(mask_dir, f"{case_id}.npy"),
        os.path.join(mask_dir, f"{case_id}.png"),
        os.path.join(mask_dir, f"{case_id}.jpg"),
        os.path.join(mask_dir, f"{case_id}.bmp"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                if p.endswith('.npy'):
                    m = np.load(p)
                    if m.ndim == 3:
                        m = np.max(m, axis=0)
                    return (m > 0).astype(np.uint8)
                else:
                    im = Image.open(p).convert('L')
                    arr = np.array(im)
                    return (arr > 0).astype(np.uint8)
            except Exception as e:
                print(f"⚠️ Failed to load GT mask {p}: {e}")
                return None
    return None


def evaluate_on_val(model_dino, model_deeplab, val_dir, mask_dir=None, save_visuals=False, vis_outdir=None, vis_max=50, vis_criteria='worst'):
    """Run ensemble inference on validation set with GT masks, compute F1"""
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    val_files = sorted([f for f in os.listdir(val_dir) if f.lower().endswith(valid_exts)])
    if len(val_files) == 0:
        print(f"⚠️ No images found in val_dir: {val_dir}")
        return [], {}

    print(f"🔍 Running ensemble validation on {len(val_files)} images from {val_dir}")
    oof_rows = []
    vis_candidates = []
    for f in tqdm(val_files, desc='Val inference'):
        pil_img = Image.open(os.path.join(val_dir, f)).convert('RGB')
        case_id = Path(f).stem
        
        # DINOv2 inference
        prob_dino = tta_predict(model_dino, pil_img, 'dino')
        
        # DeepLabV3+ inference
        prob_deeplab = tta_predict(model_deeplab, pil_img, 'deeplab')
        
        # Ensemble blend
        prob_avg = (prob_dino * CFG.W_DINO) + (prob_deeplab * CFG.W_DEEPLAB)
        
        # Post-process
        mask, enhanced = post_process(prob_avg)
        mask_resized = cv2.resize(mask, pil_img.size, interpolation=cv2.INTER_NEAREST)
        
        # Load GT mask
        gt_mask = None
        if mask_dir:
            gt_mask = load_gt_mask_from_dir(mask_dir, case_id)
            if gt_mask is not None and gt_mask.shape != mask_resized.shape:
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), pil_img.size, interpolation=cv2.INTER_NEAREST)
        
        f1 = compute_f1(gt_mask, mask_resized)
        oof_rows.append({
            'case_id': case_id,
            'f1': f1,
            'pred_label': 1 if mask_resized.sum() > 0 else 0,
            'prob_ensemble_mean': float(prob_avg.mean()),
            'area': int(mask_resized.sum()),
        })
        
        # Collect visuals
        vis_candidates.append({
            'case_id': case_id,
            'pil': pil_img,
            'prob_dino': prob_dino,
            'prob_deeplab': prob_deeplab,
            'prob_ensemble': prob_avg,
            'mask_resized': mask_resized,
            'gt_mask': gt_mask,
            'f1': float(f1)
        })

    # Summary metric
    f1_values = [r['f1'] for r in oof_rows if r['f1'] is not None]
    mean_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    metrics = {'mean_f1': mean_f1, 'n_images': len(val_files)}
    print(f"🔎 Validation mean F1: {mean_f1:.4f} over {len(val_files)} images")

    # Save visuals
    if save_visuals and len(vis_candidates) > 0:
        os.makedirs(vis_outdir, exist_ok=True)
        def score_key_v(v):
            return v.get('f1', v['prob_ensemble'].mean())
        if vis_criteria == 'worst':
            sorted_list = sorted(vis_candidates, key=lambda x: score_key_v(x))
        elif vis_criteria == 'best':
            sorted_list = sorted(vis_candidates, key=lambda x: score_key_v(x), reverse=True)
        else:
            sorted_list = random.sample(vis_candidates, min(vis_max, len(vis_candidates)))
        to_save = sorted_list[:vis_max]
        print(f"📷 Saving {len(to_save)} visuals to {vis_outdir}")
        saved = 0
        for v in to_save:
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                axes[0,0].imshow(v['pil']); axes[0,0].set_title('Original'); axes[0,0].axis('off')
                # GT overlay
                if v['gt_mask'] is not None:
                    gt_overlay = overlay_mask_on_image_pil(v['pil'], v['gt_mask'], color=(0,255,0), alpha=0.5)
                    axes[0,1].imshow(gt_overlay); axes[0,1].set_title('GT overlay'); axes[0,1].axis('off')
                else:
                    axes[0,1].imshow(v['pil']); axes[0,1].set_title('GT (none)'); axes[0,1].axis('off')
                # DINOv2 heatmap
                prob_resized_dino = cv2.resize(v['prob_dino'], v['pil'].size, interpolation=cv2.INTER_LINEAR)
                axes[0,2].imshow(prob_resized_dino, cmap='hot'); axes[0,2].set_title('DINOv2 prob'); axes[0,2].axis('off')
                # DeepLabV3+ heatmap
                prob_resized_deeplab = cv2.resize(v['prob_deeplab'], v['pil'].size, interpolation=cv2.INTER_LINEAR)
                axes[1,0].imshow(prob_resized_deeplab, cmap='hot'); axes[1,0].set_title('DeepLabV3+ prob'); axes[1,0].axis('off')
                # Ensemble heatmap
                prob_resized = cv2.resize(v['prob_ensemble'], v['pil'].size, interpolation=cv2.INTER_LINEAR)
                axes[1,1].imshow(prob_resized, cmap='hot'); axes[1,1].set_title('Ensemble prob'); axes[1,1].axis('off')
                # Ensemble overlay
                ensemble_overlay = overlay_mask_on_image_pil(v['pil'], v['mask_resized'], color=(255,0,0), alpha=0.5)
                axes[1,2].imshow(ensemble_overlay); axes[1,2].set_title(f"Ensemble overlay (F1={v['f1']:.4f})"); axes[1,2].axis('off')
                plt.tight_layout()
                fpath = os.path.join(vis_outdir, f"{v['case_id']}_val_vis.png")
                fig.savefig(fpath, dpi=100, bbox_inches='tight')
                plt.close(fig)
                saved += 1
            except Exception as e:
                print(f"⚠️ Failed to save val visual for {v['case_id']}: {e}")
        print(f"✅ Saved {saved} val visuals to {vis_outdir}")

    return oof_rows, metrics

def rle_encode(mask):
    pixels = mask.T.flatten()
    dots = np.where(pixels == 1)[0]
    if len(dots) == 0: return "authentic"
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1: run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return json.dumps([int(x) for x in run_lengths])

# ====================================================================================================
# Artifacts & Metrics
# ====================================================================================================
ARTIFACTS_DIR = "/kaggle/working/exp004tv3-artifacts"
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
VIS_DIR = os.path.join(ARTIFACTS_DIR, "visuals")

def setup_artifacts():
    """Create artifact directories"""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    return ARTIFACTS_DIR

def save_metrics(results_df, metrics_dict):
    """Save overall metrics as JSON"""
    metrics_path = os.path.join(ARTIFACTS_DIR, "overall_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")

def save_oof_predictions(oof_data):
    """Save OOF predictions (for blend analysis)"""
    oof_path = os.path.join(ARTIFACTS_DIR, "oof_all.csv")
    oof_df = pd.DataFrame(oof_data)
    oof_df.to_csv(oof_path, index=False)
    print(f"✅ OOF saved: {oof_path}")

def save_visualization(case_id, pil_img, prob_dino, prob_deeplab, prob_ensemble, 
                      mask_final, vis_outdir, idx=0):
    """Save 6-panel visualization: Original, DINO, DeepLab, Ensemble heatmap, GT overlay, Ensemble overlay"""
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Original
        axes[0, 0].imshow(pil_img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        
        # DINO Prob Map
        axes[0, 1].imshow(prob_dino, cmap="hot")
        axes[0, 1].set_title(f"DINOv2 (W={CFG.W_DINO})")
        axes[0, 1].axis("off")
        
        # DeepLabV3+ Prob Map
        axes[0, 2].imshow(prob_deeplab, cmap="hot")
        axes[0, 2].set_title(f"DeepLabV3+ (W={CFG.W_DEEPLAB})")
        axes[0, 2].axis("off")
        
        # Ensemble Prob Map
        axes[1, 0].imshow(prob_ensemble, cmap="hot")
        axes[1, 0].set_title("Ensemble Probability")
        axes[1, 0].axis("off")
        
        # Ensemble Mask Overlay
        ensemble_overlay = overlay_mask_on_image_pil(pil_img, mask_final, color=(255, 0, 0), alpha=0.5)
        axes[1, 1].imshow(ensemble_overlay)
        axes[1, 1].set_title("Ensemble Prediction")
        axes[1, 1].axis("off")
        
        # Comparison text
        axes[1, 2].axis("off")
        text_content = f"Case ID: {case_id}\n"
        text_content += f"Ensemble Strategy:\nDINOv2 (0.6) + DeepLabV3+ (0.4)\n"
        text_content += f"Expected: Breaking through 0.330"
        axes[1, 2].text(0.1, 0.5, text_content, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(f"{case_id}")
        plt.tight_layout()
        fpath = os.path.join(vis_outdir, f"{case_id}_ensemble.png")
        fig.savefig(fpath, dpi=100, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Failed to save visualization for {case_id}: {e}")

def save_run_json(metrics_dict):
    """Save lightweight run.json for aggregator compatibility"""
    run_path = os.path.join(ARTIFACTS_DIR, "run.json")
    run = {
        "metric_version": "v1",
        "experiment_id": metrics_dict.get("experiment_id", "EXP004Ev2"),
        "description": metrics_dict.get("description", ""),
        "status": "completed",
        "predictions": metrics_dict.get("predictions", {}),
        "ensemble_weights": metrics_dict.get("ensemble_weights", {}),
        "post_processing": metrics_dict.get("post_processing", {}),
        "created_at": metrics_dict.get("timestamp", pd.Timestamp.now().isoformat()),
    }
    with open(run_path, "w") as f:
        json.dump(run, f, indent=2)
    print(f"✅ Run saved: {run_path}")

def copy_submission_to_root(src_path):
    """Copy submission.csv to working dir root for Kaggle submission convenience"""
    try:
        dst_root = "/kaggle/working/submission.csv" if os.path.exists("/kaggle/working") else "submission.csv"
        shutil.copy(src_path, dst_root)
        print(f"✅ Copied submission to: {dst_root}")
    except Exception as e:
        print(f"⚠️ Failed to copy submission.csv to root: {e}")

# ====================================================================================================
# Main Execution
# ====================================================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP004Tv3 Ensemble Inference: DINOv2 + DeepLabV3+")
    parser.add_argument("--save-visuals", action="store_true", help="Save analysis visuals")
    parser.add_argument("--vis-max", type=int, default=50, help="Max number of visuals to save (single-mode)")
    parser.add_argument("--vis-criteria", type=str, choices=["worst", "best", "random"], default="random",
                        help="How to select images for visualization (single-mode)")
    # Multi-mode counts (override single-mode when any is set >0)
    parser.add_argument("--vis-worst", type=int, default=50, help="Save worst N visuals by score (production default=50)")
    parser.add_argument("--vis-best", type=int, default=50, help="Save best N visuals by score (production default=50)")
    parser.add_argument("--vis-random", type=int, default=0, help="Save random N visuals")
    parser.add_argument("--copy-submission-root", action="store_true", help="Copy submission.csv to working dir root")
    parser.add_argument("--test-dir", type=str, default=None, help="Override default TEST_DIR where test images are read from")
    parser.add_argument("--eval-val", action='store_true', help='Run ensemble evaluation on a validation directory with GT masks')
    parser.add_argument("--val-dir", type=str, default=None, help='Validation images directory')
    parser.add_argument("--val-mask-dir", type=str, default=None, help='Validation masks directory')
    parser.add_argument("--allow-visual-duplicates", action="store_true", help="Allow saving visuals with duplicate case_ids (separate files per category) - default: false")
    args, _ = parser.parse_known_args()
    # Setup artifacts
    setup_artifacts()
    
    # If CLI override test dir specified, use it
    if args.test_dir:
        if os.path.exists(args.test_dir):
            print(f"🔧 Overriding test dir: {CFG.TEST_DIR} -> {args.test_dir}")
            CFG.TEST_DIR = args.test_dir
        else:
            print(f"⚠️ Provided --test-dir does not exist: {args.test_dir} (keeping default: {CFG.TEST_DIR})")

    # Load Models
    model_dino = load_dino_model()
    model_deeplab = load_deeplab_model()
    
    if model_deeplab is None:
        print("❌ Failed to load DeepLabV3+ model. Aborting.")
        return

    # If user requested validation mode
    if args.eval_val:
        if args.val_dir is None or args.val_mask_dir is None:
            print("⚠️ --eval-val requires --val-dir and --val-mask-dir to be specified")
            return
        vis_outdir = VIS_DIR
        oof_rows, metrics_val = evaluate_on_val(model_dino, model_deeplab, args.val_dir, mask_dir=args.val_mask_dir, 
                                                 save_visuals=args.save_visuals, vis_outdir=vis_outdir, 
                                                 vis_max=args.vis_max, vis_criteria=args.vis_criteria)
        # Save OOF and metrics
        val_oof_path = os.path.join(ARTIFACTS_DIR, 'val_oof.csv')
        pd.DataFrame(oof_rows).to_csv(val_oof_path, index=False)
        print(f"✅ Val OOF saved: {val_oof_path}")
        metrics_path = os.path.join(ARTIFACTS_DIR, 'val_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_val, f, indent=2)
        print(f"✅ Val metrics saved: {metrics_path}")
        save_run_json({'experiment_id': 'EXP004Tv3_val', 'predictions': {'n_images': metrics_val.get('n_images', 0)}, 'metrics': metrics_val})
        return
    
    # Test set inference
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    try:
        test_files = sorted([f for f in os.listdir(CFG.TEST_DIR) if f.lower().endswith(valid_exts)])
    except Exception as e:
        print(f"⚠️ Failed to list TEST_DIR ({CFG.TEST_DIR}): {e}")
        test_files = []
    # Debug: print a small sample of files found
    print(f"🔍 Test dir: {CFG.TEST_DIR} contains {len(test_files)} image(s). Sample: {test_files[:10]}")
    if len(test_files) <= 1:
        print("⚠️ Note: Only 1 (or none) test image found. This explains why only 1 visual was generated.")
        print("   If you intend to generate many visuals, make sure the path is correct and contains multiple images.")
        print("   Tip: use --test-dir to override the test dir (e.g., --test-dir /kaggle/input/.../test_images)")
    print(f"🚀 Starting Ensemble Inference on {len(test_files)} images...")
    
    results = []
    oof_data = []
    vis_candidates = []
    save_visuals = args.save_visuals
    MAX_VIS = args.vis_max
    
    for f in tqdm(test_files):
        pil_img = Image.open(os.path.join(CFG.TEST_DIR, f)).convert("RGB")
        case_id = Path(f).stem
        
        # DINOv2 inference
        prob_dino = tta_predict(model_dino, pil_img, "dino")
        
        # DeepLabV3+ inference
        prob_deeplab = tta_predict(model_deeplab, pil_img, "deeplab")
        
        # Ensemble blend: DINOv2 0.6 + DeepLabV3+ 0.4
        prob_avg = (prob_dino * CFG.W_DINO) + (prob_deeplab * CFG.W_DEEPLAB)
        
        # Post-processing
        mask, enhanced = post_process(prob_avg)
        
        # Resize back to original size
        mask_resized = cv2.resize(mask, pil_img.size, interpolation=cv2.INTER_NEAREST)
        prob_resized = cv2.resize(prob_avg, pil_img.size, interpolation=cv2.INTER_LINEAR)
        
        # Confidence Filter
        area = int(mask_resized.sum())
        
        # Mean inside mask
        if mask.sum() > 0:
            mean_score = prob_avg[mask == 1].mean()
        else:
            mean_score = 0.0
            
        if area < CFG.AREA_THRESHOLD or mean_score < CFG.MEAN_THRESHOLD:
            annot = "authentic"
            pred_label = 0
        else:
            annot = rle_encode(mask_resized)
            pred_label = 1
            
        results.append({"case_id": case_id, "annotation": annot})
        
        # OOF data
        oof_data.append({
            "case_id": case_id,
            "prob_dino_mean": float(prob_dino.mean()),
            "prob_deeplab_mean": float(prob_deeplab.mean()),
            "prob_ensemble_mean": float(prob_avg.mean()),
            "pred_label": pred_label,
            "area": area,
            "mean_inside_mask": float(mean_score),
        })
        
        # Collect candidates for post-hoc visualization selection
        vis_candidates.append({
            "case_id": case_id,
            "file_path": os.path.join(CFG.TEST_DIR, f),
            "pred_label": pred_label,
            "area": area,
            "mean_inside_mask": float(mean_score),
            "prob_ensemble_mean": float(prob_avg.mean()),
        })
        
    # Save Results
    df = pd.DataFrame(results)
    sample = pd.read_csv(CFG.SAMPLE_SUB)
    sample['case_id'] = sample['case_id'].astype(str)
    df['case_id'] = df['case_id'].astype(str)
    
    final = sample[['case_id']].merge(df, on='case_id', how='left').fillna("authentic")
    sub_path = os.path.join(ARTIFACTS_DIR, "submission.csv")
    final.to_csv(sub_path, index=False)
    print(f"✅ Submission saved: {sub_path}")
    # Always copy submission to root for Kaggle submission convenience
    copy_submission_to_root(sub_path)
    
    # Save OOF Predictions
    save_oof_predictions(oof_data)
    
    # Save metrics
    metrics = {
        "experiment_id": "EXP004Tv3",
        "description": "Ensemble: DINOv2 (0.6) + DeepLabV3+ (0.4) — Breaking through 0.330 🎬",
        "ensemble_weights": {
            "dino": CFG.W_DINO,
            "deeplab": CFG.W_DEEPLAB
        },
        "post_processing": {
            "alpha_grad": CFG.ALPHA_GRAD,
            "threshold_coef": CFG.THRESHOLD_COEF,
            "area_threshold": CFG.AREA_THRESHOLD,
            "mean_threshold": CFG.MEAN_THRESHOLD,
            "kernel_close": CFG.KERNEL_CLOSE,
            "kernel_open": CFG.KERNEL_OPEN,
        },
        "predictions": {
            "total_test_images": len(test_files),
            "predicted_forged": int((df['annotation'] != 'authentic').sum()),
            "predicted_authentic": int((df['annotation'] == 'authentic').sum()),
        },
        "artifacts_dir": ARTIFACTS_DIR,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    save_metrics(df, metrics)
    save_run_json(metrics)

    # Save selected visuals (post-hoc selection similar to EXP005T)
    vis_saved = 0
    if save_visuals and len(vis_candidates) > 0:
        # Debug: Show number of collected candidates
        print(f"🔍 Collected {len(vis_candidates)} visual candidates from {len(test_files)} test images")
        # Score function: forged → mean_inside_mask, authentic → prob_ensemble_mean
        def score_key(item):
            return item["mean_inside_mask"] if item["pred_label"] == 1 else item["prob_ensemble_mean"]

        selected = []
        selection_rows = []

        # Multi-mode selection if any of vis-worst/best/random is set
        if args.vis_worst > 0 or args.vis_best > 0 or args.vis_random > 0:
            worst_sel = sorted(vis_candidates, key=score_key)[:max(0, args.vis_worst)]
            best_sel = sorted(vis_candidates, key=score_key, reverse=True)[:max(0, args.vis_best)]
            rand_pool = vis_candidates.copy()
            random.shuffle(rand_pool)
            rand_sel = rand_pool[:max(0, args.vis_random)]

            # Combine and optionally dedupe by case_id
            worst_ids = [it["case_id"] for it in worst_sel]
            best_ids = [it["case_id"] for it in best_sel]
            rand_ids = [it["case_id"] for it in rand_sel]
            print(f"🔍 worst_ids sample: {worst_ids[:10]}, best_ids sample: {best_ids[:10]}, random_ids sample: {rand_ids[:10]}")
            if args.allow_visual_duplicates:
                # Keep duplicates - preserve category for save time name
                for cat, group in [("worst", worst_sel), ("best", best_sel), ("random", rand_sel)]:
                    for it in group:
                        selected.append({**it, "selection": cat})
                        selection_rows.append({
                            "case_id": it["case_id"],
                            "selection": cat,
                            "score": score_key(it),
                            "pred_label": it["pred_label"],
                            "area": it["area"],
                            "mean_inside_mask": it["mean_inside_mask"],
                            "prob_ensemble_mean": it["prob_ensemble_mean"],
                        })
            else:
                seen = set()
                for cat, group in [("worst", worst_sel), ("best", best_sel), ("random", rand_sel)]:
                    for it in group:
                        if it["case_id"] in seen:
                            continue
                        seen.add(it["case_id"])
                        selected.append({**it, "selection": cat})
                        selection_rows.append({
                            "case_id": it["case_id"],
                            "selection": cat,
                            "score": score_key(it),
                            "pred_label": it["pred_label"],
                            "area": it["area"],
                            "mean_inside_mask": it["mean_inside_mask"],
                            "prob_ensemble_mean": it["prob_ensemble_mean"],
                        })
            # Debug: show breakdown
            print(f"🔍 Selected worst: {len(worst_sel)}, best: {len(best_sel)}, random: {len(rand_sel)} (deduped to {len(selected)})")
        else:
            # Single-mode fallback
            if MAX_VIS > 0:
                if args.vis_criteria == "best":
                    selected = sorted(vis_candidates, key=score_key, reverse=True)[:MAX_VIS]
                    select_tag = "best"
                elif args.vis_criteria == "worst":
                    selected = sorted(vis_candidates, key=score_key)[:MAX_VIS]
                    select_tag = "worst"
                else:
                    random.shuffle(vis_candidates)
                    selected = vis_candidates[:MAX_VIS]
                    select_tag = "random"
                for it in selected:
                    selection_rows.append({
                        "case_id": it["case_id"],
                        "selection": select_tag,
                        "score": score_key(it),
                        "pred_label": it["pred_label"],
                        "area": it["area"],
                        "mean_inside_mask": it["mean_inside_mask"],
                        "prob_ensemble_mean": it["prob_ensemble_mean"],
                    })

        # Save an index CSV of selected visuals
        try:
            if len(selection_rows) > 0:
                idx_path = os.path.join(VIS_DIR, "visuals_index.csv")
                pd.DataFrame(selection_rows).to_csv(idx_path, index=False)
                print(f"✅ Visuals index saved: {idx_path}")
                # Debug: list a small subset of selected case ids
                try:
                    sample_selected_ids = [r['case_id'] for r in selection_rows[:20]]
                    print(f"🔍 Sample selected case_ids (first 20): {sample_selected_ids}")
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Failed to save visuals index: {e}")

        # Render visuals
        if len(selected) == 0:
            print("⚠️ No visuals selected")
        else:
            print(f"🎯 Rendering {len(selected)} visuals...")
        for idx_item, item in enumerate(selected):
            try:
                pil_img = Image.open(item["file_path"]).convert("RGB")
                prob_dino = tta_predict(model_dino, pil_img, "dino")
                prob_deeplab = tta_predict(model_deeplab, pil_img, "deeplab")
                prob_avg = (prob_dino * CFG.W_DINO) + (prob_deeplab * CFG.W_DEEPLAB)
                mask, _ = post_process(prob_avg)
                mask_resized = cv2.resize(mask, pil_img.size, interpolation=cv2.INTER_NEAREST)
                prob_resized = cv2.resize(prob_avg, pil_img.size, interpolation=cv2.INTER_LINEAR)
                if args.allow_visual_duplicates and "selection" in item:
                    save_visualization(f"{item['case_id']}_{item['selection']}_{idx_item}", pil_img, prob_dino, prob_deeplab, prob_resized, mask_resized, VIS_DIR, vis_saved)
                else:
                    save_visualization(item["case_id"], pil_img, prob_dino, prob_deeplab, prob_resized, mask_resized, VIS_DIR, vis_saved)
                vis_saved += 1
            except Exception as e:
                import traceback
                print(f"⚠️ Visualization failed for {item['case_id']}: {e}")
                print(traceback.format_exc())
    
    # Summary
    print(f"\n{'='*100}")
    print(f"✅ EXP004Tv3 Ensemble Inference Complete!")
    print(f"{'='*100}")
    print(f"🎬 DINOv2 (0.6) + DeepLabV3+ (0.4) Chemical Reaction Results:")
    print(f"{'='*100}")
    print(f"📊 Results Summary:")
    print(f"   Total Test Images: {len(test_files)}")
    print(f"   Predicted Forged: {metrics['predictions']['predicted_forged']}")
    print(f"   Predicted Authentic: {metrics['predictions']['predicted_authentic']}")
    print(f"\n📂 Artifacts Saved:")
    print(f"   Submission: {os.path.join(ARTIFACTS_DIR, 'submission.csv')}")
    print(f"   OOF Predictions: {os.path.join(ARTIFACTS_DIR, 'oof_all.csv')}")
    print(f"   Metrics: {os.path.join(ARTIFACTS_DIR, 'overall_metrics.json')}")
    print(f"   Visualizations: {VIS_DIR} ({vis_saved} images)")
    print(f"\n🎯 Strategy: Breaking through the 0.330 barrier with ensemble strength!")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    print(f"\n{'='*100}")
    print(f"🎬 EXP004Tv3 — DINOv2 + DeepLabV3+ Ensemble Inference")
    print(f"{'='*100}\n")
    main()
    print(f"\n{'='*100}")
    print(f"✅ Ensemble inference completed! さあ、試写会（--eval-val）で確認しましょう！")
    print(f"{'='*100}\n")