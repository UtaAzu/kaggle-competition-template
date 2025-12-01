# 画像セグメンテーション用・自己完結型・訓練パイプライン作成ガイドライン（汎用版：ピクセル/特徴量両対応）

---

## 目的・前提

- Kaggle/ローカル両対応の「自己完結型」学習パイプラインを作成する
- **複数fold（CV）時は「foldごと成果物」と「全体集計成果物（overall_metrics.json等）」の両方を必ず生成・保存すること**
- 成果物は以下の場所に保存する（上位に余計な `<EXP_ID>` ディレクトリを作らない）
    - ローカル: `experiments/<exp_id>-artifacts/`
    - Kaggle: `/kaggle/working/<exp_id>-artifacts/`
    - foldごとは `fold0/`, `fold1/`, ...、全体集計は `*-artifacts/` 直下に配置
- validate.pyや検証Notebookで全体集計成果物を自動生成・確認する

---

## 0. 🏷️ メタ情報ブロック (Meta Information Block)

- 実験番号、タイトル、目的、日付、著者、概要を最上部に明記
- 例（汎用テンプレート）

```python
# === EXPXXX_vY: [実験の目的を簡潔に書く] ===
# Date: 2025-XX-XX
# Author: UtaAzu
# Purpose: [セグメンテーション/偽造検出/その他の目的]
# Strategy: [pixel_based | feature_based]
# Dataset: [データセット名]
# Expected Outcome: [期待するCV/LB目標]
# Notes: [特記事項]
```

---

## 1. 📝 設定ブロック (Configuration Block) — 戦略デスク

- **最上位戦略フラグ `TRAINING_MODE`** で学習方式を切り替え
- **ピクセルベース（`pixel_based`）**: 画像を直接入力（U-Net, SMP等）
- **特徴量ベース（`feature_based`）**: 事前抽出特徴量を入力（DINOv2等）
- **標準兵器（クラス不均衡対策）をデフォルトON**

```python
from pathlib import Path
from datetime import datetime
import torch

class Config:
    EXP_ID = "EXPXXX_vY"
    DESCRIPTION = "[実験の目的を簡潔に]"
    DATE = datetime.now().strftime('%Y-%m-%d')

    # === 最上位戦略フラグ ===
    # "pixel_based": 画像を直接入力（U-Net, SMP等）
    # "feature_based": 事前抽出した特徴量を入力（DINOv2特徴量等）
    TRAINING_MODE = "feature_based"  # or "pixel_based"

    # === ピクセルベース学習用の設定 ===
    class PixelConfig:
        BACKBONE = "efficientnet-b0"  # or "resnet34", "mobilenet_v3_small"
        MODEL_TYPE = "smp.Unet"  # "smp.Unet", "FastUNet", "custom_cnn"
        IMAGE_SIZE = 384
        AUGMENTATIONS = "medium"  # "light", "medium", "heavy", "none"

    # === 特徴量ベース学習用の設定 ===
    class FeatureConfig:
        ENCODER_ID = "dino_v2_base"  # 特徴量セットのID（記録用）
        FEATURE_DIR = Path('/kaggle/input/exp003t-dino-v2-features') if Path('/kaggle/input').exists() else Path('./features/exp003t-dino-v2')
        FEATURE_DIM = 768  # DINOv2 small:384, base:768, large:1024
        FEATURE_SUFFIX = ".npy"  # 特徴量ファイルの拡張子
        DECODER_TYPE = "SimpleDecoder"  # "SimpleDecoder", "LightUNetDecoder", "ConvHead"

    # === 全モード共通の学習設定 ===
    class TrainConfig:
        NUM_EPOCHS = 10
        BATCH_SIZE = 16
        LEARNING_RATE = 1e-3
        WEIGHT_DECAY = 1e-4

        # --- 標準兵器：クラス不均衡対策（デフォルトON） ---
        USE_WEIGHTED_SAMPLER = True
        FORGED_SAMPLE_WEIGHT = 3.0  # forged画像を何倍重視するか

        USE_WEIGHTED_LOSS = True
        POS_WEIGHT = 20.0  # 不正ピクセルへのペナルティ倍率

    # === 環境自動切替 ===
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        base = Path('/kaggle/input')
        candidates = [d for d in base.iterdir() if d.is_dir()]
        DATA_DIR = candidates[0] if candidates else base
        ARTIFACTS_ROOT = Path('/kaggle/working')
        DEBUG = False
    else:
        DATA_DIR = Path('./')
        ARTIFACTS_ROOT = Path('experiments')
        DEBUG = True

    # === Paths ===
    AUTHENTIC_DIR = DATA_DIR / 'train_images' / 'authentic'
    FORGED_DIR = DATA_DIR / 'train_images' / 'forged'
    MASKS_DIR = DATA_DIR / 'train_masks'
    TEST_DIR = DATA_DIR / 'test_images'
    SAMPLE_SUBMISSION_PATH = DATA_DIR / 'sample_submission.csv'

    # === CV ===
    N_SPLITS = 5
    GROUP_COL = 'case_id'
    USE_FIRST_FOLD_ONLY = True
    RANDOM_STATE = 42

    # === Post-processing（グリッドサーチ対象） ===
    CONFIDENCE_THRESHOLDS_GRID = [0.15, 0.30, 0.50, 0.70]
    MIN_AREA_GRID = [10, 25, 50, 100]
    MORPH_KERNEL_SIZE_GRID = [0, 3, 5]

    # === 退化解ガード（強制停止閾値） ===
    MIN_FORGED_NONEMPTY_RATIO = 0.20
    MAX_AUTHENTIC_FP_RATIO = 0.30

    # === Fallback（推論で空なら緩和再試行） ===
    FALLBACK_ON_EMPTY = True
    FALLBACK_MASK_TH = 0.20
    FALLBACK_MIN_AREA = 10
    FALLBACK_MORPH_KERNEL = 0

    # === Device ===
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Artifacts ===
    ARTIFACTS_DIR = ARTIFACTS_ROOT / f'{EXP_ID.lower().replace("_", "-")}-artifacts'
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR = ARTIFACTS_DIR / 'models'
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = ARTIFACTS_DIR  # 互換: 既存コードがOUTPUT_DIRを参照する場合に備える
```

---

## 2. ⏱️ ロギングブロック (Logging Block)

- コンソール＋ファイル出力
- バージョン情報（主要ライブラリ）を記録

```python
import sys

class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.info("=== Environment Information ===")
        self.info(f"Python: {sys.version}")
        self.info(f"PyTorch: {torch.__version__}")
        self.info(f"Device: {Config.DEVICE}")
        self.info("=" * 50)
        for lib in ['torch', 'numpy', 'pandas', 'opencv-python']:
            try:
                if lib == 'opencv-python':
                    import cv2
                    self.info(f"opencv: {cv2.__version__}")
                else:
                    self.info(f"{lib}: {__import__(lib).__version__}")
            except Exception:
                self.info(f"{lib}: n/a")

    def info(self, message: str):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {message}"
        print(line)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
```

---

## 3. 🔧 データ処理ブロック (Data Processing Block) — 調理場

- **`TRAINING_MODE`に応じて適切なDatasetを返すファクトリー**を導入
- ピクセルベース: `PixelDataset`
- 特徴量ベース: `FeatureDataset`

```python
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset

def prepare_dataframe(config: Config) -> pd.DataFrame:
    """
    画像パスとマスクパスを含むDataFrameを作成
    """
    rows = []
    # authentic
    for p in sorted(config.AUTHENTIC_DIR.glob('*.png')):
        image_id = p.stem
        case_id = image_id.split('_')[0]
        rows.append({
            'image_id': image_id,
            'case_id': case_id,
            'is_forged': 0,
            'image_path': str(p),
            'mask_path': None
        })
    # forged
    for p in sorted(config.FORGED_DIR.glob('*.png')):
        image_id = p.stem
        case_id = image_id.split('_')[0]
        mask_path = config.MASKS_DIR / f"{image_id}.npy"
        rows.append({
            'image_id': image_id,
            'case_id': case_id,
            'is_forged': 1,
            'image_path': str(p),
            'mask_path': str(mask_path) if mask_path.exists() else None
        })
    return pd.DataFrame(rows)

# === PixelDataset（画像から学習） ===
class PixelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: Config.PixelConfig, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_size = config.IMAGE_SIZE
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        if pd.notna(row['mask_path']) and Path(row['mask_path']).exists():
            mask = np.load(row['mask_path']).astype(np.uint8)
            if mask.ndim == 3:
                mask = mask.max(axis=0)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask, idx

# === FeatureDataset（特徴量から学習） ===
class FeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: Config.FeatureConfig):
        self.df = df.reset_index(drop=True)
        self.feature_dir = config.FEATURE_DIR
        self.feature_dim = config.FEATURE_DIM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # 特徴量読み込み（train_features/{authentic|forged}/{image_id}.npy想定）
        label = 'forged' if row['is_forged'] == 1 else 'authentic'
        feat_path = self.feature_dir / 'train_features' / label / f"{row['image_id']}.npy"
        feat = np.load(feat_path).astype(np.float32)
        if feat.ndim == 3 and feat.shape[0] != self.feature_dim and feat.shape[-1] == self.feature_dim:
            feat = feat.transpose(2, 0, 1)  # (H,W,C)->(C,H,W)
        feat_t = torch.from_numpy(feat)

        # GT mask
        if pd.notna(row['mask_path']) and Path(row['mask_path']).exists():
            mask = np.load(row['mask_path']).astype(np.uint8)
            if mask.ndim == 3:
                mask = mask.max(axis=0)
            _, H, W = feat_t.shape
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
        else:
            _, H, W = feat_t.shape
            mask = np.zeros((H, W), dtype=np.float32)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return feat_t, mask_t, idx

# === Datasetを生成する「工場」 ===
def get_dataset(df: pd.DataFrame, config: Config, is_train: bool):
    if config.TRAINING_MODE == "pixel_based":
        transforms = get_augmentations(config.PixelConfig.AUGMENTATIONS) if is_train else None
        return PixelDataset(df, config.PixelConfig, transforms=transforms)
    elif config.TRAINING_MODE == "feature_based":
        return FeatureDataset(df, config.FeatureConfig)
    else:
        raise ValueError(f"Unknown TRAINING_MODE: {config.TRAINING_MODE}")

def get_augmentations(level: str):
    """Augmentationレベルに応じた変換（albumentations使用想定）"""
    # 実装例（albumentations省略時はNoneを返す）
    if level == "none":
        return None
    # elif level == "light": return A.Compose([A.HorizontalFlip(p=0.5)])
    # elif level == "medium": return A.Compose([...])
    # elif level == "heavy": return A.Compose([...])
    return None
```

---

## 4. 🤖 モデル定義ブロック (Model Definition Block) — 俳優事務所

- **`TRAINING_MODE`とモデルタイプに応じて適切なモデルを返すファクトリー**を導入

```python
import torch.nn as nn

# === SimpleDecoder（特徴量ベース用） ===
class SimpleDecoder(nn.Module):
    def __init__(self, in_channels: int = 768, mid_channels: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.head(x))

# === FastUNet（ピクセルベース用・軽量） ===
class FastUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.bottleneck = self.conv_block(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = self.conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self.conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = self.conv_block(64, 32)
        self.out = nn.Conv2d(32, out_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return torch.sigmoid(self.out(d1))

# === モデルを生成する「俳優事務所」 ===
def build_model(config: Config):
    if config.TRAINING_MODE == "pixel_based":
        if config.PixelConfig.MODEL_TYPE == "FastUNet":
            return FastUNet(in_channels=3, out_channels=1).to(config.DEVICE)
        # elif config.PixelConfig.MODEL_TYPE == "smp.Unet":
        #     import segmentation_models_pytorch as smp
        #     return smp.Unet(encoder_name=config.PixelConfig.BACKBONE, ...).to(config.DEVICE)
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {config.PixelConfig.MODEL_TYPE}")

    elif config.TRAINING_MODE == "feature_based":
        if config.FeatureConfig.DECODER_TYPE == "SimpleDecoder":
            return SimpleDecoder(in_channels=config.FeatureConfig.FEATURE_DIM).to(config.DEVICE)
        # elif config.FeatureConfig.DECODER_TYPE == "LightUNetDecoder":
        #     return LightUNetDecoder(...).to(config.DEVICE)
        else:
            raise ValueError(f"Unknown DECODER_TYPE: {config.FeatureConfig.DECODER_TYPE}")

    else:
        raise ValueError(f"Unknown TRAINING_MODE: {config.TRAINING_MODE}")
```

---

## 5. 🎓 学習・評価関数ブロック (Training & Evaluation)

- **標準兵器（WeightedBCEDiceLoss）を標準実装**
- train_epochは従来通り（BCE/Dice等）
- validateはoOF1ベースのvalidate_gridを採用

```python
from tqdm import tqdm

# === 標準兵器：WeightedBCEDiceLoss ===
class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        # Weighted BCE
        bce_loss = self.bce(pred, target)
        weight_map = torch.where(target > 0.5, self.pos_weight, 1.0)
        bce_loss = (bce_loss * weight_map).mean()

        # Dice
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for feats, masks, _ in tqdm(loader, desc="Training"):
        feats, masks = feats.to(device), masks.to(device)
        optimizer.zero_grad()
        pred = model(feats)
        loss = criterion(pred, masks)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def mask_f1_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool).ravel()
    gt = gt.astype(bool).ravel()
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    if tp + fp + fn == 0:
        return 1.0
    return float(2.0 * tp / (2.0 * tp + fp + fn + 1e-9))

def validate_grid(model, loader, thresholds, min_area_grid, morph_kernel_grid, device, ref_df):
    model.eval()
    best_cfg, best_mean_f1 = None, -1.0
    results = []
    with torch.no_grad():
        for min_area in min_area_grid:
            for morph_k in morph_kernel_grid:
                rows = []
                agg = {float(t): [] for t in thresholds}
                forged_nonempty, authentic_nonempty = [], []
                for feats, masks_gt, idxs in tqdm(loader, desc=f"OOF area={min_area}, kernel={morph_k}"):
                    feats = feats.to(device)
                    probs = model(feats).cpu().numpy()
                    masks_gt_np = masks_gt.numpy()
                    for i, idx in enumerate(idxs):
                        prob = probs[i, 0]
                        gt = masks_gt_np[i, 0]
                        rec = ref_df.iloc[idx.item()]
                        per_t = {}
                        best_t, best_f1, best_bin = None, -1, None
                        for t in thresholds:
                            pred_bin = (prob > float(t)).astype(np.uint8)
                            if morph_k > 0:
                                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
                                pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)
                                pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel)
                            num_labels, labels = cv2.connectedComponents(pred_bin)
                            pred_filtered = np.zeros_like(pred_bin)
                            for lbl in range(1, num_labels):
                                if np.sum(labels == lbl) >= min_area:
                                    pred_filtered[labels == lbl] = 1
                            f1 = mask_f1_binary(pred_filtered, gt)
                            per_t[float(t)] = f1
                            agg[float(t)].append(f1)
                            if f1 > best_f1:
                                best_f1, best_t, best_bin = f1, float(t), pred_filtered
                        nonempty = int(best_bin.sum() > 0)
                        if int(rec["is_forged"]) == 1:
                            forged_nonempty.append(nonempty)
                        else:
                            authentic_nonempty.append(nonempty)
                        rows.append({
                            "case_id": rec["case_id"], "image_id": rec["image_id"], "is_forged": int(rec["is_forged"]),
                            "min_area": min_area, "morph_kernel": morph_k, "best_threshold": best_t,
                            "best_f1": best_f1, "pred_nonempty": nonempty,
                            **{f"f1@{t}": per_t[float(t)] for t in thresholds}
                        })
                th2mean = {float(t): float(np.mean(agg[float(t)])) if len(agg[float(t)]) else 0.0 for t in thresholds}
                mean_f1 = max(th2mean.values())
                forged_nonempty_ratio = float(np.mean(forged_nonempty)) if len(forged_nonempty) else 0.0
                authentic_nonempty_ratio = float(np.mean(authentic_nonempty)) if len(authentic_nonempty) else 0.0
                results.append((min_area, morph_k, mean_f1, th2mean, rows, {
                    "forged_nonempty_ratio": forged_nonempty_ratio,
                    "authentic_nonempty_ratio": authentic_nonempty_ratio,
                }))
                if forged_nonempty_ratio >= 0.05 and mean_f1 > best_mean_f1:
                    best_mean_f1 = mean_f1
                    best_cfg = (min_area, morph_k, th2mean, rows)
    return best_cfg, results
```

---

## 6. 💾 成果物保存ブロック (Artifact Saving Block)＋強制停止

- metrics.json/run.jsonへmacro_f1, forged_nonempty_ratio, authentic_fp_ratioを記録
- 分離指標が不健全ならSystemExit(1)で強制停止

- **foldごとに `fold{n}/metrics.json`, `fold{n}/oof.csv` などを保存**
- **全fold終了後、`overall_metrics.json`, `oof_all.csv`, `validate_summary.csv` など全体集計成果物を `*-artifacts/` 直下に必ず生成・保存すること**
- validate.pyや検証Notebookで全体集計成果物を自動生成・確認する

```python
import json

def save_validation_artifacts(oof_rows, th2mean, out_dir, exp_id, best_min_area, best_morph_k, config: Config):
    # foldディレクトリを必ず作成してから保存
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(out_dir / "oof.csv", index=False)

    best_t = max(th2mean, key=th2mean.get)
    zero_f1_ratio = float((oof_df["best_f1"] == 0.0).mean()) if len(oof_df) else 0.0
    perfect_f1_ratio = float((oof_df["best_f1"] == 1.0).mean()) if len(oof_df) else 0.0

    authentic_df = oof_df[oof_df["is_forged"] == 0]
    forged_df = oof_df[oof_df["is_forged"] == 1]
    authentic_f1_mean = float(authentic_df["best_f1"].mean()) if len(authentic_df) else 0.0
    forged_f1_mean = float(forged_df["best_f1"].mean()) if len(forged_df) else 0.0
    forged_nonempty_ratio = float(forged_df["pred_nonempty"].mean()) if len(forged_df) else 0.0
    authentic_fp_ratio = float(authentic_df["pred_nonempty"].mean()) if len(authentic_df) else 0.0

    macro_f1 = float(np.mean([authentic_f1_mean, forged_f1_mean]))
    overall_f1 = float(oof_df["best_f1"].mean()) if len(oof_df) else 0.0

    metrics = {
        "n_samples": int(len(oof_df)),
        "best_threshold": float(best_t),
        "macro_f1": macro_f1,
        "overall_f1": overall_f1,
        "macro_f1_std": float(oof_df["best_f1"].std() if len(oof_df) else 0.0),
        "thresholds": {str(k): float(v) for k, v in th2mean.items()},
        "zero_f1_ratio": zero_f1_ratio,
        "perfect_f1_ratio": perfect_f1_ratio,
        "forged_f1_mean": forged_f1_mean,
        "authentic_f1_mean": authentic_f1_mean,
        "forged_nonempty_ratio": forged_nonempty_ratio,
        "authentic_fp_ratio": authentic_fp_ratio,
        "best_postprocessing": {"min_area": int(best_min_area), "morph_kernel": int(best_morph_k)}
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    run = {
        "experiment_id": exp_id,
        "date": datetime.now().date().isoformat(),
        "status": "completed",
        "cv": {"macro_f1": macro_f1},
        "artifacts": {"experiment_dir": str(out_dir)},
        "forged_nonempty_ratio": forged_nonempty_ratio,
        "authentic_fp_ratio": authentic_fp_ratio
    }
    with open(out_dir / "run.json", "w") as f:
        json.dump(run, f, indent=2)

    # 強制停止（退化解ガード）
    if forged_nonempty_ratio < config.MIN_FORGED_NONEMPTY_RATIO:
        raise SystemExit(f"❌ forged_nonempty_ratio={forged_nonempty_ratio:.3f} < {config.MIN_FORGED_NONEMPTY_RATIO} → 強制停止")
    if authentic_fp_ratio > config.MAX_AUTHENTIC_FP_RATIO:
        raise SystemExit(f"❌ authentic_fp_ratio={authentic_fp_ratio:.3f} > {config.MAX_AUTHENTIC_FP_RATIO} → 強制停止")
    return metrics
```

---

## 7. 🔄 メイン実行ブロック (Main Pipeline Block) — 現場指揮官

- **`Config`の指示を忠実に実行**
- **標準兵器（WeightedSampler/WeightedLoss）をスイッチに応じて自動装備**
- **全fold終了後に全体集計成果物を生成する処理を必ず追加すること**

```python
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold

def main():
    config = Config()
    logger = Logger(config.ARTIFACTS_DIR / 'train.log')
    logger.info(f"Experiment started: {config.EXP_ID}")
    logger.info(f"TRAINING_MODE: {config.TRAINING_MODE}")

    # Data
    df = prepare_dataframe(config)
    gkf = GroupKFold(n_splits=config.N_SPLITS)
    groups = df[config.GROUP_COL]

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        if config.USE_FIRST_FOLD_ONLY and fold > 0:
            break

        logger.info(f"\n{'=' * 80}")
        logger.info(f"FOLD {fold+1}/{config.N_SPLITS}")
        logger.info(f"{'=' * 80}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"✓ Train: {len(train_df)}, Val: {len(val_df)}")

        train_ds = get_dataset(train_df, config, is_train=True)
        val_ds = get_dataset(val_df, config, is_train=False)

        # === DataLoaderの構築（標準兵器：WeightedSampler） ===
        train_sampler = None
        shuffle = True
        if config.TrainConfig.USE_WEIGHTED_SAMPLER:
            logger.info(f"✓ Using WeightedRandomSampler with forged_weight={config.TrainConfig.FORGED_SAMPLE_WEIGHT}")
            train_labels = train_df['is_forged'].values
            sample_weights = [config.TrainConfig.FORGED_SAMPLE_WEIGHT if label == 1 else 1.0 for label in train_labels]
            train_sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights))
            shuffle = False

        train_loader = DataLoader(train_ds, batch_size=config.TrainConfig.BATCH_SIZE, sampler=train_sampler, shuffle=shuffle, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config.TrainConfig.BATCH_SIZE, shuffle=False, num_workers=0)

        # === モデルと損失関数の構築（標準兵器：WeightedLoss） ===
        model = build_model(config)
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model created with {params:,} parameters")

        if config.TrainConfig.USE_WEIGHTED_LOSS:
            logger.info(f"✓ Using WeightedBCEDiceLoss with pos_weight={config.TrainConfig.POS_WEIGHT}")
            criterion = WeightedBCEDiceLoss(pos_weight=config.TrainConfig.POS_WEIGHT)
        else:
            logger.info("✓ Using simple BCE loss (no weighting)")
            criterion = nn.BCELoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.TrainConfig.LEARNING_RATE, weight_decay=config.TrainConfig.WEIGHT_DECAY)

        # Train
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING")
        logger.info("=" * 80)
        for epoch in range(config.TrainConfig.NUM_EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
            logger.info(f"Epoch {epoch+1}/{config.TrainConfig.NUM_EPOCHS} - Loss: {loss:.4f}")
            if (epoch + 1) % 1 == 0:
                torch.save(model.state_dict(), config.MODELS_DIR / f"model_fold{fold}_epoch{epoch+1}.pth")

        # Validation (oOF1)
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION (OOF, grid search)")
        logger.info("=" * 80)
        best_cfg, grid_results = validate_grid(
            model, val_loader,
            config.CONFIDENCE_THRESHOLDS_GRID,
            config.MIN_AREA_GRID,
            config.MORPH_KERNEL_SIZE_GRID,
            config.DEVICE, val_df
        )
        min_area, morph_k, th2mean, oof_rows = best_cfg
        best_t = max(th2mean, key=th2mean.get)
        logger.info(f"✓ Best PP: min_area={min_area}, morph_kernel={morph_k}, best_threshold={best_t:.3f}")

        # Save metrics + 強制停止ガード
        # fold成果物は fold{n} サブディレクトリへ
        fold_dir = config.ARTIFACTS_DIR / f'fold{fold}'
        metrics = save_validation_artifacts(
            oof_rows, th2mean, fold_dir, config.EXP_ID,
            best_min_area=min_area, best_morph_k=morph_k, config=config
        )
        logger.info(f"✓ macro_f1={metrics['macro_f1']:.4f}, forged_nonempty_ratio={metrics['forged_nonempty_ratio']:.3f}, authentic_fp_ratio={metrics['authentic_fp_ratio']:.3f}")

        # 完了
        torch.save(model.state_dict(), config.MODELS_DIR / f"model_fold{fold}_final.pth")
        logger.info("\n" + "=" * 80)
        logger.info("FOLD COMPLETE")
        logger.info("=" * 80)

    # === 全体集計成果物生成 ===
    aggregate_oof_all_and_metrics(config)

if __name__ == "__main__":
    main()
```

---

## 8. 🚀 実行トリガーブロック (Execution Trigger Block)

```python
if __name__ == "__main__":
    main()
```

---

## 9. 📊 環境バージョン記録ブロック (Environment Version Logging Block)

- 2章Logger初期化で既に出力済み。追加で必要なら関数化して再利用

---

## 10. 🧪 提出補助（任意）: RLEエンコード・提出形式検証（OOF確立後に導入）

- 列メジャー/1-indexed/JSON配列形式

```python
def rle_encode(mask):
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return "authentic"
    pixels = mask.T.flatten()
    runs = []
    prev = 0
    pos = 0
    for i, p in enumerate(pixels):
        if p != prev:
            if prev == 1:
                runs.extend([pos + 1, i - pos])
            if p == 1:
                pos = i
            prev = p
    if prev == 1:
        runs.extend([pos + 1, len(pixels) - pos])
    return json.dumps([int(x) for x in runs])
```


## 11. 🗂️ 成果物配置・命名規則（foldごと保存＋全体集計保存の方針）

- 成果物は必ず以下に保存してください（余計な上位階層は作らない）
    - ローカル: `experiments/<exp_id>-artifacts/`
    - Kaggle: `/kaggle/working/<exp_id>-artifacts/`
    **Kaggle提出時は submission.csv を `/kaggle/working/submission.csv`（output直下）にもコピーしてください。Kaggle提出UIは output直下のみ提出対象として認識するため、サブディレクトリ内のファイルは自動検出されません。**
- foldごとの成果物（metrics.json, oof.csv, run.json等）は `fold0/`, `fold1/`, ... のサブディレクトリに保存します（単一foldでも `fold0/` を使用）。
- 全体集計成果物（submission.csv, oof_all.csv, overall_metrics.json, validate_summary.csv, train.log 等）は `*-artifacts/` 直下に必ず保存してください。
- 学習済みモデルは `*-artifacts/models/` 配下に保存し、ファイル名は `decoder_fold{n}_final.pth` 等、foldが識別できる命名にしてください。
- これにより、validate.pyや検証Notebookが全体集計成果物を自動検出・集計できます。
- **複数fold（CV）時は「foldごと成果物」と「全体集計成果物」の両方を必ず保存・管理すること。**

例（5-foldの想定）:

```
<exp_id>-artifacts/
├─ fold0/
│  ├─ metrics.json
│  ├─ oof.csv
│  └─ run.json
├─ fold1/
│  └─ ...
├─ fold2/
│  └─ ...
├─ fold3/
│  └─ ...
├─ fold4/
│  └─ ...
├─ models/
│  ├─ decoder_fold0_final.pth
│  ├─ decoder_fold1_final.pth
│  └─ ...
├─ oof_all.csv
├─ overall_metrics.json
├─ validate_summary.csv
├─ submission.csv
└─ train.log
```

---

## 12. 📝 チェックリスト

- [ ] foldごと成果物（metrics.json, oof.csv, run.json等）が揃っている
- [ ] 全体集計成果物（overall_metrics.json, oof_all.csv, validate_summary.csv等）が揃っている
- [ ] run.json, report.md, README.md も更新済み

---

## 付録A: 仕様の要点（差分サマリ）

- **最上位戦略フラグ `TRAINING_MODE`** で学習方式切り替え（pixel_based/feature_based）
- **標準兵器（WeightedSampler/WeightedLoss）をデフォルトON**
- Datasetはファクトリー（`get_dataset`）で自動生成
- モデルもファクトリー（`build_model`）で自動生成
- 検証はvalidate_grid（Val Loss非採用）。指標はmacro_f1/forged_nonempty_ratio/authentic_fp_ratio
- 退化解ガードは強制停止（成果物を残さずrun終了）

---

## 付録B: 使用例

### EXP003T（DINOv2特徴量ベース）で使う場合

```python
# Configで以下を設定
TRAINING_MODE = "feature_based"
FeatureConfig.FEATURE_DIR = Path('/kaggle/input/exp003t-dino-v2-features')
FeatureConfig.FEATURE_DIM = 768
FeatureConfig.DECODER_TYPE = "SimpleDecoder"
```

### EXP002T（ピクセルベース・U-Net）で使う場合

```python
# Configで以下を設定
TRAINING_MODE = "pixel_based"
PixelConfig.MODEL_TYPE = "FastUNet"
PixelConfig.IMAGE_SIZE = 256
PixelConfig.AUGMENTATIONS = "medium"
```

### 🧪 メトリクススキーマ（metrics.json / run.json）

指標の命名を以下に統一し、`macro_f1` の意味の混乱（forged-only平均との混同）を避ける。

| フィールド | 定義 | 備考 |
|-----------|------|------|
| metric_version | スキーマバージョン。`v1` 現行, `v0` 旧(forced-only) | validateで分岐 |
| mean_f1_forged | forged画像に対する平均F1 | 旧 `macro_f1` (v0) 相当 |
| f1_authentic | authentic画像のF1 (空を正しく空と出せば1, 非空なら0) | `1 - authentic_fp_ratio` と同義 |
| macro_f1 | (mean_f1_forged + f1_authentic) / 2 | ダッシュボードの主軸 |
| forged_nonempty_ratio | forged画像で非空予測した割合 | 検出率 |
| authentic_fp_ratio | authentic画像で非空予測した割合 | 誤検出率 (低いほど良い) |
| best_threshold | マスク二値化に用いた最良閾値 | グリッドサーチ時 |
| best_postprocessing | 最良後処理パラメータ | {min_area, morph_kernel, ...} |
| created_at | 生成日時 | ISO8601 |

`overall_metrics.json` では上述フィールドの *_mean / *_std を付加し、fold配列に各foldのmetrics.json内容をそのまま格納する。

### 🔄 レガシー互換 (v0 → v1 移行)

旧フォーマット(v0)では `macro_f1` が forged-only 平均を指している。validate側で以下ロジックを適用し自動補完する:
1. `metric_version` がない or `v0` → `mean_f1_forged = macro_f1`
2. `f1_authentic` が無ければ `authentic_fp_ratio` があれば `f1_authentic = 1 - authentic_fp_ratio`、無ければ `null`。
3. `macro_f1` を再計算できれば再計算、できなければ forged-only を暫定値とし `macro_f1_legacy = macro_f1` を内部利用。

### ✅ validate_summary.csv 追加列

`validate_summary.csv` には少なくとも以下列を含める:
```
fold, macro_f1, mean_f1_forged, f1_authentic,
forged_nonempty_ratio, authentic_fp_ratio
```

これにより横断比較・健全性判定・推移分析が容易になる。