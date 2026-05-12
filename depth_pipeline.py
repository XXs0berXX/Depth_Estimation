#!/usr/bin/env python3
"""
Monocular Depth Estimation Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • YOLOv8n           — object detection + bounding boxes
  • MiDaS (DPT)       — dense monocular depth map
  • DINOv2            — per-object + global feature extraction

MiDaS model options (--depth):
  Intel/dpt-hybrid-midas   ← default, fast, good quality (MiDaS v3.0 hybrid)
  Intel/dpt-large          ← better quality, slower

Usage:
  python depth_pipeline.py --input dataset/ --output output/
  python depth_pipeline.py --input dataset/ --depth Intel/dpt-large
"""

import os
import sys
import warnings
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    input_dir:    str   = "dataset"
    output_dir:   str   = "output"
    yolo_model:   str   = "yolov8n.pt"
    depth_model:  str   = "Intel/dpt-hybrid-midas"   # MiDaS DPT-Hybrid (v3.0)
    dino_model:   str   = "facebook/dinov2-base"
    device:       str   = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_conf:    float = 0.25
    yolo_iou:     float = 0.45
    colormap:     str   = "inferno"
    save_parts:   bool  = True
    image_exts:   tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_name:     str
    confidence:     float
    bbox:           Tuple[int, int, int, int]   # x1 y1 x2 y2
    depth_median:   float = 0.0
    depth_mean:     float = 0.0
    depth_std:      float = 0.0
    depth_min:      float = 0.0
    depth_max:      float = 0.0
    dino_feat_norm: float = 0.0
    dino_features:  Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return (
            f"  [{self.class_name:<15s}]  conf={self.confidence:.2f}  "
            f"bbox=({x1},{y1},{x2},{y2})  "
            f"depth_median={self.depth_median:.4f}  "
            f"depth_std={self.depth_std:.4f}  "
            f"dino‖feat‖={self.dino_feat_norm:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
]

def _color(idx: int) -> Tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 — Object Detector (YOLOv8n)
# ─────────────────────────────────────────────────────────────────────────────

class ObjectDetector:
    def __init__(self, cfg: Config):
        print(f"  ▶ YOLOv8n   loading '{cfg.yolo_model}' …")
        from ultralytics import YOLO
        self.model  = YOLO(cfg.yolo_model)
        self.conf   = cfg.yolo_conf
        self.iou    = cfg.yolo_iou
        self.device = cfg.device

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self.model(
            image_bgr, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False
        )[0]

        detections: List[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = results.names[cls_id]
            detections.append(Detection(
                class_name=class_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
            ))
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 — Depth Estimator (MiDaS via HuggingFace DPT)
# ─────────────────────────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Dense monocular depth via MiDaS (DPT) from HuggingFace.

    Models:
      Intel/dpt-hybrid-midas  — MiDaS v3.0 DPT-Hybrid  (default, ~400 MB)
      Intel/dpt-large         — MiDaS v3.0 DPT-Large    (better, ~800 MB)

    Output convention: MiDaS outputs disparity (inverse depth).
    Higher raw value = CLOSER to the camera.
    We normalise to 0-1 where 1 = closest, 0 = furthest — same convention
    as was used with Depth Anything V2, so nothing else in the pipeline changes.
    """

    def __init__(self, cfg: Config):
        print(f"  ▶ MiDaS     loading '{cfg.depth_model}' …")
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        self.processor = DPTImageProcessor.from_pretrained(cfg.depth_model)
        self.model     = DPTForDepthEstimation.from_pretrained(
            cfg.depth_model, low_cpu_mem_usage=True
        )
        self.model.to(cfg.device).eval()
        self.device = cfg.device

    @torch.no_grad()
    def estimate(self, image_pil: Image.Image) -> np.ndarray:
        """
        Returns raw disparity map (float32) resized to original image resolution.
        Higher value = closer. Safe to normalise 0-1 downstream.
        """
        inputs  = self.processor(images=image_pil, return_tensors="pt")
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # predicted_depth shape: (1, H', W')
        depth   = outputs.predicted_depth.squeeze().cpu().numpy()

        # Resize back to original image size
        depth_resized = cv2.resize(
            depth, (image_pil.width, image_pil.height),
            interpolation=cv2.INTER_LINEAR
        )
        return depth_resized.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 — Feature Extractor (DINOv2)
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2Extractor:
    """
    Extracts visual features using DINOv2.
      • Global CLS token  → (768,) scene-level descriptor
      • Patch tokens      → (N_patches, 768) spatial features
      • Crop CLS token    → (768,) per-detected-object descriptor
    """

    MIN_CROP = 14   # DINOv2 patch size

    def __init__(self, cfg: Config):
        print(f"  ▶ DINOv2    loading '{cfg.dino_model}' …")
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(cfg.dino_model)
        self.model     = AutoModel.from_pretrained(cfg.dino_model)
        self.model.to(cfg.device).eval()
        self.device = cfg.device

    @torch.no_grad()
    def _forward(self, image_pil: Image.Image):
        inputs  = self.processor(images=image_pil, return_tensors="pt")
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs)

    def global_cls(self, image_pil: Image.Image) -> np.ndarray:
        out = self._forward(image_pil)
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    def patch_features(self, image_pil: Image.Image) -> np.ndarray:
        out = self._forward(image_pil)
        return out.last_hidden_state[:, 1:, :].squeeze().cpu().numpy()

    def crop_cls(self, image_pil: Image.Image, bbox: Tuple[int,int,int,int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < self.MIN_CROP or (y2 - y1) < self.MIN_CROP:
            return np.zeros(768, dtype=np.float32)
        crop = image_pil.crop((x1, y1, x2, y2))
        out  = self._forward(crop)
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _colorize_depth(depth: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
    d_min, d_max = depth.min(), depth.max()
    norm    = (depth - d_min) / (d_max - d_min + 1e-8)
    cmap    = plt.get_cmap(cmap_name)
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def _draw_boxes(
    image_bgr: np.ndarray,
    detections: List[Detection],
    depth_norm: np.ndarray,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color   = _color(i)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        bar_h = y2 - y1
        fill  = int(bar_h * (1.0 - det.depth_median))
        cv2.rectangle(out, (x1 - 6, y1), (x1 - 2, y2), (50, 50, 50), -1)
        cv2.rectangle(out, (x1 - 6, y2 - fill), (x1 - 2, y2), color, -1)

        label = f"{det.class_name}  {det.confidence:.2f}  d={det.depth_median:.3f}"
        (tw, th), bl = cv2.getTextSize(label, font, 0.45, 1)
        lx = max(x1, 0)
        ly = max(y1 - th - bl - 4, 0)
        cv2.rectangle(out, (lx, ly), (lx + tw + 4, ly + th + bl + 4), color, -1)
        cv2.putText(out, label, (lx + 2, ly + th + 2),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def _make_composite(
    orig:    np.ndarray,
    depth_c: np.ndarray,
    annot:   np.ndarray,
    panel_w: int = 640,
) -> np.ndarray:
    h, w   = orig.shape[:2]
    scale  = panel_w / w
    new_h  = int(h * scale)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    panels = []
    labels = ["Original", "MiDaS Depth", "YOLOv8n + Depth"]
    for img, lbl in zip([orig, depth_c, annot], labels):
        p = cv2.resize(img, (panel_w, new_h))
        cv2.putText(p, lbl, (12, 30), font, 0.75, (0, 0, 0),    3, cv2.LINE_AA)
        cv2.putText(p, lbl, (12, 30), font, 0.75, (255,255,255), 1, cv2.LINE_AA)
        cv2.rectangle(p, (0, 0), (panel_w - 1, new_h - 1), (220, 220, 220), 1)
        panels.append(p)

    return np.hstack(panels)


def _save_depth_legend(output_dir: Path, cmap_name: str):
    fig, ax = plt.subplots(figsize=(5, 0.45))
    fig.subplots_adjust(bottom=0.5)
    matplotlib.colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(cmap_name),
        orientation="horizontal",
    ).set_label("Relative depth  (0 = far, 1 = close)")
    plt.savefig(output_dir / "depth_legend.png", dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DepthPipeline:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        print("\n" + "═" * 60)
        print("  Depth Estimation Pipeline  (MiDaS edition)")
        print(f"  Device  : {cfg.device}")
        print(f"  YOLO    : {cfg.yolo_model}")
        print(f"  Depth   : {cfg.depth_model}")
        print(f"  DINOv2  : {cfg.dino_model}")
        print("═" * 60)

        print("\n[Loading models]")
        self.detector  = ObjectDetector(cfg)
        self.depth_est = DepthEstimator(cfg)
        self.dino      = DINOv2Extractor(cfg)

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _save_depth_legend(out, cfg.colormap)
        print("\n[Models ready]\n")

    def process_image(self, image_path: Path) -> Optional[Dict]:
        print(f"{'─'*60}")
        print(f"  {image_path.name}")

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"  [!] Cannot load image — skipping.")
            return None
        h, w = image_bgr.shape[:2]
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        # ── 1. MiDaS ─────────────────────────────────────────────────────────
        print("  [1/3] MiDaS depth …", end=" ", flush=True)
        depth_raw  = self.depth_est.estimate(image_pil)
        # Normalise 0-1 (1 = closest, 0 = furthest — MiDaS is disparity so higher = closer)
        depth_norm = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-8)
        depth_col  = _colorize_depth(depth_raw, self.cfg.colormap)
        print(f"done  (min={depth_raw.min():.2f}, max={depth_raw.max():.2f})")

        # ── 2. YOLOv8n ───────────────────────────────────────────────────────
        print("  [2/3] YOLOv8n detection …", end=" ", flush=True)
        detections = self.detector.detect(image_bgr)
        print(f"done  ({len(detections)} object(s) detected)")

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            roi = depth_norm[y1c:y2c, x1c:x2c]
            if roi.size > 0:
                det.depth_median = float(np.median(roi))
                det.depth_mean   = float(np.mean(roi))
                det.depth_std    = float(np.std(roi))
                det.depth_min    = float(roi.min())
                det.depth_max    = float(roi.max())

        # ── 3. DINOv2 ────────────────────────────────────────────────────────
        print("  [3/3] DINOv2 features …", end=" ", flush=True)
        global_feat = self.dino.global_cls(image_pil)
        patch_feats = self.dino.patch_features(image_pil)

        for det in detections:
            det.dino_features  = self.dino.crop_cls(image_pil, det.bbox)
            det.dino_feat_norm = float(np.linalg.norm(det.dino_features))
        print("done")

        # ── Visualise ────────────────────────────────────────────────────────
        annotated = _draw_boxes(image_bgr, detections, depth_norm)
        composite = _make_composite(image_bgr, depth_col, annotated)

        stem = image_path.stem
        out  = Path(self.cfg.output_dir)
        cv2.imwrite(str(out / f"{stem}_composite.jpg"), composite,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if self.cfg.save_parts:
            cv2.imwrite(str(out / f"{stem}_depth.jpg"),    depth_col)
            cv2.imwrite(str(out / f"{stem}_detected.jpg"), annotated)

        if detections:
            print("  Detections:")
            for det in detections:
                print(det.summary())
        else:
            print("  No objects detected above confidence threshold.")

        return {
            "file":             image_path.name,
            "image_size":       (w, h),
            "n_detections":     len(detections),
            "detections":       detections,
            "depth_raw":        depth_raw,
            "depth_normalized": depth_norm,
            "global_dino_feat": global_feat,
            "patch_dino_feats": patch_feats,
        }

    def run(self) -> List[Dict]:
        input_path = Path(self.cfg.input_dir)
        if not input_path.exists():
            sys.exit(f"[!] Input folder not found: '{input_path}'")

        images = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in self.cfg.image_exts
        ])
        if not images:
            sys.exit(f"[!] No images found in '{input_path}'")

        print(f"Found {len(images)} image(s)  →  output: '{self.cfg.output_dir}'\n")

        results = []
        for img_path in images:
            r = self.process_image(img_path)
            if r:
                results.append(r)

        # ── JSON summary ──────────────────────────────────────────────────────
        summary = []
        for r in results:
            dets = []
            for d in r["detections"]:
                dets.append({
                    "class":          d.class_name,
                    "confidence":     round(d.confidence, 4),
                    "bbox":           list(d.bbox),
                    "depth_median":   round(d.depth_median, 6),
                    "depth_mean":     round(d.depth_mean, 6),
                    "depth_std":      round(d.depth_std, 6),
                    "depth_min":      round(d.depth_min, 6),
                    "depth_max":      round(d.depth_max, 6),
                    "dino_feat_norm": round(d.dino_feat_norm, 4),
                })
            summary.append({
                "file":         r["file"],
                "image_size":   r["image_size"],
                "n_detections": r["n_detections"],
                "detections":   dets,
            })

        json_path = Path(self.cfg.output_dir) / "results.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'═'*60}")
        print(f"  Done — {len(results)}/{len(images)} images processed")
        print(f"  Outputs → '{self.cfg.output_dir}/'")
        print(f"  JSON    → '{json_path}'")
        print("═" * 60)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="YOLOv8n + MiDaS + DINOv2 depth pipeline"
    )
    p.add_argument("--input",    default="dataset",
                   help="Folder containing input images  (default: dataset)")
    p.add_argument("--output",   default="output",
                   help="Folder for output results        (default: output)")
    p.add_argument("--yolo",     default="yolov8n.pt")
    p.add_argument("--depth",    default="Intel/dpt-hybrid-midas",
                   help="MiDaS model:\n"
                        "  Intel/dpt-hybrid-midas  (default, ~400 MB)\n"
                        "  Intel/dpt-large         (better,  ~800 MB)")
    p.add_argument("--dino",     default="facebook/dinov2-base",
                   help="DINOv2 model: dinov2-small / base / large")
    p.add_argument("--conf",     type=float, default=0.25)
    p.add_argument("--iou",      type=float, default=0.45)
    p.add_argument("--colormap", default="inferno",
                   help="inferno | plasma | magma | viridis")
    p.add_argument("--device",   default=None,
                   help="cuda | mps | cpu  (auto-detected if omitted)")
    p.add_argument("--no-parts", action="store_true",
                   help="Skip saving individual depth/annotated images")
    a = p.parse_args()

    # Device priority: explicit flag > CUDA > MPS (Apple Silicon) > CPU
    if a.device:
        device = a.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return Config(
        input_dir   = a.input,
        output_dir  = a.output,
        yolo_model  = a.yolo,
        depth_model = a.depth,
        dino_model  = a.dino,
        yolo_conf   = a.conf,
        yolo_iou    = a.iou,
        colormap    = a.colormap,
        save_parts  = not a.no_parts,
        device      = device,
    )


if __name__ == "__main__":
    cfg      = parse_args()
    pipeline = DepthPipeline(cfg)
    pipeline.run()
