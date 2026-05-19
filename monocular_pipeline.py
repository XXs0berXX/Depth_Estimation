#!/usr/bin/env python3
"""
Monocular Depth Pipeline — Relative + Metric
═══════════════════════════════════════════════════════════════════════════════
Pipeline:
  Input Image
      ↓
  Relative Depth Model  →  Relative Depth Map  (0-1, unitless)
      ↓
  Metric Depth Model    →  Scale Alignment
      ↓
  Metric Depth Map      (real-world meters)
      ↓
  Output / Visualisation

Relative models (choose one via --relative):
  depth-anything-v2   →  depth-anything/Depth-Anything-V2-Small-hf  [default, fastest]
  depth-anything-v2b  →  depth-anything/Depth-Anything-V2-Base-hf
  depth-anything-v2l  →  depth-anything/Depth-Anything-V2-Large-hf
  marigold            →  prs-eth/marigold-depth-lcm-v1-0            [diffusion, slower]

Metric models (choose one via --metric):
  depth-pro           →  apple/DepthPro-hf                          [best, ~1 GB]
  unidepth            →  lpiccinelli/unidepth-v2-vitl14             [good, ~600 MB]
  dav2-metric         →  depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf

Usage:
  python monocular_pipeline.py --input dataset/ --output output/
  python monocular_pipeline.py --relative marigold --metric depth-pro
  python monocular_pipeline.py --relative depth-anything-v2l --metric unidepth
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Model registries
# ─────────────────────────────────────────────────────────────────────────────

RELATIVE_MODELS = {
    "depth-anything-v2":  "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything-v2b": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything-v2l": "depth-anything/Depth-Anything-V2-Large-hf",
    "marigold":           "prs-eth/marigold-depth-lcm-v1-0",
}

METRIC_MODELS = {
    "depth-pro":  "apple/DepthPro-hf",
    "unidepth":   "lpiccinelli/unidepth-v2-vitl14",
    "dav2-metric": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    def __init__(self, args):
        self.input_dir     = args.input
        self.output_dir    = args.output
        self.relative_key  = args.relative
        self.metric_key    = args.metric
        self.colormap      = args.colormap
        self.image_exts    = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

        # Device: explicit > cuda > mps > cpu
        if args.device:
            self.device = args.device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.relative_model_id = RELATIVE_MODELS[self.relative_key]
        self.metric_model_id   = METRIC_MODELS[self.metric_key]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Relative Depth
# ─────────────────────────────────────────────────────────────────────────────

class RelativeDepthEstimator:
    """
    Wraps Depth Anything V2 (default) or Marigold.
    Returns a float32 array, same spatial size as input, values in [0, 1]
    where 1 = closest pixel, 0 = furthest.
    """

    def __init__(self, cfg: Config):
        self.cfg   = cfg
        self.key   = cfg.relative_key
        self.model_id = cfg.relative_model_id
        print(f"  ▶ Relative  [{self.key}]  loading '{self.model_id}' …")

        if self.key == "marigold":
            self._load_marigold()
        else:
            self._load_dav2()

    def _load_dav2(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model     = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.cfg.device).eval()

    def _load_marigold(self):
        from diffusers import MarigoldDepthPipeline
        self._pipe = MarigoldDepthPipeline.from_pretrained(
            self.model_id, variant="fp16", torch_dtype=torch.float16
        ).to(self.cfg.device)

    @torch.no_grad()
    def estimate(self, image_pil: Image.Image) -> np.ndarray:
        """Returns relative depth map normalised to [0,1], 1=closest."""
        W, H = image_pil.size

        if self.key == "marigold":
            result = self._pipe(image_pil, num_inference_steps=4)
            depth  = result.prediction[0].squeeze()          # (H', W')
            depth  = cv2.resize(np.array(depth), (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            inputs  = self._processor(images=image_pil, return_tensors="pt")
            inputs  = {k: v.to(self.cfg.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            depth   = outputs.predicted_depth.squeeze().cpu().numpy()
            depth   = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        depth = depth.astype(np.float32)

        # Normalise to [0, 1].
        # Depth Anything V2 & Marigold: higher raw = closer (disparity convention).
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth)
        return (depth - d_min) / (d_max - d_min)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Metric Depth
# ─────────────────────────────────────────────────────────────────────────────

class MetricDepthEstimator:
    """
    Wraps Depth Pro, UniDepthV2, or Depth Anything V2 Metric.
    Returns a float32 array in METRES, same spatial size as input.
    """

    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self.key      = cfg.metric_key
        self.model_id = cfg.metric_model_id
        print(f"  ▶ Metric    [{self.key}]  loading '{self.model_id}' …")

        if self.key == "depth-pro":
            self._load_depth_pro()
        elif self.key == "unidepth":
            self._load_unidepth()
        else:
            self._load_dav2_metric()

    # ── Depth Pro ─────────────────────────────────────────────────────────────
    def _load_depth_pro(self):
        from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
        self._processor = DepthProImageProcessorFast.from_pretrained(self.model_id)
        self._model     = DepthProForDepthEstimation.from_pretrained(
            self.model_id, torch_dtype=torch.float16
        ).to(self.cfg.device).eval()

    @torch.no_grad()
    def _infer_depth_pro(self, image_pil: Image.Image) -> np.ndarray:
        W, H   = image_pil.size
        inputs = self._processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        with torch.autocast(self.cfg.device if self.cfg.device != "mps" else "cpu"):
            outputs = self._model(**inputs)
        # post_process gives metric depth in metres
        post = self._processor.post_process_depth_estimation(
            outputs, target_sizes=[(H, W)]
        )
        depth = post[0]["predicted_depth"].cpu().numpy().astype(np.float32)
        return depth

    # ── UniDepthV2 ────────────────────────────────────────────────────────────
    def _load_unidepth(self):
        try:
            from unidepth.models import UniDepthV2
            self._model = UniDepthV2.from_pretrained(self.model_id)
            self._model.to(self.cfg.device).eval()
        except ImportError:
            raise ImportError(
                "UniDepth not installed. Run:\n"
                "  pip install git+https://github.com/lpiccinelli-eth/UniDepth.git"
            )

    @torch.no_grad()
    def _infer_unidepth(self, image_pil: Image.Image) -> np.ndarray:
        W, H  = image_pil.size
        rgb   = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float()
        rgb   = rgb.to(self.cfg.device)
        preds = self._model.infer(rgb)
        depth = preds["depth"].squeeze().cpu().numpy().astype(np.float32)
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        return depth

    # ── Depth Anything V2 Metric ──────────────────────────────────────────────
    def _load_dav2_metric(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model     = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.cfg.device).eval()

    @torch.no_grad()
    def _infer_dav2_metric(self, image_pil: Image.Image) -> np.ndarray:
        W, H    = image_pil.size
        inputs  = self._processor(images=image_pil, return_tensors="pt")
        inputs  = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        depth   = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        return depth

    def estimate(self, image_pil: Image.Image) -> np.ndarray:
        """Returns depth in METRES (float32, same size as image)."""
        if self.key == "depth-pro":
            return self._infer_depth_pro(image_pil)
        elif self.key == "unidepth":
            return self._infer_unidepth(image_pil)
        else:
            return self._infer_dav2_metric(image_pil)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Scale Alignment
# ─────────────────────────────────────────────────────────────────────────────

def scale_align(relative: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """
    Align the relative depth map to metric scale using least-squares fitting.

    Finds (scale, shift) such that:
        scale * relative + shift ≈ metric

    Returns the aligned relative map in metres. This is useful for comparing
    the two maps and for cases where the relative model has higher spatial
    resolution than the metric model.

    Reference: Ranftl et al., "Towards Robust Monocular Depth Estimation" (2020)
               scale-and-shift invariant alignment.
    """
    r = relative.flatten().astype(np.float64)
    m = metric.flatten().astype(np.float64)

    # Least-squares: [r, 1] @ [scale, shift]^T = m
    A = np.stack([r, np.ones_like(r)], axis=1)
    result, _, _, _ = np.linalg.lstsq(A, m, rcond=None)
    scale, shift = result

    aligned = scale * relative + shift
    # Clamp: no negative distances
    aligned = np.clip(aligned, 0.01, None).astype(np.float32)
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def colorize(depth: np.ndarray, cmap_name: str = "inferno",
             vmin: float = None, vmax: float = None) -> np.ndarray:
    """Colorize a depth map. Returns uint8 BGR."""
    vmin = vmin if vmin is not None else depth.min()
    vmax = vmax if vmax is not None else depth.max()
    norm    = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cmap    = plt.get_cmap(cmap_name)
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    out  = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, text, (12, 30), font, 0.75, (0, 0, 0),    3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 30), font, 0.75, (255,255,255), 1, cv2.LINE_AA)
    return out


def make_composite(
    original:  np.ndarray,
    rel_col:   np.ndarray,
    metric_col: np.ndarray,
    aligned_col: np.ndarray,
    rel_label:  str,
    met_label:  str,
    panel_w:   int = 480,
) -> np.ndarray:
    """4-panel composite: Original | Relative | Metric | Aligned."""
    h, w   = original.shape[:2]
    ph     = int(h * panel_w / w)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    panels = []

    data = [
        (original,    "Original"),
        (rel_col,     f"Relative\n{rel_label}"),
        (metric_col,  f"Metric\n{met_label}"),
        (aligned_col, "Aligned\n(metres)"),
    ]
    for img, lbl in data:
        p   = cv2.resize(img, (panel_w, ph))
        # multi-line label
        for i, line in enumerate(lbl.split("\n")):
            y = 26 + i * 22
            cv2.putText(p, line, (10, y), font, 0.55, (0, 0, 0),    2, cv2.LINE_AA)
            cv2.putText(p, line, (10, y), font, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.rectangle(p, (0, 0), (panel_w - 1, ph - 1), (180, 180, 180), 1)
        panels.append(p)

    return np.hstack(panels)


def save_colourbar(path: Path, cmap: str, label: str):
    fig, ax = plt.subplots(figsize=(5, 0.45))
    fig.subplots_adjust(bottom=0.5)
    matplotlib.colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(cmap), orientation="horizontal"
    ).set_label(label)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MonocularPipeline:

    def __init__(self, cfg: Config):
        self.cfg = cfg

        print("\n" + "═" * 68)
        print("  Monocular Depth Pipeline")
        print(f"  Device   : {cfg.device}")
        print(f"  Relative : {cfg.relative_key}  ({cfg.relative_model_id})")
        print(f"  Metric   : {cfg.metric_key}  ({cfg.metric_model_id})")
        print("═" * 68)

        print("\n[Loading models]")
        self.relative = RelativeDepthEstimator(cfg)
        self.metric   = MetricDepthEstimator(cfg)

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        save_colourbar(
            out / "legend_relative.png", cfg.colormap,
            "Relative depth  (0=far, 1=close)"
        )
        save_colourbar(
            out / "legend_metric.png", "plasma",
            "Metric depth (metres)"
        )
        print("\n[Models ready]\n")

    def process(self, image_path: Path) -> Optional[Dict]:
        print(f"{'─' * 68}")
        print(f"  {image_path.name}")

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"  [!] Cannot read — skipping.")
            return None
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        W, H = image_pil.size

        # ── Stage 1: Relative depth ──────────────────────────────────────────
        print("  [1/3] Relative depth …", end=" ", flush=True)
        rel = self.relative.estimate(image_pil)   # [0,1]
        print(f"done  shape={rel.shape}  range=[{rel.min():.3f}, {rel.max():.3f}]")

        # ── Stage 2: Metric depth ────────────────────────────────────────────
        print("  [2/3] Metric depth …", end=" ", flush=True)
        met = self.metric.estimate(image_pil)     # metres
        print(f"done  range=[{met.min():.2f}, {met.max():.2f}] m")

        # ── Stage 3: Scale alignment ─────────────────────────────────────────
        print("  [3/3] Scale alignment …", end=" ", flush=True)
        aligned = scale_align(rel, met)           # metres, aligned
        print(f"done  range=[{aligned.min():.2f}, {aligned.max():.2f}] m")

        # ── Visualise ────────────────────────────────────────────────────────
        rel_col     = colorize(rel,     self.cfg.colormap)
        met_col     = colorize(met,     "plasma")
        aligned_col = colorize(aligned, "plasma",
                                vmin=met.min(), vmax=met.max())  # same scale as metric

        composite = make_composite(
            image_bgr, rel_col, met_col, aligned_col,
            self.cfg.relative_key, self.cfg.metric_key
        )

        stem = image_path.stem
        out  = Path(self.cfg.output_dir)
        cv2.imwrite(str(out / f"{stem}_composite.jpg"), composite,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / f"{stem}_relative.jpg"),  rel_col)
        cv2.imwrite(str(out / f"{stem}_metric.jpg"),    met_col)
        cv2.imwrite(str(out / f"{stem}_aligned.jpg"),   aligned_col)

        # Save metric depth as float32 .npy for downstream use
        np.save(str(out / f"{stem}_metric.npy"), met)
        np.save(str(out / f"{stem}_aligned.npy"), aligned)

        result = {
            "file":              image_path.name,
            "image_size":        [W, H],
            "relative_model":    self.cfg.relative_key,
            "metric_model":      self.cfg.metric_key,
            "relative_range":    [round(float(rel.min()), 4),
                                   round(float(rel.max()), 4)],
            "metric_range_m":    [round(float(met.min()), 3),
                                   round(float(met.max()), 3)],
            "aligned_range_m":   [round(float(aligned.min()), 3),
                                   round(float(aligned.max()), 3)],
            "metric_mean_m":     round(float(met.mean()), 3),
            "metric_median_m":   round(float(np.median(met)), 3),
            "metric_std_m":      round(float(met.std()), 3),
        }

        print(f"  Metric depth: mean={result['metric_mean_m']} m  "
              f"median={result['metric_median_m']} m  "
              f"std={result['metric_std_m']} m")
        return result

    def run(self) -> List[Dict]:
        inp = Path(self.cfg.input_dir)
        if not inp.exists():
            sys.exit(f"[!] Input folder not found: '{inp}'")

        images = sorted([
            f for f in inp.iterdir()
            if f.suffix.lower() in self.cfg.image_exts
        ])
        if not images:
            sys.exit(f"[!] No images found in '{inp}'")

        print(f"Found {len(images)} image(s)  →  output: '{self.cfg.output_dir}'\n")

        results = []
        for img_path in tqdm(images, desc="Processing", unit="img"):
            r = self.process(img_path)
            if r:
                results.append(r)

        json_path = Path(self.cfg.output_dir) / "results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'═' * 68}")
        print(f"  Done — {len(results)}/{len(images)} images processed")
        print(f"  Outputs → '{self.cfg.output_dir}/'")
        print(f"  JSON    → '{json_path}'")
        print("═" * 68)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Relative + Metric monocular depth pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--input",    default="dataset",
                   help="Input image folder  (default: dataset)")
    p.add_argument("--output",   default="output",
                   help="Output folder       (default: output)")
    p.add_argument("--relative", default="depth-anything-v2",
                   choices=list(RELATIVE_MODELS.keys()),
                   help=(
                       "Relative depth model:\n"
                       "  depth-anything-v2   Small, fastest      [default]\n"
                       "  depth-anything-v2b  Base\n"
                       "  depth-anything-v2l  Large, best relative\n"
                       "  marigold            Diffusion, high quality\n"
                   ))
    p.add_argument("--metric",   default="depth-pro",
                   choices=list(METRIC_MODELS.keys()),
                   help=(
                       "Metric depth model:\n"
                       "  depth-pro    Apple DepthPro, best      [default]\n"
                       "  unidepth     UniDepthV2\n"
                       "  dav2-metric  Depth Anything V2 Metric\n"
                   ))
    p.add_argument("--colormap", default="inferno",
                   help="Colormap for relative map: inferno|plasma|magma|viridis")
    p.add_argument("--device",   default=None,
                   help="cuda | mps | cpu  (auto-detected if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    cfg      = Config(args)
    pipeline = MonocularPipeline(cfg)
    pipeline.run()
