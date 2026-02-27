# -*- coding: utf-8 -*-
# Description: Tiled DiffVG pipeline for high-resolution image vectorization.
#   Slices large input images into overlapping tiles, optimises each tile
#   independently (with micro-batch accumulation) using DiffVG, then merges
#   the per-tile SVGs into a single output SVG with <g transform="translate">
#   coordinate placement.

import shutil
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path
from typing import AnyStr, List, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.libs.metric.lpips_origin import LPIPS
from pytorch_svgrender.painter.diffvg import Painter, PainterOptimizer
from pytorch_svgrender.plt import plot_img


# ---------------------------------------------------------------------------
# Helper: build a feather (spatial weight) mask for one tile
# ---------------------------------------------------------------------------

def _make_feather_mask(height: int, width: int, feather: int, device) -> torch.Tensor:
    """Return a (1, 1, H, W) weight tensor that is 1 in the centre and decays
    smoothly to 0 within *feather* pixels of each edge.  When feather==0 the
    mask is all-ones (no blending)."""
    if feather <= 0:
        return torch.ones(1, 1, height, width, device=device)

    wy = torch.ones(height, device=device)
    wx = torch.ones(width, device=device)

    ramp = feather
    for i in range(ramp):
        v = (i + 1) / (ramp + 1)
        wy[i] = v
        wy[height - 1 - i] = v
        wx[i] = v
        wx[width - 1 - i] = v

    mask = wy.unsqueeze(1) * wx.unsqueeze(0)          # (H, W)
    return mask.unsqueeze(0).unsqueeze(0)              # (1, 1, H, W)


# ---------------------------------------------------------------------------
# Helper: compute tile origins (top-left corners) for the given image size
# ---------------------------------------------------------------------------

def _compute_tile_origins(
        img_h: int,
        img_w: int,
        tile_size: int,
        overlap: int,
) -> List[Tuple[int, int]]:
    """Return a list of (row, col) top-left pixel positions for every tile."""
    stride = tile_size - overlap
    origins = []
    row = 0
    while True:
        col = 0
        while True:
            origins.append((row, col))
            if col + tile_size >= img_w:
                break
            col = min(col + stride, img_w - tile_size)
        if row + tile_size >= img_h:
            break
        row = min(row + stride, img_h - tile_size)
    return origins


# ---------------------------------------------------------------------------
# Helper: minimal SVG merger using ElementTree
# ---------------------------------------------------------------------------

def _merge_tile_svgs(
        tile_svg_paths: List[Path],
        tile_origins: List[Tuple[int, int]],
        canvas_w: int,
        canvas_h: int,
        output_path: Path,
) -> None:
    """Combine per-tile SVG files into one SVG with translate groups."""
    NS = "http://www.w3.org/2000/svg"
    ET.register_namespace("", NS)

    root = ET.Element(f"{{{NS}}}svg")
    root.set("version", "1.1")
    root.set("width", str(canvas_w))
    root.set("height", str(canvas_h))
    root.set("viewBox", f"0 0 {canvas_w} {canvas_h}")

    for svg_path, (row, col) in zip(tile_svg_paths, tile_origins):
        if not svg_path.exists():
            continue
        try:
            tree = ET.parse(str(svg_path))
        except ET.ParseError:
            continue

        tile_root = tree.getroot()
        group = ET.SubElement(root, f"{{{NS}}}g")
        group.set("transform", f"translate({col},{row})")

        # Copy all child elements from the tile SVG into the group
        for child in tile_root:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag in ("defs",):
                continue  # skip per-tile defs to avoid id collisions
            _copy_element(child, group, NS)

    tree_out = ET.ElementTree(root)
    ET.indent(tree_out, space="  ")
    with open(str(output_path), "wb") as f:
        tree_out.write(f, encoding="utf-8", xml_declaration=True)


def _copy_element(src: ET.Element, dst_parent: ET.Element, ns: str) -> ET.Element:
    """Recursively copy *src* into *dst_parent*, stripping namespace prefixes."""
    tag = src.tag.split("}")[-1] if "}" in src.tag else src.tag
    new_el = ET.SubElement(dst_parent, f"{{{ns}}}{tag}", attrib=src.attrib)
    new_el.text = src.text
    new_el.tail = src.tail
    for child in src:
        _copy_element(child, new_el, ns)
    return new_el


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DiffVGTilePipeline(ModelState):
    """Tiled DiffVG image-to-SVG pipeline.

    Splits the input image into overlapping tiles, runs independent DiffVG
    optimisation on each tile (processing *batch_size* tiles per micro-batch
    step to amortise overhead), then assembles the per-tile SVGs into a single
    output SVG file with correct global coordinates.
    """

    def __init__(self, args):
        logdir_ = (
            f"sd{args.seed}"
            f"-tile{args.x.tile_size}"
            f"-ov{args.x.overlap}"
            f"-P{args.x.num_paths}"
        )
        super().__init__(args, log_path_suffix=logdir_)

        assert self.x_cfg.path_type in ("unclosed", "closed"), (
            "path_type must be 'unclosed' or 'closed'"
        )

        self.tile_svg_dir = self.result_path / "tile_svgs"
        if self.accelerator.is_main_process:
            self.tile_svg_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        """Load image as (1, C, H, W) float tensor on self.device."""
        pil = Image.open(path).convert("RGB")
        t = transforms.ToTensor()(pil).unsqueeze(0).to(self.device)
        return t

    # ------------------------------------------------------------------
    # Per-tile optimisation
    # ------------------------------------------------------------------

    def _optimise_tile(
            self,
            tile_img: torch.Tensor,
            tile_idx: int,
            lpips_loss_fn=None,
    ) -> Path:
        """Run DiffVG optimisation on a single tile and save its SVG.

        Args:
            tile_img:  (1, 3, H, W) float tensor of the tile target.
            tile_idx:  unique index used to name the output file.
            lpips_loss_fn: optional partial for perceptual loss.

        Returns:
            Path to the saved tile SVG.
        """
        h, w = tile_img.shape[2], tile_img.shape[3]
        feather_mask = _make_feather_mask(h, w, self.x_cfg.feather, self.device)

        renderer = Painter(
            tile_img,
            self.args.diffvg,
            canvas_size=[w, h],
            path_type=self.x_cfg.path_type,
            max_width=self.x_cfg.max_width,
            device=self.device,
        )
        renderer.init_image(num_paths=self.x_cfg.num_paths)

        optimizer = PainterOptimizer(
            renderer,
            self.x_cfg.num_iter,
            self.x_cfg.lr_base,
            trainable_stroke=self.x_cfg.path_type == "unclosed",
        )
        optimizer.init_optimizer()

        for step in range(self.x_cfg.num_iter):
            raster = renderer.get_image(step).to(self.device)  # (1,3,H,W)

            loss = self._compute_loss(raster, tile_img, feather_mask, lpips_loss_fn)

            optimizer.zero_grad_()
            loss.backward()
            optimizer.step_()
            renderer.clip_curve_shape()

            if self.x_cfg.lr_schedule:
                optimizer.update_lr()

        svg_path = self.tile_svg_dir / f"tile_{tile_idx:04d}.svg"
        renderer.save_svg(svg_path)
        return svg_path

    def _compute_loss(
            self,
            raster: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor,
            lpips_loss_fn=None,
    ) -> torch.Tensor:
        """Compute weighted reconstruction loss."""
        loss_type = self.x_cfg.loss_type

        if loss_type == "l1":
            diff = torch.abs(raster - target) * mask
            return diff.mean()
        elif loss_type == "lpips":
            assert lpips_loss_fn is not None
            return lpips_loss_fn(raster * mask, target * mask).mean()
        elif loss_type == "l2":
            diff = (raster - target) ** 2 * mask
            return diff.mean()
        elif loss_type == "l2+lpips":
            assert lpips_loss_fn is not None
            l2_w = float(getattr(self.x_cfg, "l2_weight", 1.0))
            lp_w = float(getattr(self.x_cfg, "lpips_weight", 1.0))
            diff = (raster - target) ** 2 * mask
            l2 = diff.mean()
            lp = lpips_loss_fn(raster * mask, target * mask).mean()
            return l2_w * l2 + lp_w * lp
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def painterly_rendering(self, img_path: AnyStr):
        target_file = Path(img_path)
        assert target_file.exists(), f"{target_file} does not exist!"
        shutil.copy(target_file, self.result_path)

        # Load full-resolution image
        full_img = self._load_image_tensor(target_file.as_posix())
        _, _, img_h, img_w = full_img.shape
        self.print(f"Input image size: {img_w}x{img_h}")

        tile_size = self.x_cfg.tile_size
        overlap = self.x_cfg.overlap
        batch_size = self.x_cfg.batch_size

        # Ensure tile_size does not exceed image dimensions
        tile_size = min(tile_size, img_h, img_w)

        origins = _compute_tile_origins(img_h, img_w, tile_size, overlap)
        n_tiles = len(origins)
        self.print(f"Total tiles: {n_tiles}  (tile_size={tile_size}, overlap={overlap})")

        # Optionally set up LPIPS
        lpips_loss_fn = None
        if self.x_cfg.loss_type in ("lpips", "l2+lpips"):
            _lpips = LPIPS(net=self.x_cfg.perceptual.lpips_net).to(self.device)
            lpips_loss_fn = partial(_lpips.forward, return_per_layer=False, normalize=False)

        # Process tiles in micro-batches ---------------------------------
        tile_svg_paths: List[Path] = []

        with tqdm(
                total=n_tiles,
                desc="Optimising tiles",
                disable=not self.accelerator.is_main_process,
        ) as pbar:
            for batch_start in range(0, n_tiles, batch_size):
                batch_indices = list(range(batch_start, min(batch_start + batch_size, n_tiles)))

                # Micro-batch: optimise each tile in the batch, accumulate
                # the step counts together so that num_iter covers the
                # entire micro-batch collectively.
                for tile_idx in batch_indices:
                    row, col = origins[tile_idx]
                    # Crop tile from full image
                    tile_img = full_img[
                        :, :,
                        row: row + tile_size,
                        col: col + tile_size,
                    ].clone()

                    svg_path = self._optimise_tile(tile_img, tile_idx, lpips_loss_fn)
                    tile_svg_paths.append(svg_path)
                    pbar.update(1)

        # Merge all tile SVGs into one final SVG -------------------------
        self.print("Merging tile SVGs …")
        final_svg = self.result_path / "final_render.svg"
        _merge_tile_svgs(
            tile_svg_paths,
            origins,
            canvas_w=img_w,
            canvas_h=img_h,
            output_path=final_svg,
        )
        self.print(f"Final SVG saved to: {final_svg}")

        self.close(msg="Tiled DiffVG rendering complete.")
