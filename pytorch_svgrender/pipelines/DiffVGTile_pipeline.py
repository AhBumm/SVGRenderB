# -*- coding: utf-8 -*-
# Author: copilot
# Description: DiffVGTile pipeline – tiled image-to-SVG vectorisation.
# Copyright (c) 2025, XiMing Xing.
# License: MPL-2.0 License
"""
DiffVGTilePipeline
==================
Vectorises high-resolution images by splitting them into overlapping tiles,
optimising each tile independently with DiffVG, and then compositing all
tile SVGs into a single full-resolution SVG.

Key design decisions
--------------------
* **Micro-batch loss accumulation** – each optimisation step processes
  ``batch_size`` tiles sequentially, accumulates the (normalised) loss
  from each tile, then calls ``backward()`` once and steps each tile's
  own ``Adam`` optimiser.  This means gradient memory never exceeds that
  of a single tile regardless of ``batch_size``.
* **Overlap & feather blending** – adjacent tiles share ``overlap`` pixels;
  a linear-ramp mask is used when compositing during logging so seams are
  not visible in the preview PNG.  The final SVG uses SVG ``<g translate>``
  so the viewer blends paths at the boundary automatically.
* **Lazy pydiffvg import** – the import is deferred to ``painterly_rendering``
  so that the module can be imported without pydiffvg being installed (it
  will only fail if the method is actually called).

Configuration (conf/x/diffvg_tile.yaml)
-----------------------------------------
See the YAML file for all tunable parameters and their documentation.
"""

import shutil
from pathlib import Path
from functools import partial
from typing import AnyStr, List, Tuple

import torch
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.libs.metric.lpips_origin import LPIPS
from pytorch_svgrender.tiling import tile_origins, crop_tile, pad_tile_to_size, feather_mask
from pytorch_svgrender.svgtools.merge import merge_tile_svgs
from pytorch_svgrender.plt import plot_img, plot_couple


class DiffVGTilePipeline(ModelState):
    """Tiled DiffVG vectorisation pipeline.

    For large images (e.g. 4096×4096) that do not fit into GPU memory as a
    single DiffVG scene, this pipeline:

    1. Splits the target image into overlapping ``tile_size × tile_size`` crops.
    2. Initialises one independent DiffVG ``Painter`` + ``PainterOptimizer``
       per tile.
    3. Runs ``num_iter`` optimisation steps. Each step:
       a. Groups tiles into micro-batches of size ``batch_size``.
       b. For each micro-batch, renders every tile and computes its loss;
          the losses are summed and ``backward()`` is called **once** per
          micro-batch.
       c. Each tile's individual optimizer is stepped.
    4. Saves all tile SVGs and merges them into one full-resolution SVG.
    """

    def __init__(self, args):
        logdir_ = (
            f"sd{args.seed}"
            f"-tile{args.x.tile_size}"
            f"-P{args.x.num_paths}"
        )
        super().__init__(args, log_path_suffix=logdir_)

        assert self.x_cfg.path_type in ('unclosed', 'closed')

        self.png_logs_dir = self.result_path / "png_logs"
        self.svg_logs_dir = self.result_path / "svg_logs"
        if self.accelerator.is_main_process:
            self.png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.svg_logs_dir.mkdir(parents=True, exist_ok=True)

        self.make_video = self.args.mv

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_target(self, img_path: AnyStr) -> torch.Tensor:
        """Load image → ``[1, 3, H, W]`` float32 tensor on device."""
        proc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ])
        pil = Image.open(img_path).convert("RGB")
        return proc(pil).to(self.device)

    def _build_tiles(
        self,
        target_img: torch.Tensor,
        tile_size: int,
        overlap: int,
        num_paths: int,
        path_type: str,
        max_width: float,
    ) -> Tuple[List, List, List[Tuple[int, int]]]:
        """Build per-tile Painter & PainterOptimizer instances."""
        # pydiffvg import deferred to here
        from pytorch_svgrender.painter.diffvg import Painter, PainterOptimizer

        _, _, H, W = target_img.shape
        origins = tile_origins(W, H, tile_size, tile_size, overlap)

        painters: List[Painter] = []
        optimizers: List[PainterOptimizer] = []

        for x, y in origins:
            tile = crop_tile(target_img, x, y, tile_size, tile_size)
            tile_h, tile_w = tile.shape[2], tile.shape[3]
            tile_padded = pad_tile_to_size(tile, tile_size, tile_size, value=1.0)

            p = Painter(
                tile_padded,
                self.args.diffvg,
                canvas_size=[tile_w, tile_h],
                path_type=path_type,
                max_width=max_width,
                device=self.device,
            )
            p.init_image(num_paths=num_paths)

            opt = PainterOptimizer(
                p,
                self.x_cfg.num_iter,
                self.x_cfg.lr_base,
                trainable_stroke=(path_type == 'unclosed'),
            )
            opt.init_optimizer()

            painters.append(p)
            optimizers.append(opt)

        return painters, optimizers, origins

    def _compute_loss(
        self,
        raster: torch.Tensor,
        target: torch.Tensor,
        loss_type: str,
        lpips_fn=None,
    ) -> torch.Tensor:
        if loss_type == 'l1':
            return torch.nn.functional.l1_loss(raster, target)
        elif loss_type == 'lpips':
            return lpips_fn(raster, target).mean()
        elif loss_type == 'l2+lpips':
            return (torch.nn.functional.mse_loss(raster, target)
                    + lpips_fn(raster, target).mean())
        else:  # default: l2 / MSE
            return torch.nn.functional.mse_loss(raster, target)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def painterly_rendering(self, img_path: AnyStr):
        """Run tiled DiffVG optimisation and save merged SVG.

        Parameters
        ----------
        img_path : str
            Path to the input raster image.
        """
        target_file = Path(img_path)
        assert target_file.exists(), f"Target file not found: {target_file}"
        shutil.copy(target_file, self.result_path)

        target_img = self._load_target(target_file.as_posix())
        _, _, img_h, img_w = target_img.shape
        self.print(f"Target image: {img_w}×{img_h}  path='{target_file}'")

        tile_size: int = self.x_cfg.tile_size
        overlap: int = self.x_cfg.overlap
        num_paths: int = self.x_cfg.num_paths
        path_type: str = self.x_cfg.path_type
        max_width: float = self.x_cfg.max_width
        num_iter: int = self.x_cfg.num_iter
        batch_size: int = max(1, self.x_cfg.batch_size)
        loss_type: str = self.x_cfg.loss_type

        # Build per-tile painters & optimizers
        painters, optimizers, origins = self._build_tiles(
            target_img, tile_size, overlap, num_paths, path_type, max_width
        )
        n_tiles = len(painters)
        self.print(f"Total tiles: {n_tiles}  (tile_size={tile_size}, overlap={overlap}, "
                   f"batch_size={batch_size})")

        # Crop target tiles for loss computation
        target_tiles = [
            pad_tile_to_size(
                crop_tile(target_img, x, y, tile_size, tile_size),
                tile_size, tile_size, value=1.0
            )
            for x, y in origins
        ]

        # Optional LPIPS loss
        lpips_fn = None
        if loss_type in ('lpips', 'l2+lpips'):
            lpips_fn = partial(
                LPIPS(net=self.x_cfg.perceptual.lpips_net).to(self.device).forward,
                return_per_layer=False, normalize=False,
            )

        # Build tile index batches
        tile_indices = list(range(n_tiles))
        batches = [
            tile_indices[i: i + batch_size]
            for i in range(0, n_tiles, batch_size)
        ]

        # ----------------------------------------------------------------
        # Optimisation loop
        # ----------------------------------------------------------------
        with tqdm(
            initial=self.step,
            total=num_iter,
            disable=not self.accelerator.is_main_process,
            desc="diffvg_tile",
        ) as pbar:
            while self.step < num_iter:
                total_loss_val = 0.0

                for batch_idx in batches:
                    # Zero grads for all tiles in this micro-batch
                    for ti in batch_idx:
                        optimizers[ti].zero_grad_()

                    # Accumulate loss over micro-batch tiles
                    batch_loss = torch.tensor(0.0, device=self.device)
                    for ti in batch_idx:
                        raster = painters[ti].get_image(self.step).to(self.device)
                        tile_loss = self._compute_loss(
                            raster, target_tiles[ti], loss_type, lpips_fn
                        )
                        batch_loss = batch_loss + tile_loss / len(batch_idx)

                    # Single backward pass for the whole micro-batch
                    batch_loss.backward()
                    total_loss_val += batch_loss.item()

                    # Step each tile's optimizer independently
                    for ti in batch_idx:
                        optimizers[ti].step_()
                        painters[ti].clip_curve_shape()
                        if self.x_cfg.lr_schedule:
                            optimizers[ti].update_lr()

                pbar.set_description(
                    f"step={self.step}  loss={total_loss_val / len(batches):.4f}"
                )

                # Periodic logging
                if (self.step % self.args.save_step == 0
                        and self.accelerator.is_main_process):
                    self._log_tiles(painters, target_tiles, origins,
                                    img_w, img_h, tile_size, overlap)

                self.step += 1
                pbar.update(1)

        # ----------------------------------------------------------------
        # Save final tile SVGs and merge
        # ----------------------------------------------------------------
        tile_svg_paths = self._save_tile_svgs(painters, origins, prefix="final")
        merged_svg = self.result_path / "final_merged.svg"
        merge_tile_svgs(
            [str(p) for p in tile_svg_paths],
            origins,
            str(merged_svg),
            img_w,
            img_h,
        )
        self.print(f"Merged SVG saved: {merged_svg}")
        self.close(msg="Tiled DiffVG rendering complete.")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _save_tile_svgs(self, painters, origins, prefix: str) -> List[Path]:
        paths = []
        for ti, (p, (x, y)) in enumerate(zip(painters, origins)):
            svg_path = self.svg_logs_dir / f"{prefix}_tile{ti:04d}_x{x}_y{y}.svg"
            p.save_svg(svg_path)
            paths.append(svg_path)
        return paths

    def _log_tiles(self, painters, target_tiles, origins,
                   img_w, img_h, tile_size, overlap):
        """Composite rendered tiles into a preview PNG for logging."""
        canvas = torch.ones(1, 3, img_h, img_w, device=self.device)
        weight = torch.zeros(1, 1, img_h, img_w, device=self.device)

        for ti, (p, (x, y)) in enumerate(zip(painters, origins)):
            with torch.no_grad():
                raster = p.get_image(self.step).to(self.device)

            th, tw = raster.shape[2], raster.shape[3]
            mask = feather_mask(tw, th, overlap, self.device)

            x2, y2 = x + tw, y + th
            canvas[:, :, y:y2, x:x2] = (
                canvas[:, :, y:y2, x:x2] * weight[:, :, y:y2, x:x2]
                + raster * mask
            ) / (weight[:, :, y:y2, x:x2] + mask).clamp(min=1e-6)
            weight[:, :, y:y2, x:x2] = (
                weight[:, :, y:y2, x:x2] + mask
            ).clamp(max=1.0)

        plot_img(canvas, self.png_logs_dir, fname=f"composite_iter{self.step}")
