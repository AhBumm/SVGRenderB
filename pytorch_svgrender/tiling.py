# -*- coding: utf-8 -*-
# Author: copilot
# Description: Shared tiling utilities – tile-origin computation, crop/pad,
#              feather (overlap-blend) masks.
# Copyright (c) 2025, XiMing Xing.
# License: MPL-2.0 License
"""
Tiling utilities used by DiffVGTilePipeline and any future tiled pipeline.

Public API
----------
tile_origins(img_w, img_h, tile_w, tile_h, overlap)
    Return a list of (x_offset, y_offset) for each tile.

crop_tile(img_tensor, x, y, tile_w, tile_h)
    Crop a [N, C, H, W] tensor at position (x, y) with given tile size.

feather_mask(tile_w, tile_h, overlap, device)
    Return a [1, 1, tile_h, tile_w] weight mask that blends tile edges
    smoothly with an *overlap*-pixel linear ramp.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tile origin computation
# ---------------------------------------------------------------------------

def tile_origins(
    img_w: int,
    img_h: int,
    tile_w: int,
    tile_h: int,
    overlap: int = 64,
) -> List[Tuple[int, int]]:
    """Compute (x, y) pixel origins for a regular tile grid.

    Tiles overlap by *overlap* pixels on each edge so that per-tile
    SVG paths at boundaries blend smoothly when composited.

    Parameters
    ----------
    img_w, img_h : int
        Full-image canvas dimensions.
    tile_w, tile_h : int
        Tile canvas dimensions (before overlap).
    overlap : int
        Number of pixels each adjacent tile shares on each side.

    Returns
    -------
    List[Tuple[int, int]]
        List of ``(x_offset, y_offset)`` origins in raster order
        (left-to-right, top-to-bottom).
    """
    stride_x = max(tile_w - overlap, 1)
    stride_y = max(tile_h - overlap, 1)

    origins: List[Tuple[int, int]] = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            origins.append((x, y))
            if x + tile_w >= img_w:
                break
            x += stride_x
        if y + tile_h >= img_h:
            break
        y += stride_y
    return origins


def num_tiles(img_w: int, img_h: int, tile_w: int, tile_h: int, overlap: int = 64) -> int:
    """Return the total number of tiles for the given canvas / tile settings."""
    return len(tile_origins(img_w, img_h, tile_w, tile_h, overlap))


# ---------------------------------------------------------------------------
# Tile crop helper
# ---------------------------------------------------------------------------

def crop_tile(
    img: torch.Tensor,
    x: int,
    y: int,
    tile_w: int,
    tile_h: int,
) -> torch.Tensor:
    """Crop ``img`` at pixel origin ``(x, y)`` with size ``(tile_w, tile_h)``.

    The image is *not* padded; the actual crop size may be smaller than
    ``(tile_h, tile_w)`` at the right/bottom border.

    Parameters
    ----------
    img : torch.Tensor
        Shape ``[N, C, H, W]`` or ``[C, H, W]``.
    x, y : int
        Top-left pixel of the crop (column, row).
    tile_w, tile_h : int
        Requested tile size.

    Returns
    -------
    torch.Tensor
        Cropped tile with the same number of leading dimensions as *img*.
    """
    ndim = img.ndim
    if ndim == 3:
        img = img.unsqueeze(0)

    _, _, H, W = img.shape
    x2 = min(x + tile_w, W)
    y2 = min(y + tile_h, H)
    tile = img[:, :, y:y2, x:x2]

    if ndim == 3:
        tile = tile.squeeze(0)
    return tile


def pad_tile_to_size(
    tile: torch.Tensor,
    tile_w: int,
    tile_h: int,
    value: float = 1.0,
) -> torch.Tensor:
    """Right-pad / bottom-pad *tile* to ``(tile_h, tile_w)`` with constant *value*.

    Useful when a border tile is smaller than the nominal tile size.

    Parameters
    ----------
    tile : torch.Tensor
        Shape ``[N, C, H, W]``.
    tile_w, tile_h : int
        Target tile dimensions.
    value : float
        Fill value (default: 1.0 = white for RGB images).

    Returns
    -------
    torch.Tensor
        Shape ``[N, C, tile_h, tile_w]``.
    """
    _, _, H, W = tile.shape
    pad_w = tile_w - W
    pad_h = tile_h - H
    if pad_w == 0 and pad_h == 0:
        return tile
    # F.pad order: (left, right, top, bottom)
    return F.pad(tile, (0, pad_w, 0, pad_h), mode="constant", value=value)


# ---------------------------------------------------------------------------
# Feather (overlap blending) mask
# ---------------------------------------------------------------------------

def feather_mask(
    tile_w: int,
    tile_h: int,
    overlap: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a ``[1, 1, tile_h, tile_w]`` linear-ramp blending mask.

    The mask is 1.0 in the centre and ramps linearly to 0.0 at each edge
    over a width of ``overlap`` pixels.  Multiplying each tile's rendered
    image by this mask before accumulating into the full canvas produces a
    smooth seam-free blend.

    Parameters
    ----------
    tile_w, tile_h : int
        Tile canvas dimensions.
    overlap : int
        Width of the fade region at each edge.
    device : torch.device, optional
        Target device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Shape ``[1, 1, tile_h, tile_w]``, values in ``[0, 1]``.
    """
    mask_x = _edge_ramp(tile_w, overlap, device)   # [tile_w]
    mask_y = _edge_ramp(tile_h, overlap, device)   # [tile_h]
    # Outer product → [tile_h, tile_w]
    mask = mask_y.unsqueeze(1) * mask_x.unsqueeze(0)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def _edge_ramp(size: int, ramp: int, device: torch.device) -> torch.Tensor:
    """1-D ramp: linear 0→1 over first *ramp* pixels, 1 in the middle,
    linear 1→0 over last *ramp* pixels."""
    ramp = min(ramp, size // 2)
    ones = torch.ones(size, device=device)
    if ramp <= 0:
        return ones
    t = torch.linspace(0.0, 1.0, ramp, device=device)
    ones[:ramp] = t
    ones[-ramp:] = t.flip(0)
    return ones
