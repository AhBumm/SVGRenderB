# -*- coding: utf-8 -*-
# Author: ximing / copilot
# Description: Pipeline registry – auto-discovers available pipelines.
# Copyright (c) 2025, XiMing Xing.
# License: MPL-2.0 License
"""
Registry / factory for SVG-render pipelines.

Usage (in svg_render.py or other entry-points)::

    from pytorch_svgrender.pipelines.registry import REGISTRY, get_pipeline

    pipe_cls = get_pipeline("diffvg")       # raises KeyError if unknown
    pipe = pipe_cls(cfg)

To register a new pipeline, call :func:`register` at module import time::

    from pytorch_svgrender.pipelines.registry import register

    @register("my_method")
    class MyPipeline:
        ...

The registry is intentionally lazy: pipeline classes are imported only when
:func:`get_pipeline` is called, so an unavailable optional dependency (e.g.
``pydiffvg``) does **not** break the entire import chain.
"""

from typing import Callable, Dict, Type

# ---------------------------------------------------------------------------
# Central registry: method_name -> (lazy_import_fn OR class)
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    """Class decorator that registers a pipeline under *name*."""
    def _decorator(cls):
        _REGISTRY[name] = lambda: cls
        return cls
    return _decorator


def register_lazy(name: str, import_fn: Callable):
    """Register a callable that returns the pipeline class (for lazy import)."""
    _REGISTRY[name] = import_fn


def get_pipeline(name: str) -> Type:
    """Return the pipeline class registered under *name*.

    Raises
    ------
    KeyError
        If *name* is not registered.
    ImportError
        If the lazy import fails (e.g. missing optional dependency).
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown pipeline '{name}'. Available: {available}"
        )
    return _REGISTRY[name]()


def available_methods():
    """Return sorted list of registered method names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Register built-in pipelines lazily so that missing optional deps
# (pydiffvg, CLIP, diffusers, …) do not break the import.
# ---------------------------------------------------------------------------

def _lazy(module_path: str, class_name: str):
    def _import():
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    return _import


register_lazy("diffvg",           _lazy("pytorch_svgrender.pipelines.DiffVG_pipeline",             "DiffVGPipeline"))
register_lazy("diffvg_tile",      _lazy("pytorch_svgrender.pipelines.DiffVGTile_pipeline",         "DiffVGTilePipeline"))
register_lazy("live",             _lazy("pytorch_svgrender.pipelines.LIVE_pipeline",               "LIVEPipeline"))
register_lazy("vectorfusion",     _lazy("pytorch_svgrender.pipelines.VectorFusion_pipeline",       "VectorFusionPipeline"))
register_lazy("clipasso",         _lazy("pytorch_svgrender.pipelines.CLIPasso_pipeline",           "CLIPassoPipeline"))
register_lazy("clipascene",       _lazy("pytorch_svgrender.pipelines.CLIPascene_pipeline",         "CLIPascenePipeline"))
register_lazy("diffsketcher",     _lazy("pytorch_svgrender.pipelines.DiffSketcher_pipeline",       "DiffSketcherPipeline"))
register_lazy("stylediffsketcher",_lazy("pytorch_svgrender.pipelines.DiffSketcher_stylized_pipeline", "StylizedDiffSketcherPipeline"))
register_lazy("clipdraw",         _lazy("pytorch_svgrender.pipelines.CLIPDraw_pipeline",           "CLIPDrawPipeline"))
register_lazy("styleclipdraw",    _lazy("pytorch_svgrender.pipelines.StyleCLIPDraw_pipeline",      "StyleCLIPDrawPipeline"))
register_lazy("wordasimage",      _lazy("pytorch_svgrender.pipelines.WordAsImage_pipeline",        "WordAsImagePipeline"))
register_lazy("clipfont",         _lazy("pytorch_svgrender.pipelines.CLIPFont_pipeline",           "CLIPFontPipeline"))
register_lazy("svgdreamer",       _lazy("pytorch_svgrender.pipelines.SVGDreamer_pipeline",         "SVGDreamerPipeline"))
