# -*- coding: utf-8 -*-
# Author: ximing xing
# Description: the main func of this project.
# Copyright (c) 2023, XiMing Xing.

from functools import partial

from accelerate.utils import set_seed
import hydra
import omegaconf

from pytorch_svgrender.utils import render_batch_wrap, get_seed_range
from pytorch_svgrender.pipelines.registry import available_methods, get_pipeline

# ---------------------------------------------------------------------------
# METHODS is derived from the pipeline registry so that new pipelines
# registered via registry.register_lazy appear here automatically.
# ---------------------------------------------------------------------------
METHODS = available_methods()


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(cfg: omegaconf.DictConfig):
    """
    The project configuration is stored in './conf/config.yaml'
    And method configurations are stored in './conf/x/'
    """

    # print(omegaconf.OmegaConf.to_yaml(cfg))
    flag = cfg.x.method
    if flag not in METHODS:
        raise ValueError(
            f"'{flag}' is not currently supported! "
            f"Available methods: {sorted(METHODS)}"
        )

    # seed prepare
    set_seed(cfg.seed)
    seed_range = get_seed_range(cfg.srange) if cfg.multirun else None

    # render function
    render_batch_fn = partial(render_batch_wrap, cfg=cfg, seed_range=seed_range)

    # -----------------------------------------------------------------------
    # img2svg pipelines
    # -----------------------------------------------------------------------
    if flag == "diffvg":
        pipe = get_pipeline("diffvg")(cfg)
        pipe.painterly_rendering(cfg.target)

    elif flag == "diffvg_tile":  # img2svg (tiled high-resolution)
        pipe = get_pipeline("diffvg_tile")(cfg)
        pipe.painterly_rendering(cfg.target)

    elif flag == "live":
        pipe = get_pipeline("live")(cfg)
        pipe.painterly_rendering(cfg.target)

    # -----------------------------------------------------------------------
    # text2svg pipelines
    # -----------------------------------------------------------------------
    elif flag == "vectorfusion":
        if not cfg.multirun:
            pipe = get_pipeline("vectorfusion")(cfg)
            pipe.painterly_rendering(cfg.prompt)
        else:
            render_batch_fn(pipeline=get_pipeline("vectorfusion"), text_prompt=cfg.prompt)

    elif flag == "svgdreamer":
        if not cfg.multirun:
            pipe = get_pipeline("svgdreamer")(cfg)
            pipe.painterly_rendering(cfg.prompt)
        else:
            render_batch_fn(pipeline=get_pipeline("svgdreamer"), text_prompt=cfg.prompt, target_file=None)

    elif flag == "wordasimage":
        pipe = get_pipeline("wordasimage")(cfg)
        pipe.painterly_rendering(cfg.x.word, cfg.prompt, cfg.x.optim_letter)

    # -----------------------------------------------------------------------
    # img2sketch pipelines
    # -----------------------------------------------------------------------
    elif flag == "clipasso":
        pipe = get_pipeline("clipasso")(cfg)
        pipe.painterly_rendering(cfg.target)

    elif flag == 'clipascene':
        pipe = get_pipeline("clipascene")(cfg)
        pipe.painterly_rendering(cfg.target)

    # -----------------------------------------------------------------------
    # text+img hybrid pipelines
    # -----------------------------------------------------------------------
    elif flag == "clipdraw":
        if not cfg.multirun:
            pipe = get_pipeline("clipdraw")(cfg)
            pipe.painterly_rendering(cfg.prompt)
        else:
            render_batch_fn(pipeline=get_pipeline("clipdraw"), prompt=cfg.prompt)

    elif flag == "clipfont":
        if not cfg.multirun:
            pipe = get_pipeline("clipfont")(cfg)
            pipe.painterly_rendering(svg_path=cfg.target, prompt=cfg.prompt)
        else:
            render_batch_fn(pipeline=get_pipeline("clipfont"), svg_path=cfg.target, prompt=cfg.prompt)

    elif flag == "styleclipdraw":
        if not cfg.multirun:
            pipe = get_pipeline("styleclipdraw")(cfg)
            pipe.painterly_rendering(cfg.prompt, style_fpath=cfg.target)
        else:
            render_batch_fn(pipeline=get_pipeline("styleclipdraw"), prompt=cfg.prompt, style_fpath=cfg.target)

    elif flag == "diffsketcher":
        if not cfg.multirun:
            pipe = get_pipeline("diffsketcher")(cfg)
            pipe.painterly_rendering(cfg.prompt)
        else:
            render_batch_fn(pipeline=get_pipeline("diffsketcher"), prompt=cfg.prompt)

    elif flag == "stylediffsketcher":
        if not cfg.multirun:
            pipe = get_pipeline("stylediffsketcher")(cfg)
            pipe.painterly_rendering(cfg.prompt, style_fpath=cfg.target)
        else:
            render_batch_fn(pipeline=get_pipeline("stylediffsketcher"), prompt=cfg.prompt, style_fpath=cfg.style_file)


if __name__ == '__main__':
    main()
