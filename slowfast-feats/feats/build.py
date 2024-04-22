#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

from slowfast.models.video_model_builder import _POOL1

from .video_model_builder import SlowFastFeat, ResNetFeat
from .head_helper import ResNetBasicHead


MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg):
    """
    Builds the video model.
    The original function has been customized to load the feature extractor models with overwritten forward functions
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    # model = MODEL_REGISTRY.get(name)(cfg)

    # load feature extractor models
    width_per_group = cfg.RESNET.WIDTH_PER_GROUP
    pool_size = _POOL1[cfg.MODEL.ARCH]
    print(cfg.DATA)

    if name == "SlowFast":
        model = SlowFastFeat(cfg)
        model.head = ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES
                    // cfg.SLOWFAST.ALPHA
                    // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                ],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )
    elif name == "ResNet":
        model = ResNetFeat(cfg)
        model.head = ResNetBasicHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ]
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )

    # if cfg.NUM_GPUS:
    #     # Determine the GPU used by the current process
    #     cur_device = torch.cuda.current_device()
    #     # Transfer the model to the current GPU device
    #     model = model.cuda(device=cur_device)
    # # Use multi-process data parallel model in the multi-gpu setting
    # if cfg.NUM_GPUS > 1:

    return model
