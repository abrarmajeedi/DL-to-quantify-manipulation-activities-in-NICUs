#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch

from slowfast.models.head_helper import ResNetBasicHead


class ResNetBasicHead(ResNetBasicHead):
    # Overwrite function to return features
    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))


        # print([feat_path.shape for feat_path in pool_out ])
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # save features
        feat = x.clone().detach()
        # print(feat.shape)
        # flatten the features tensor
        feat = feat.mean(3).mean(2).reshape(feat.shape[0], -1)
        # print(feat.shape)
        return feat

"""
crp = 224
[[[ 32 // 4     // 1,    crp // 32 // 1,   crp // 32 // 1],       ],      [     32 // 1,     crp // 32 // 1,          crp // 32 // 1,    ],  ]

len inputs:  2
inputs 0 :  torch.Size([2, 2048, 8, 10, 13])
inputs 1 :  torch.Size([2, 256, 32, 10, 13])
avg pool out 0 :  torch.Size([2, 2048, 1, 2, 5])
avg pool out 1 :  torch.Size([2, 256, 1, 2, 5])
x:  torch.Size([2, 2304, 1, 2, 5])
x perm:  torch.Size([2, 1, 2, 5, 2304])
feat.mean(3).mean(2).reshape(feat.shape[0], -1)
feat : torch.Size([2, 2304])


resnet
len inputs:  1
inputs 0 :  torch.Size([2, 2048, 4, 10, 13])
avg pool out 0 :  torch.Size([2, 2048, 1, 2, 5])
x:  torch.Size([2, 2048, 1, 2, 5])
feat : torch.Size([2, 2048])


import torch
import torch.nn as nn
crp = 224
def func(crp):
    pool_size = [( 32 // 4     // 1,    crp // 32 // 1,   crp // 32 // 1),      (    32 // 1,     crp // 32 // 1,          crp // 32 // 1,)  ]
    pools = [nn.AvgPool3d(pool_size[0], stride=1),nn.AvgPool3d(pool_size[1], stride=1)]
    inputs = [torch.randn([2, 2048, 8, 10, 13]),torch.randn([2, 256, 32, 10, 13])]
    outputs = []
    for i in range(2):
        outputs.append(pools[i](inputs[i]))
        print(outputs[i].shape)
    x = torch.cat(outputs, 1)
    # (N, C, T, H, W) -> (N, T, H, W, C).
    x = x.permute((0, 2, 3, 4, 1))
    print(x.mean(3).mean(2).reshape(x.shape[0], -1).shape)
    
func(400)


main inp: torch.Size([3, 3, 4, 300, 400])
main inp: torch.Size([3, 3, 32, 300, 400])
len inputs:  2
inputs 0 :  torch.Size([2, 2048, 4, 10, 13])
inputs 1 :  torch.Size([2, 256, 32, 10, 13])
avg pool out 0 :  torch.Size([2, 2048, 1, 2, 5])
avg pool out 1 :  torch.Size([2, 256, 1, 2, 5])
x:  torch.Size([2, 2304, 1, 2, 5])
x perm:  torch.Size([2, 1, 2, 5, 2304])
feat : torch.Size([2, 2304])
len inputs:  2
inputs 0 :  torch.Size([1, 2048, 4, 10, 13])
inputs 1 :  torch.Size([1, 256, 32, 10, 13])
avg pool out 0 :  torch.Size([1, 2048, 1, 2, 5])
avg pool out 1 :  torch.Size([1, 256, 1, 2, 5])
x:  torch.Size([1, 2304, 1, 2, 5])
x perm:  torch.Size([1, 1, 2, 5, 2304])
feat : torch.Size([1, 2304])
(3, 2304)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 260/260 [01:18<00:00,  3.29it/s]
(1039, 2304)
"""