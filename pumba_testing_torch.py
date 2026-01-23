#!/usr/bin/env python3
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dipy.io.image import load_nifti, save_nifti
from pumba_utils import transform_img, recover_img, post_process

TORCH_STATE   = "torch_weights.pt"

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, gn_groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=True)
        self.gn = nn.GroupNorm(gn_groups, out_ch, affine=True, eps=1e-3)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.gn(x)
        return x

class Pumba(nn.Module):
    def __init__(self, in_ch=1, num_classes=3, gn_groups=8):
        super().__init__()
        enc = [8, 16, 32, 64, 128]
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)

        ch = in_ch
        for f in enc:
            self.enc_blocks.append(ConvBlock(ch, f, gn_groups))
            ch = f

        self.bn_conv1 = nn.Conv3d(ch, 256, 3, padding=1, bias=True)
        self.bn_conv2 = nn.Conv3d(256, 256, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

        dec = [128, 64, 32, 16, 8]
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_blocks = nn.ModuleList()

        ch = 256
        for i, f in enumerate(dec):
            skip_ch = enc[::-1][i]
            self.dec_blocks.append(ConvBlock(ch + skip_ch, f, gn_groups))
            ch = f

        self.final_conv = nn.Conv3d(ch, num_classes, 1, bias=True)

    def forward(self, x):
        skips = []
        for blk in self.enc_blocks:
            x = blk(x)
            skips.append(x)
            x = self.pool(x)

        x = self.relu(self.bn_conv1(x))
        x = self.relu(self.bn_conv2(x))

        for i, blk in enumerate(self.dec_blocks):
            x = self.up(x)
            x = torch.cat([skips[-1 - i], x], dim=1)
            x = blk(x)

        x = self.final_conv(x)
        return F.softmax(x, dim=1)

file_path = sys.argv[1]
output_path = sys.argv[2]
image, affine = load_nifti(file_path)
x, params = transform_img(
    image, affine, target_voxsize=(2, 2, 2), final_size=(128, 128, 128)
)
x = np.interp(x, (np.percentile(x, 1), np.percentile(x, 99)), (0.0, 1.0))
x = np.reshape(x, (1, 1, 128, 128, 128)).astype(np.float32)  # (1,1,D,H,W)
x_torch = torch.from_numpy(x)

torch_m = Pumba().eval()
sd = torch.load(TORCH_STATE, map_location="cpu")
torch_m.load_state_dict(sd, strict=True)
with torch.no_grad():
    y_torch = torch_m(x_torch).numpy()
y_torch = np.squeeze(y_torch, axis=0)  # (C,D,H,W) 
y_torch = np.transpose(y_torch, (1, 2, 3, 0))  # (D,H,W,C)
if skip_postprocess := ("--skip-postprocess" in sys.argv):
    y_torch = np.argmax(y_torch, axis=-1)
else:
    y_torch = post_process(y_torch)
recovered = recover_img(y_torch, params, order=0)
save_nifti(output_path, np.round(recovered).astype(np.uint8), affine)
