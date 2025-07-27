
"""ESEG Implementation (Event‑based Segmentation boosted by Explicit Edge‑semantic Guidance)
==========================================================================================
This file contains the code for the **ESEG** core framework structure described in
"ESEG: Event‑Based Segmentation Boosted by Explicit Edge‑Semantic Guidance"
(AAAI‑25, Zhao *et al.*) in PyTorch for quick reference and understanding. 
The code keeps the exact computational graph of the original prototype that 
directly reference the corresponding sections, figures, and equations in the paper.

The backbone is composed of three main parts:

1. **MixVisionTransformer** *(MiT)* encoder (Identical to SegFormer‑B0/B1‑style
   encoders) – §Approach, *Dense‑semantic branch*.
2. **FusionModule (D2CAF)** – our re‑implementation of *Density‑Aware
   Dynamic‑Window Cross‑Attention Fusion* (§D2CAF, *Fig. 4*, Eq. 1‑8).  Both
   the lightweight "Lite" variant (commented‑out) and the full "Nano" variant
   used in the paper are included.
3. **Utility blocks** (Mlp, Attention, Block, OverlapPatchEmbed, DwConv) which
   are unchanged w.r.t. the SegFormer reference implementation.



"""

import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
# NOTE: `BACKBONES` is used by MMSegmentation; uncomment if needed.
# from mmseg.models.builder import BACKBONES
from mmengine.runner import load_checkpoint

# -----------------------------------------------------------------------------
# Density‑Aware Dynamic‑Window Cross‑Attention Fusion (D2CAF)
# (see §D2CAF, *Fig. 4*).
# -----------------------------------------------------------------------------



class FusionModule(nn.Module):
    """Full D2CAF block – see paper §D2CAF and *Fig. 4*.

    Parameters
    ----------
    C_edge : int
        Channel dimension of the incoming *edge‑semantic* features (EFi).
    C_fused : int
        Channel dimension of the fused features from the previous stage
        (DF'_{i‑1}).
    C_semantic : int
        Channel dimension of the *dense‑semantic* pathway (DFi).
    Htrain, Wtrain : int
        Spatial resolution of the feature map during **training** (used to
        pre‑compute the attention masks discussed in §Dynamic Window Masking).
    Htest,  Wtest  : int
        Spatial resolution during **inference**.  Separate masks avoid run‑time
        allocation on every forward pass.
    mask_size : int
        Square window size *w* controlling the dynamic neighborhood (*Eq. 2*).
    """

    def __init__(self,
                 C_edge: int,
                 C_fused: int,
                 C_semantic: int,
                 Htrain: int,
                 Wtrain: int,
                 Htest: int,
                 Wtest: int,
                 mask_size: int):
        super().__init__()

        # 1×1 convs to align channel dimensions before fusion  (Eq. 1)
        self.conv_align_channels = nn.Conv2d(C_edge,   C_semantic, 1)
        self.conv_fused_align_channels = nn.Conv2d(C_fused, C_semantic, 1)

        # 1×1 conv projections for Q/K (operating on flattened tokens)
        # dim = 2*C_semantic because edge & semantic tokens are concatenated.
        self.conv_edge_projection = nn.Conv1d(2 * C_semantic, 2 * C_semantic, 1)
        self.conv_semantic_projection = nn.Conv1d(2 * C_semantic, 2 * C_semantic, 1)

        # Final fusion conv (concatenate + 1×1 – see *Eq. 8*)
        self.conv_fusion = nn.Conv2d(2 * C_semantic, C_semantic, 1)

        # Pre‑compute binary attention masks for **training** and **testing**
        # following Algorithm 2 (*Dynamic Window Masking*, §D2CAF).
        self.attention_mask_train = self._create_attention_masks(Htrain, Wtrain, mask_size)
        self.attention_mask_test  = self._create_attention_masks(Htest,  Wtest,  mask_size)

    # ------------------------------------------------------------------
    # Helper – pre‑compute dynamic‑window masks (Fig. 5, Eq. 2‑3)
    # ------------------------------------------------------------------
    @staticmethod
    def _create_attention_masks(H: int, W: int, mask_size: int) -> torch.Tensor:
        """Generates *M_dw* for all pixel locations at a given resolution."""
        # The mask tensor has shape (N, H*W) where N = H*W.  Each row i is the
        # flattened attention bias for query position i (−1e4 outside the window).
        masks = -10000 * torch.ones(H * W, H * W)

        half = mask_size // 2
        for q_x in range(H):
            for q_y in range(W):
                x0, x1 = max(0, q_x - half), min(H, q_x + half + 1)
                y0, y1 = max(0, q_y - half), min(W, q_y + half + 1)

                local = -10000 * torch.ones(H, W)
                local[x0:x1, y0:y1] = 0  # 0‑bias inside the dynamic window
                masks[q_x * W + q_y, :] = local.flatten()
        print(f"[D2CAF] Pre‑computed mask for {H}×{W} – window {mask_size} done.")
        return masks.cuda()  # Move to GPU once to avoid host<‑>device copy.

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self,
                edge_features: torch.Tensor,
                semantic_features: torch.Tensor,
                fused_features: torch.Tensor) -> torch.Tensor:
        """Fuse *dense‑semantic* and *edge‑semantic* cues as in *Eq. 1‑8*."""

        # --------------------------------------------------------------
        # (1) Spatial alignment  (‑‑> EFi~, DF'_{i‑1}~ in paper)
        # --------------------------------------------------------------
        _, C_sem, H, W = semantic_features.shape

        edge_features = F.interpolate(edge_features, (H, W), mode="bilinear", align_corners=False)
        edge_features = self.conv_align_channels(edge_features)

        fused_features = F.interpolate(fused_features, (H, W), mode="bilinear", align_corners=False)
        fused_features = self.conv_fused_align_channels(fused_features)

        # --------------------------------------------------------------
        # (2) Compute similarity & density terms (DIM)  (Eq. 3, Fig. 4)
        # --------------------------------------------------------------
        l2_edge      = torch.norm(edge_features,     p=2, dim=1, keepdim=True)
        l2_semantic  = torch.norm(semantic_features, p=2, dim=1, keepdim=True)
        edge_norm    = edge_features     / (l2_edge + 1e-6)
        sem_norm     = semantic_features / (l2_semantic + 1e-6)

        similarity   = (edge_norm * sem_norm).sum(dim=1, keepdim=True)  # [0,1]
        similarity   = (similarity + 1) / 2

        density      = torch.norm(edge_features, p=2, dim=1, keepdim=True)  # edge ‑> sparse
        density_flat = density.view(density.size(0), -1)
        density_flat = F.softmax(density_flat, dim=1) * density_flat.shape[-1]  # re‑scale
        density      = density_flat.view_as(density)

        # --------------------------------------------------------------
        # (3) Token preparation  (flatten H×W => N)  – see Eq. 4‑6
        # --------------------------------------------------------------
        b, c, h, w = semantic_features.size()
        edge_t   = (edge_features * similarity).reshape(b, c, -1)
        sem_t    = (semantic_features * density).reshape(b, c, -1)
        fused_t  = fused_features.reshape(b, c, -1)
        sem_raw  = semantic_features.reshape(b, c, -1)

        # Duplicate tokens so [edge|edge] and [semantic|fused]
        edge_cat = torch.cat([edge_t, edge_t], dim=1)  # Q
        sem_cat  = torch.cat([sem_t, fused_t], dim=1)  # K
        sem_val  = torch.cat([sem_raw, fused_t], dim=1)  # V (note paper §Mapping)

        # Linear projections (1×1 conv over channel dim) – channels act as seq len
        q = self.conv_edge_projection(edge_cat)
        k = self.conv_semantic_projection(sem_cat)

        # --------------------------------------------------------------
        # (4) Scaled dot‑product cross‑attention with dynamic mask  (Eq. 7)
        # --------------------------------------------------------------
        attn = torch.bmm(q.transpose(1, 2), k) / ((2 * c) ** 0.5)  # [B, N, N]
        mask = self.attention_mask_train if h == w else self.attention_mask_test
        attn = attn + mask  # broadcast over batch dim
        attn = F.softmax(attn, dim=-1)

        out  = torch.bmm(attn, sem_val.transpose(1, 2))  # [B, N, 2c]
        out  = out.view(b, h, w, 2 * c).permute(0, 3, 1, 2).contiguous()

        # --------------------------------------------------------------
        # (5) Final fusion conv  (Eq. 8)
        # --------------------------------------------------------------
        fused = self.conv_fusion(out)
        print("[D2CAF] Feature shapes:", fused.shape, semantic_features.shape)

        # Concatenate with original semantic for downstream heads if desired.
        return fused


# -----------------------------------------------------------------------------
# Feed‑Forward Network (identical to SegFormer except added DW‑Conv for local
# positional encoding – §Approach, *Dense‑semantic branch*)
# -----------------------------------------------------------------------------


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features or in_features

        self.fc1    = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)  # 3×3 depthwise conv – local bias
        self.act    = act_layer()
        self.fc2    = nn.Linear(hidden_features, out_features)
        self.drop   = nn.Dropout(drop)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H: int, W: int):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)  # Local positional encoding (§SegFormer)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -----------------------------------------------------------------------------
# Multi‑Head Attention with Spatial‑Reduction (identical to SegFormer except we
# keep extensive inline logging to match original experimentation).
# -----------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # (B, heads, N, C')

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        k, v = kv[:, :, 0], kv[:, :, 1]

        k = k.permute(0, 2, 1, 3)  # (B, heads, M, C')
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------------------------------------------------------
# Transformer Block (norm‑attn‑mlp) with DropPath – unchanged.
# -----------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=hidden, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H: int, W: int):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# -----------------------------------------------------------------------------
# Overlap Patch Embedding (7‑4‑3 vs. 3‑2‑1 kernel/stride for stage1‑4) – matches
# SegFormer (§Approach, dense‑semantic branch).
# -----------------------------------------------------------------------------


class OverlapPatchEmbed(nn.Module):
    """Image‑to‑patch embedding with *overlap* (kernel > stride)."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size   = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, H, W


# -----------------------------------------------------------------------------
# MixVisionTransformer (SegFormer encoder) + edge‑guided fusion (ESEG‑specific)
# -----------------------------------------------------------------------------


class MixVisionTransformer(nn.Module):
    """Encoder backbone with four stages + ESEG fusion pathway."""

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths      = depths

        # ------------------------------------------------------------------
        # Patch embeddings (OverlapPatchEmbed) – stage 1‑4
        # ------------------------------------------------------------------
        self.patch_embed1 = OverlapPatchEmbed(img_size,       7, 4, in_chans,      embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size // 4,  3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size // 8,  3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])

        # Fusion modules – channel configs follow paper implementation details.
        self.fusion_module1 = FusionModule(64,  32,  32, 120, 120, 110, 160, 11)
        self.fusion_module2 = FusionModule(128, 32,  64,  60,  60,  55,  80,  7)
        self.fusion_module3 = FusionModule(256, 64, 160,  30,  30,  28,  40,  5)
        self.fusion_module4 = FusionModule(512, 160, 256,  15,  15,  14,  20,  3)

        # ------------------------------------------------------------------
        # Transformer encoder blocks
        # ------------------------------------------------------------------
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale,
                  drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer,
                  sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale,
                  drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer,
                  sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale,
                  drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer,
                  sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale,
                  drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer,
                  sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    # ---------------------------- utility -----------------------------
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # ---------------------------- MMSeg hook -------------------------
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    # ---------------------------- forward passes ----------------------
    def forward_features(self, x, x1, x2, x3, x4):
        """Forward pass returning:
        * `out`  – list of MiT stage outputs (DF1‑DF4) *before* fusion
        * `outs` – list of concatenated tensors [DFi, DF' i] (see paper Eq. 1)
        """
        B = x.shape[0]
        outs, out = [], []

        # ---------------- Stage 1 ----------------
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:  # local and global context mixing
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        fused1 = self.fusion_module1(x1, x, x)
        outs.append(torch.cat([x, fused1], dim=1))
        print("[Stage1]", x.shape, fused1.shape)

        # ---------------- Stage 2 ----------------
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        fused2 = self.fusion_module2(x2, x, fused1)
        outs.append(torch.cat([x, fused2], dim=1))
        print("[Stage2]", x.shape, fused2.shape)

        # ---------------- Stage 3 ----------------
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        fused3 = self.fusion_module3(x3, x, fused2)
        outs.append(torch.cat([x, fused3], dim=1))
        print("[Stage3]", x.shape, fused3.shape)

        # ---------------- Stage 4 ----------------
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        fused4 = self.fusion_module4(x4, x, fused3)
        outs.append(torch.cat([x, fused4], dim=1))
        print("[Stage4]", x.shape, fused4.shape)

        return out, outs

    def forward(self, x, x1, x2, x3, x4):
        return self.forward_features(x, x1, x2, x3, x4)


# -----------------------------------------------------------------------------
# Depth‑wise 3×3 conv (DWConv) – adds positional bias in MLP path (SegFormer)
# -----------------------------------------------------------------------------


class DWConv(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# -----------------------------------------------------------------------------
# Backbone factory helpers (mit_b0‑b5) – *identical* to MMSeg, listed for
# completeness.  Register with @BACKBONES if using MMSEG.
# -----------------------------------------------------------------------------


def _build_mit(depths, **kwargs):
    return MixVisionTransformer(depths=depths, **kwargs)


class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4,
                         embed_dims=[32, 64, 160, 256],
                         num_heads=[1, 2, 5, 8],
                         mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6),
                         depths=[2, 2, 2, 2],
                         sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0,
                         drop_path_rate=0.1,
                         **kwargs)


class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4,
                         embed_dims=[64, 128, 320, 512],
                         num_heads=[1, 2, 5, 8],
                         mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6),
                         depths=[2, 2, 2, 2],
                         sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0,
                         drop_path_rate=0.1,
                         **kwargs)


class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4,
                         embed_dims=[64, 128, 320, 512],
                         num_heads=[1, 2, 5, 8],
                         mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6),
                         depths=[3, 4, 6, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0,
                         drop_path
