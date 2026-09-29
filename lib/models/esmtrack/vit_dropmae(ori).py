import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.esmtrack.HeatMapVis import visualize_heatmap
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock
from .utils import combine_tokens, token2feature, feature2token
try:
    from dinov3.models.vision_transformer import vit_base
    HAS_DINOV3 = True
except:
    HAS_DINOV3 = False
    print("[Warning] dinov3 repo not found.")

_logger = logging.getLogger(__name__)

class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear.
    y = W x + (alpha / r) * (B (A x))
    We keep original Linear (W) as .linear and add low-rank params A, B.
    """
    def __init__(self, in_features, out_features, r=4, alpha=16, dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # base (original) linear - we'll copy weights from existing linear when replacing
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if r > 0:
            # LoRA params: A (r x in), B (out x r)
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        else:
            self.lora_A = None
            self.lora_B = None

        self.merged = False
        self.use_lora = True

    def enable_lora(self):
        self.use_lora = True

    def disable_lora(self):
        self.use_lora = False

    def forward(self, x):
        base = self.linear(x)
        if self.r > 0 and (not self.merged):
            x_d = self.dropout(x)
            # x (..., in) @ A.T(in, r) -> (..., r)
            lora_inter = x_d.matmul(self.lora_A.t())
            # (..., r) @ B.T(r, out) -> (..., out) ; B is (out, r)
            lora_out = lora_inter.matmul(self.lora_B.t())
            return base + lora_out * self.scaling
        else:
            return base

    def merge_weights(self):
        """Merge LoRA into base linear.weight (in-place)."""
        if self.r > 0 and not self.merged:
            delta = (self.lora_B @ self.lora_A) * (self.scaling)  # (out, in)
            self.linear.weight.data += delta
            self.merged = True

    def unmerge_weights(self):
        if self.r > 0 and self.merged:
            delta = (self.lora_B @ self.lora_A) * (self.scaling)
            self.linear.weight.data -= delta
            self.merged = False

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B * N, C)         # flatten patches
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)   # L2 normalize
        return x.reshape(B, N, -1)

class Interface_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None):
        super(Interface_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C/2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = x0 + x1
        return self.conv1x1(x0)

class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=nn.LayerNorm,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, add_cls_token=False, interface_dim=8):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        # norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_interface = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.add_cls_token = add_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.temporal_pos_embed_x = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.temporal_pos_embed_z = nn.Parameter(torch.randn(1, 1, embed_dim))

        interface_blocks = []
        block_nums = depth
        for i in range(block_nums):
            interface_blocks.append(Interface_block(inplanes=embed_dim, hide_channel=interface_dim))
        self.interface_blocks = nn.Sequential(*interface_blocks)
        self.interface_proj_head = ProjectionHead(in_dim=768, out_dim=256)
        interface_norms = []
        for i in range(block_nums):
            interface_norms.append(norm_layer(embed_dim))
        self.interface_norms = nn.Sequential(*interface_norms)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )
        self.blocks = nn.Sequential(*blocks)
        self.i = 0
        self.spliti = 9
        rgb_interface_blocks, dte_interface_blocks = [], []
        for i in range(depth - self.spliti):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            rgb_interface_blocks.append(CEBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, drop=drop_rate,
                                      attn_drop=attn_drop_rate, drop_path=dpr[i],
                                      norm_layer=norm_layer, act_layer=act_layer,
                                      keep_ratio_search=ce_keep_ratio_i))
            dte_interface_blocks.append(CEBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, drop=drop_rate,
                                      attn_drop=attn_drop_rate, drop_path=dpr[9],
                                      norm_layer=norm_layer, act_layer=act_layer,
                                      keep_ratio_search=ce_keep_ratio_i))
        self.rgb_interface_blocks = nn.Sequential(*rgb_interface_blocks)
        self.dte_interface_blocks = nn.Sequential(*dte_interface_blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)


    def enable_ir_lora(self):
        self.ir_lora_enabled = True
        for blk in list(self.blocks)[:self.spliti]:
            for m in blk.modules():
                if isinstance(m, LoRALinear):
                    m.unmerge_weights()

    def disable_ir_lora(self):
        self.ir_lora_enabled = False
        for blk in list(self.blocks)[:self.spliti]:
            for m in blk.modules():
                if isinstance(m, LoRALinear):
                    m.merge_weights()

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, track_query=None,
                         token_type="add", token_len=1, training=True,
                         cross_modality=False):
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
            B, T_x, C_x, H, W = x.shape
            X_RGB = x[:, :, :3, :, :].squeeze()
            X_DTE = x[:, :, 3:, :, :].squeeze()
            x = x.flatten(0, 1)
        else:
            B, H, W = x.shape[0], x.shape[2], x.shape[3]
            X_RGB = x[:, :3, :, :]
            X_DTE = x[:, 3:, :, :]

        num_template = len(z)
        # print('z:', num_template)
        z = torch.stack(z, dim=1)  # [(B, C, H, W),...] to ((B, T, C, H, W))
        Z_RGB = z[:, 0, :3, :, :].squeeze(1)
        Z_DTE = z[:, 0, 3:, :, :].squeeze(1)
        _, T_z, C_z, H_z, W_z = z.shape
        z = z.flatten(0, 1)        # (bs*T, C, H, W)
        # rgb image
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # multi-modal image
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]

        x_rgb = self.patch_embed(x_rgb)
        z_rgb = self.patch_embed(z_rgb)    # (bs*T, (H//16 * W//16), 768)

        z_dte = self.patch_embed_interface(z_dte)
        x_dte = self.patch_embed_interface(x_dte)

        # single modality branch
        Z_DTE = self.patch_embed_interface(Z_DTE)
        X_DTE = self.patch_embed_interface(X_DTE)
        Z_RGB = self.patch_embed(Z_RGB)
        X_RGB = self.patch_embed(X_RGB)

        Z_DTE = Z_DTE + self.pos_embed_z
        X_DTE = X_DTE + self.pos_embed_x
        Z_DTE += self.temporal_pos_embed_z
        X_DTE += self.temporal_pos_embed_x

        Z_RGB = Z_RGB + self.pos_embed_z
        X_RGB = X_RGB + self.pos_embed_x
        Z_RGB += self.temporal_pos_embed_z
        X_RGB += self.temporal_pos_embed_x

        z_rgb_feat = token2feature(self.interface_norms[0](z_rgb))
        x_rgb_feat = token2feature(self.interface_norms[0](x_rgb))
        z_dte_feat = token2feature(self.interface_norms[0](z_dte))
        x_dte_feat = token2feature(self.interface_norms[0](x_dte))

        z_feat = torch.cat([z_rgb_feat, z_dte_feat], dim=1)
        x_feat = torch.cat([x_rgb_feat, x_dte_feat], dim=1)
        z_feat = self.interface_blocks[0](z_feat)
        x_feat = self.interface_blocks[0](x_feat)
        z_dte = feature2token(z_feat)
        x_dte = feature2token(x_feat)
        x = x_rgb + x_dte
        z = z_rgb + z_dte
        z = z + self.pos_embed_z
        x = x + self.pos_embed_x

        z += self.temporal_pos_embed_z
        x += self.temporal_pos_embed_x
        x_dte = x_dte.reshape(-1, x_dte.size(1), x_dte.size(-1))
        z_dte = z_dte.reshape(-1, num_template * z_dte.size(1), z_dte.size(-1))
        z = z.reshape(-1, num_template * z.size(1), z.size(-1))
        x = x.reshape(-1, x.size(1), x.size(-1))

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            if token_type == "concat":
                if track_query is None:
                    query = self.cls_token.expand(B, token_len, -1)
                else:
                    track_len = track_query.size(1)
                    new_query = self.cls_token.expand(B, token_len - track_len, -1)
                    query = torch.cat([new_query, track_query], dim=1)
            elif token_type == "add":
                new_query = self.cls_token.expand(B, token_len, -1)  # copy B times
                query = new_query if track_query is None else track_query + new_query
            query = query + self.cls_pos_embed
        
        if self.add_sep_seg:
            x = x + self.search_segment_pos_embed
            z = z + self.template_segment_pos_embed

        if T_z > 1:
            z = z.view(B, T_z, -1, z.size()[-1]).contiguous() # (bs, T, HW, C)
            z = z.flatten(1, 2)  # (bs, THW, C)

        lens_z = z.shape[1]  # THW
        lens_x = x.shape[1]  # THW

        len_z_single = Z_RGB.shape[1]
        ZX_RGB = combine_tokens(Z_RGB, X_RGB, mode=self.cat_mode)
        ZX_DTE = combine_tokens(Z_DTE, X_DTE, mode=self.cat_mode)

        zx = combine_tokens(z, x, mode=self.cat_mode)  # (B, z+x, 768)
        if self.add_cls_token:
            zx = torch.cat([query, zx], dim=1)     # (B, 1+z+x, 768)
            ZX_RGB = torch.cat([query, ZX_RGB], dim=1)
            ZX_DTE = torch.cat([query, ZX_DTE], dim=1)
            query_len = query.size(1)
        zx = self.pos_drop(zx)
        ZX_RGB = self.pos_drop(ZX_RGB)
        ZX_DTE = self.pos_drop(ZX_DTE)
        
        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        
        removed_indexes_s = []
        loss_cm_list = []
        z_rgb_raw_list = []
        z_dte_raw_list = []

        X_RGB_list = []
        X_DTE_list = []
        Z_RGB_list = []
        Z_DTE_list = []
        for i in range(self.spliti):
            if self.add_cls_token:
                # self.disable_ir_lora()
                ZX_RGB, global_index_t, global_index_s, removed_index_s, attn = \
                    self.blocks[i](ZX_RGB, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token, query_len=query_len)
                # self.enable_ir_lora()
                ZX_DTE, global_index_t, global_index_s, removed_index_s, attn = \
                    self.blocks[i](ZX_DTE, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token, query_len=query_len)
            else:
                # self.disable_ir_lora()
                ZX_RGB, global_index_t, global_index_s, removed_index_s, attn = \
                    self.blocks[i](ZX_RGB, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token)
                # self.enable_ir_lora()
                ZX_DTE, global_index_t, global_index_s, removed_index_s, attn = \
                    self.blocks[i](ZX_DTE, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token, query_len=query_len)
            Z_RGB = ZX_RGB[:, 1:len_z_single+1, :]
            X_RGB = ZX_RGB[:, len_z_single+1:, :]
            Z_DTE = ZX_DTE[:, 1:len_z_single+1, :]
            X_DTE = ZX_DTE[:, len_z_single+1:, :]
            X_RGB_list.append(X_RGB)
            X_DTE_list.append(X_DTE)
            Z_RGB_list.append(Z_RGB)
            Z_DTE_list.append(Z_DTE)

        if training:
            for i in range(len(self.blocks) - self.spliti):
                if self.add_cls_token:
                    ZX_RGB, global_index_t, global_index_s, removed_index_s, attn = \
                        self.rgb_interface_blocks[i](ZX_RGB, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                            add_cls_token=self.add_cls_token, query_len=query_len)
                else:
                    ZX_RGB, global_index_t, global_index_s, removed_index_s, attn = \
                        self.rgb_interface_blocks[i](ZX_RGB, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                            add_cls_token=self.add_cls_token)
                if self.add_cls_token:
                    ZX_DTE, global_index_t, global_index_s, removed_index_s, attn = \
                        self.dte_interface_blocks[0](ZX_DTE, global_index_t, global_index_s, mask_x, ce_template_mask,
                                                     ce_keep_rate,
                                                     add_cls_token=self.add_cls_token, query_len=query_len)
                else:
                    ZX_DTE, global_index_t, global_index_s, removed_index_s, attn = \
                        self.dte_interface_blocks[0](ZX_DTE, global_index_t, global_index_s, mask_x, ce_template_mask,
                                                     ce_keep_rate,
                                                     add_cls_token=self.add_cls_token, query_len=query_len)
        else:
            ZX_RGB = None
            ZX_DTE = None
        # self.disable_ir_lora()
        for i, blk in enumerate(self.blocks):
            if i >= 1:
                if self.add_cls_token:
                    query = zx[:, :query_len, :]
                    zx = zx[:, query_len:, :]
                zx_ori = zx
                z = zx[:, :lens_z, :]
                x = zx[:, lens_z:, :]
                if i < self.spliti:
                    x = x + X_DTE_list[i]
                    if z.shape[1] == Z_DTE_list[i].shape[1]:
                        z = z + Z_DTE_list[i]
                    else:
                        z = z + Z_DTE_list[i].repeat(1, int(z.shape[1]/64), 1)
                x = x.reshape(x.size(0), -1, x.size(-1))
                z = z.reshape(num_template * z.size(0), -1, z.size(-1))
                x_dte = x_dte.reshape(x_dte.size(0), -1, x.size(-1))
                z_dte = z_dte.reshape(num_template * z_dte.size(0), -1, z.size(-1))
                x_rgb_feat = token2feature(self.interface_norms[i](x))
                z_rgb_feat = token2feature(self.interface_norms[i](z))
                x_dte_feat = token2feature(self.interface_norms[i](x_dte))
                z_dte_feat = token2feature(self.interface_norms[i](z_dte))
                z_feat = torch.cat([z_rgb_feat, z_dte_feat], dim=1)
                x_feat = torch.cat([x_rgb_feat, x_dte_feat], dim=1)
                z_feat = self.interface_blocks[i](z_feat)
                x_feat = self.interface_blocks[i](x_feat)
                z_dte = feature2token(z_feat)
                x_dte = feature2token(x_feat)
                x_dte = x_dte.reshape(-1,x_dte.size(1),x_dte.size(-1))
                z_dte = z_dte.reshape(-1,num_template * z_dte.size(1),z_dte.size(-1))
                zx_dte = combine_tokens(z_dte, x_dte, mode=self.cat_mode)
                assert zx_ori.shape[1] == zx_dte.shape[1]
                zx = zx_ori + zx_dte
                if self.add_cls_token:
                    zx = torch.cat([query, zx], dim=1)
            if self.add_cls_token:
                zx, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(zx, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token, query_len=query_len)
            else:
                zx, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(zx, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, add_cls_token=self.add_cls_token)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        # visualize_heatmap(x_rgb_ori, attn[:, :, query_len + lens_z:, query_len + lens_z:], H, W, self.i, title='Before')
        # self.i = self.i + 1
        # if len(loss_cm_list) > 0:
        #     z_rgb_raw = torch.stack(z_rgb_raw_list).mean()
        #     z_dte_raw = torch.stack(z_dte_raw_list).mean()
        #     loss_cm = torch.stack(loss_cm_list).mean()
        # else:
        #     z_rgb_raw = torch.tensor(0.0, device=z_rgb_feat.device)
        #     z_dte_raw = torch.tensor(0.0, device=z_rgb_feat.device)
        #     loss_cm = torch.tensor(0.0, device=z_rgb_feat.device)

        x = self.norm(zx)
        if training:
            ZX_RGB = self.norm(ZX_RGB)
            ZX_DTE = self.norm(ZX_DTE)
            X_RGB = ZX_RGB[:, len_z_single+1:, :]
            X_DTE = ZX_DTE[:, len_z_single+1:, :]
        # lens_x_new = global_index_s.shape[1]
        # lens_z_new = global_index_t.shape[1]
        #
        # if self.add_cls_token:
        #     query = x[:, :query_len]
        #     z = x[:, query_len:lens_z_new+query_len]
        #     x = x[:, lens_z_new+query_len:]
        # else:
        #     z = x[:, :lens_z_new]
        #     x = x[:, lens_z_new:]
        #
        # if removed_indexes_s and removed_indexes_s[0] is not None:
        #     removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
        #
        #     pruned_lens_x = lens_x - lens_x_new
        #     pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
        #     x = torch.cat([x, pad_x], dim=1)
        #     index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
        #     # recover original token order
        #     C = x.shape[-1]
        #     # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
        #     x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
        #
        # x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        #
        # if self.add_cls_token:
        #     x_cat = torch.cat([query, z, x], dim=1)
        # else:
        #     x_cat = torch.cat([z, x], dim=1)
        x_fused = x
        z_fused = z
        x_cat = x
        # aux_dict = {}
        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "x_fused": x_fused,
            "x_rgb_feat": X_RGB,
            "x_dte_feat": X_DTE,
            "z_fused": z_fused
        }

        return x_cat, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False, track_query=None, 
                token_type="add", token_len=1, training=True, cross_modality=False):
        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            track_query=track_query, token_type=token_type, token_len=token_len, training=training,
                                            cross_modality=cross_modality)
        return x, aux_dict

    def _replace_linear_with_lora(self, module, r=8, alpha=32):
        """
        Only replace:
           - MSA: qkv, proj
           - MLP: fc1, fc2
        """
        for name, child in list(module.named_children()):
            # Attention.qkv / Attention.proj
            if isinstance(child, nn.Linear) and (
                "qkv" in name or "proj" in name or
                "fc1" in name or "fc2" in name
            ):
                new = LoRALinear(
                    child.in_features, child.out_features,
                    r=r, alpha=alpha, bias=(child.bias is not None)
                )
                new.linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new.linear.bias.data.copy_(child.bias.data)
                setattr(module, name, new)
            else:
                self._replace_linear_with_lora(child, r=r, alpha=alpha)

    # def init_lora(self, r=8, alpha=32, num_shared_blocks=9):
    #     """
    #     Enable LoRA only on IR branch:
    #     - dte_interface_blocks
    #     - optionally first num_shared_blocks of backbone blocks
    #     """
    #     # IR interface blocks
    #     # self._replace_linear_with_lora(self.blocks, r=r, alpha=alpha)
    #     for blk in list(self.blocks)[:num_shared_blocks]:
    #         self._replace_linear_with_lora(blk, r=r, alpha=alpha)

    def get_lora_params(self):
        for m in self.modules():
            if isinstance(m, LoRALinear):
                if m.lora_A is not None:
                    yield m.lora_A
                    yield m.lora_B


    def merge_all_lora(self):
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.merge_weights()

    def unmerge_all_lora(self):
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.unmerge_weights()

def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            # do something here
            checkpoint_model = checkpoint['model']
            key_names = list(checkpoint_model.keys())
            for key_name in key_names:
                if 'decoder' in key_name:
                    del checkpoint_model[key_name]

            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
            print(missing_keys)
            print(unexpected_keys)
            print('Load pretrained model from: ' + pretrained)
    # if pretrained:
    #     if 'npz' in pretrained:
    #         model.load_pretrained(pretrained, prefix='')
    #     else:
    #         checkpoint = torch.load(pretrained, map_location="cpu")
    #         # do something here
    #         checkpoint_model = checkpoint['model']
    #
    #         new_pretrained_dict = {}
    #         for k, v in checkpoint_model.items():
    #             if k.startswith('backbone.model.'):
    #                 new_k = k.replace('backbone.model.', '')
    #             else:
    #                 new_k = k
    #             new_pretrained_dict[new_k] = v
    #         key_names = list(new_pretrained_dict.keys())
    #         for key_name in key_names:
    #             if 'decoder' in key_name:
    #                 del new_pretrained_dict[key_name]
    #
    #         missing_keys, unexpected_keys = model.load_state_dict(new_pretrained_dict, strict=False)
    #         print(missing_keys)
    #         print(unexpected_keys)
    #         print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_dropmae_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    # for blk in list(model.blocks)[:model.spliti]:
    #     model._replace_linear_with_lora(blk, r=8, alpha=32)
    return model

def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model