"""
Basic ESMTrack model.
"""
import copy
import math
import os
from typing import List
from ...utils.focal_loss import FocalLoss
import torch
import cv2
import numpy as np
from torchvision.ops import box_iou
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.models.layers.head import build_box_head
from lib.models.esmtrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.esmtrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.esmtrack.vit_dropmae import vit_base_dropmae_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.box_ops import giou_loss as giou_func
from lib.train.data.processing import TemplateProcessing
from lib.utils.ce_utils import generate_mask_cond, generate_attn_box_mask
import matplotlib.pyplot as plt
from lib.utils.nt_xent import NTXentLoss


def visualize_bbox(image, box, index=0):
    img = image[index].detach().cpu().permute(1, 2, 0).contiguous().numpy()
    img = img[:, :, :3]
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)
    box = box * 255
    cx, cy, w, h = box[index].detach().cpu().numpy()

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    # x1 = int(cx)
    # y1 = int(cy)
    # x2 = int(cx + w)
    # y2 = int(cy + h)

    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("bbox", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def agg_lang_feat(features, mask, pool_type="average"):
    """average pooling of language features
    """
    # feat: (bs, seq_len, C)
    # mask: (bs, seq_len)
    if pool_type == "average":
        # mask = mask.clamp(min=1e-6, max=1-1e-6)
        embedded = features * mask.unsqueeze(-1).float() # use mask to zero out invalid token features
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())
    elif pool_type == "max":
        out = []
        for i in range(len(features)):
            pool_feat, _ = torch.max(features[i][mask[i]], 0) # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0) # (bs, C)
    else:
        raise ValueError("pool_type should be average or max")
    return aggregate

def compute_apce(score_map: torch.Tensor, eps: float = 1e-6):
    """
    Compute APCE from response maps.

    Args:
        score_map: Tensor of shape [B, H, W] or [B, 1, H, W]
        eps: Small constant for numerical stability

    Returns:
        apce: Tensor of shape [B]
    """
    if score_map.dim() == 4:
        score_map = score_map.squeeze(1)  # [B, H, W]

    B = score_map.size(0)

    # flatten spatial dimensions
    score_flat = score_map.view(B, -1)  # [B, HW]

    r_max = score_flat.max(dim=1, keepdim=True)[0]  # [B, 1]
    r_min = score_flat.min(dim=1, keepdim=True)[0]  # [B, 1]

    numerator = (r_max - r_min).pow(2)  # [B, 1]

    denominator = (score_flat - r_min).pow(2).mean(dim=1, keepdim=True)  # [B, 1]

    apce = numerator / (denominator + eps)

    return apce.squeeze(1)  # [B]
    
# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.2):
#         super().__init__()
#         self.margin = margin
#
#     def forward(self, anchor, positive, negative):
#         d_ap = 1 - F.cosine_similarity(anchor, positive, dim=-1)
#         d_an = 1 - F.cosine_similarity(anchor, negative, dim=-1)
#         return F.relu(d_ap - d_an + self.margin).mean()

class TripletLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, anchor, positive, negative):

        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        pos = torch.sum(anchor * positive, dim=-1, keepdim=True)
        neg = torch.sum(anchor.unsqueeze(1) * negative, dim=-1)

        logits = torch.cat([pos, neg], dim=1) / self.tau
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)
    
    
class ESMTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", cfg=None, settings=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.track_query = None
        self.focal_loss = FocalLoss()
        self.cfg = cfg
        self.processing = TemplateProcessing(settings) if settings is not None else None
        
        # contrastive loss and KL loss
        self.cross_modality = True
        self.contrastive_loss = cfg.TRAIN.CONTRASTIVE_LOSS
        self.triplet_loss_fn = TripletLoss()
        self.lambda_triplet = 1.0
        self.device = self._get_device()
        if cfg.TRAIN.CONTRASTIVE_LOSS:
            self.NTXentLoss = NTXentLoss(device=self.device, batch_size=settings.batchsize,
                                         temperature=0.5, use_cosine_similarity=True)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print("\nRunning on:", device)

        # if device == 'cuda':
        #     device_name = torch.cuda.get_device_name()
            # print("The device name is:", device_name)
            # cap = torch.cuda.get_device_capability(device=None)
            # print("The capability of this device is:", cap, '\n')
        return device

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                grounding=None,
                grounding_masks=None,
                attn_box_mask_z=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                data=None,
                training=False
                ):
        if training:
            return self.train_forward(template=template, search=search, 
                                      grounding=grounding, grounding_masks=grounding_masks,
                                      attn_box_mask_z=attn_box_mask_z,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate, 
                                    return_last_attn=return_last_attn,
                                    data=data)
        else:
            return self.inference(template, search, 
                                  attn_box_mask_z=attn_box_mask_z,
                            ce_template_mask=ce_template_mask,
                            ce_keep_rate=ce_keep_rate, 
                            return_last_attn=return_last_attn,
                            data=data)
    
    def train_forward(self, template: torch.Tensor,
                search: torch.Tensor,
                grounding: torch.Tensor,
                grounding_masks=None,
                attn_box_mask_z=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                data=None
                ):
        assert isinstance(search, list), "The type of search is not List"
        bs = search[0].size(0)
        out_dict = []
        view_feats = []
        single_modality_feats = []

        # ===================== Forward Tracking =====================
        for i in range(len(search)):
            x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn,
                                        track_query=self.track_query,
                                        cross_modality=self.cross_modality
                                        # attn_box_mask_z=attn_box_mask_z
                                        )

            # ------- Cross modality loss during the Forward Tracking -------
            cm_loss = torch.tensor(0., dtype=torch.float).cuda()
            if self.cross_modality:
                x_rgb_feat = aux_dict["x_rgb_feat"]
                x_dte_feat = aux_dict["x_dte_feat"]
                rgb_relibility, ir_relibility = self.get_response(x, x_rgb_feat, x_dte_feat)
            cm_loss = cm_loss + self.cross_modality_loss(x_rgb_feat, x_dte_feat, rgb_relibility=rgb_relibility, ir_relibility=ir_relibility)

            enc_opt = x[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            enc_opt = self.backbone.interface_fusion(enc_opt, x_rgb_feat, x_dte_feat, rgb_relibility=rgb_relibility, ir_relibility=ir_relibility)
#            enc_opt = enc_opt + x_rgb_feat + x_dte_feat
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :1].clone()).detach() # stop grad  (B, N, C)
            
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = enc_opt * att
            
            out = self.forward_head(opt, None) #[cx, cy, w, h]
            out_dict.append(out)
            
            if self.contrastive_loss:  # For contrastive loss
                view_feats.append(enc_opt)
                
            # Crop search frame according to predict results
            search_range = data['search_range'][i]  # (T, B, 4)
            pred_boxes = out['pred_boxes'].view(-1, 4).detach()

            search_frames_list = []
            for b in range(bs):
                search_frames_list.append(data['search_frames_path'][b][i])
            self.processing(data, search_frames_list, pred_boxes, search_range, 
                            search_ori_anno=data['search_ori_anno'][i], search_anno=data['search_anno'][i])

        # ----------- Backward Tracking -----------
        if 'memory_images' in data.keys():
            memory_list = data['memory_images']
            memory_annos = torch.stack(data['memory_annos'], dim=0)
            
            memory_box_mask = []  # CE module
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                for i in range(len(memory_list)):
                    memory_box_mask.append(generate_mask_cond(self.cfg, memory_list[0].shape[0], memory_list[i].device, memory_annos[i]))
                memory_box_mask = torch.cat(memory_box_mask, dim=1)

            for i in range(len(grounding)):
                x, aux_dict = self.backbone(z=memory_list, x=grounding[i],
                                            ce_template_mask=memory_box_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, 
                                            track_query=self.track_query,
                                            cross_modality=self.cross_modality
                                            )
                if self.cross_modality:
                    x_rgb_feat = aux_dict["x_rgb_feat"]
                    single_modality_feats.append(x_rgb_feat)
                    x_dte_feat = aux_dict["x_dte_feat"]
                    single_modality_feats.append(x_dte_feat)

                enc_opt = x[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
                enc_opt = self.backbone.interface_fusion(enc_opt, x_rgb_feat, x_dte_feat, rgb_relibility=rgb_relibility, ir_relibility=ir_relibility)

                if self.backbone.add_cls_token:
                    self.track_query = (x[:, :1].clone()).detach() # stop grad  (B, N, C)
                    
                att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
                opt = enc_opt * att

                out = self.forward_head(opt, None)
                out_dict.append(out)

                if self.contrastive_loss:  # For contrastive loss
                    view_feats.append(enc_opt)

        contrastive_loss = torch.tensor(0., dtype=torch.float).cuda()
        rgb_ir_loss = torch.tensor(0., dtype=torch.float).cuda()
        triplet_loss = torch.tensor(0., dtype=torch.float).cuda()

        if self.contrastive_loss:
            anchors_list = []
            positives_list = []
            negatives_list = []

            def get_bias_box(boxes, bias_val=0.08):
                shifted_boxes = boxes.clone()
                noise = (torch.randn_like(boxes[:, :2]) * 2 - 1) * bias_val 
                shifted_boxes[:, :2] += noise.to(boxes.device)
                return shifted_boxes

            # ================= Grounding (Backward) =================
            for j in range(len(grounding)):
                # --- Anchor: GT  ---
                gt_mask = generate_attn_box_mask(self.cfg, bs, grounding[j].device, 
                                               data['grounding_anno'][j], box_type='search_feat').squeeze().flatten(1, 2)
                # --- Positive: predict ---
                pred_box_raw = out_dict[-len(grounding) + j]['pred_boxes'][:, 0]
                pred_mask = generate_attn_box_mask(self.cfg, bs, grounding[j].device, 
                                                 pred_box_raw, box_type='search_feat').squeeze().flatten(1, 2)
                # --- Negative: predict + Bias ---
                bias_box = get_bias_box(pred_box_raw, bias_val=0.08)
                bias_mask = generate_attn_box_mask(self.cfg, bs, grounding[j].device, 
                                                 bias_box, box_type='search_feat').squeeze().flatten(1, 2)

                feat_v = view_feats[-len(grounding) + j] # [B, HW, C]

                anchors_list.append(agg_lang_feat(feat_v, gt_mask, "average"))   # [B, C]
                positives_list.append(agg_lang_feat(feat_v, pred_mask, "average")) # [B, C]
                negatives_list.append(agg_lang_feat(feat_v, bias_mask, "average")) # [B, C]

            # ================= (Forward) =================
            num_forward = len(view_feats) - len(grounding)
            for k in range(num_forward):
                gt_anno_f = data['search_anno'][k] if 'search_anno' in data else data['grounding_anno'][0]
                
                f_gt_mask = generate_attn_box_mask(self.cfg, bs, view_feats[k].device, 
                                                 gt_anno_f, box_type='search_feat').squeeze().flatten(1, 2)
                
                f_pred_box = out_dict[k]['pred_boxes'][:, 0]
                f_pred_mask = generate_attn_box_mask(self.cfg, bs, view_feats[k].device, 
                                                   f_pred_box, box_type='search_feat').squeeze().flatten(1, 2)
                
                f_bias_box = get_bias_box(f_pred_box, bias_val=0.08)
                f_bias_mask = generate_attn_box_mask(self.cfg, bs, view_feats[k].device, 
                                                   f_bias_box, box_type='search_feat').squeeze().flatten(1, 2)

                feat_f = view_feats[k]
                
                anchors_list.append(agg_lang_feat(feat_f, f_gt_mask, "average"))
                positives_list.append(agg_lang_feat(feat_f, f_pred_mask, "average"))
                negatives_list.append(agg_lang_feat(feat_f, f_bias_mask, "average"))

            A = torch.cat(anchors_list, dim=0)
            P = torch.cat(positives_list, dim=0)
            N = torch.cat(negatives_list, dim=0)

            mask_valid = ~(torch.isnan(A).any(dim=1) | torch.isnan(P).any(dim=1) | torch.isnan(N).any(dim=1))
            if mask_valid.any():
                triplet_loss = self.triplet_loss_fn(A[mask_valid], P[mask_valid], N[mask_valid])
            else:
                triplet_loss = torch.tensor(0., device=A.device)

            box_mask_x = []
            for i in range(len(grounding)):
                box_mask_x.append(generate_attn_box_mask(
                    self.cfg, grounding[i].shape[0], grounding[i].device,
                    out_dict[-len(grounding)+i]['pred_boxes'][:, 0],
                    box_type='search_feat'
                ))
            # visualize_bbox(grounding[0], out_dict[-len(grounding)+0]['pred_boxes'][:, 0])

            # vis1 = out_dict[-len(grounding) + 0]['pred_boxes'][:, 0]
            # filter all 0 masks
            sampled_boxmask_x, sampled_view_feats = [[] for _ in range(len(grounding))], [[] for _ in range(len(grounding))]
            sampled_rgb_feats, sampled_ir_feats = [[] for _ in range(len(grounding))], [[] for _ in range(len(grounding))]

            for i in range(bs):
                if box_mask_x[0][i].sum() == 0 or box_mask_x[1][i].sum() == 0:
                    continue
                for j in range(len(grounding)):
                    sampled_boxmask_x[j].append(box_mask_x[j][i])
                    sampled_view_feats[j].append(view_feats[-len(grounding) + j][i])

                    sampled_rgb_feats[j].append(single_modality_feats[2 * j][i])  # j=0  rgb1,  j=1  rgb2
                    sampled_ir_feats[j].append(single_modality_feats[2 * j + 1][i])  # j=0  ir1,   j=1  ir2

            if len(sampled_boxmask_x[0]) > 0:
                grounding_view_feats = []
                grounding_rgb_feats = []
                grounding_ir_feats = []

                for j in range(len(grounding)):

                    boxmask = torch.cat(sampled_boxmask_x[j], dim=0).flatten(1, 2)

                    # view-level pooled features
                    view_f = torch.stack(sampled_view_feats[j], dim=0)
                    grounding_view_feats.append(
                        agg_lang_feat(view_f, boxmask, pool_type="average")
                    )

                    # ------RGB pooled feature------
                    rgb_f = torch.stack(sampled_rgb_feats[j], dim=0)
                    grounding_rgb_feats.append(
                        agg_lang_feat(rgb_f, boxmask, pool_type="average")
                    )

                    # ------IR pooled feature------
                    ir_f = torch.stack(sampled_ir_feats[j], dim=0)
                    grounding_ir_feats.append(
                        agg_lang_feat(ir_f, boxmask, pool_type="average")
                    )

                contrastive_loss = contrastive_loss + \
                    self.NTXentLoss(grounding_view_feats[0], grounding_view_feats[1])

                rgb_ir_loss = rgb_ir_loss + self.NTXentLoss(grounding_rgb_feats[0], grounding_ir_feats[1]) + \
                              self.NTXentLoss(grounding_rgb_feats[1], grounding_ir_feats[0])

        return out_dict, cm_loss, contrastive_loss, rgb_ir_loss, triplet_loss

    def get_response(self, x, x_rgb_feat, x_dte_feat, T=0.5):
        enc_opt = x[:, -self.feat_len_s:]
        # > 0.2 drop
        # enc_opt_no_drop = self.backbone.interface_fusion(enc_opt, x_rgb_feat, x_dte_feat, rgb_relibility=0, ir_relibility=0)
        # enc_opt_drop_rgb = self.backbone.interface_fusion(enc_opt, x_rgb_feat, x_dte_feat, rgb_relibility=1, ir_relibility=0)
        # enc_opt_drop_ir = self.backbone.interface_fusion(enc_opt, x_rgb_feat, x_dte_feat, rgb_relibility=0, ir_relibility=1)
        enc_opt_no_drop = enc_opt
        enc_opt_drop_rgb = x_dte_feat
        enc_opt_drop_ir = x_rgb_feat
        # enc_opt_drop_both = self.backbone.interface_fusion(enc_opt, x_rgb_feat, x_dte_feat, rgb_drop=0, ir_drop=0)
        if self.backbone.add_cls_token:
            self.track_query = (x[:, :1].clone()).detach()  # stop grad  (B, N, C)

        att = torch.matmul(enc_opt_no_drop, x[:, :1].transpose(1, 2))  # (B, HW, N)
        opt_no_drop = enc_opt_no_drop * att
        out_no_drop = self.forward_head(opt_no_drop, None)
        score_map_all = out_no_drop['score_map']
        psr_all = compute_apce(score_map_all)

        att = torch.matmul(enc_opt_drop_rgb, x[:, :1].transpose(1, 2))  # (B, HW, N)
        opt_drop_rgb = enc_opt_drop_rgb * att
        out_drop_rgb = self.forward_head(opt_drop_rgb, None)
        score_map_drop_rgb = out_drop_rgb['score_map']
        psr_drop_rgb = compute_apce(score_map_drop_rgb)

        att = torch.matmul(enc_opt_drop_ir, x[:, :1].transpose(1, 2))  # (B, HW, N)
        opt_drop_ir = enc_opt_drop_ir * att
        out_drop_ir = self.forward_head(opt_drop_ir, None)
        score_map_drop_ir = out_drop_ir['score_map']
        psr_drop_ir = compute_apce(score_map_drop_ir)

        delta_rgb = psr_all - psr_drop_rgb
        delta_ir = psr_all - psr_drop_ir

        w_rgb = torch.sigmoid(delta_rgb / T)
        w_ir = torch.sigmoid(delta_ir / T)
        # w_rgb = w_rgb.detach()
        # w_ir = w_ir.detach()
        # delta = torch.stack([delta_rgb, delta_ir], dim=1)
        # delta = torch.relu(delta)
        # w = torch.max(delta / T, dim=1)
        # w_rgb, w_ir = delta[:, 0], delta[:, 1]
        w_rgb = w_rgb.mean().detach()
        w_ir = w_ir.mean().detach()

        return w_rgb, w_ir

    def cross_modality_loss(self, rgb_proj, tir_proj, rgb_relibility=0.5, ir_relibility=0.5, temperature=0.07):
        # rgb_proj, tir_proj: [B, N, D]
        B, N, D = rgb_proj.shape
        rgb_flat = rgb_proj.reshape(B * N, D)
        tir_flat = tir_proj.reshape(B * N, D)

        logits = rgb_flat @ tir_flat.T / temperature  # [BN, BN]
        labels = torch.arange(B * N, device=rgb_proj.device)

        # InfoNCE loss from both sides (symmetrical)
        loss_rgb_to_tir = F.cross_entropy(logits, labels)
        loss_tir_to_rgb = F.cross_entropy(logits.T, labels)
        return rgb_relibility * loss_rgb_to_tir + ir_relibility * loss_tir_to_rgb

    def inference(self, template: torch.Tensor,
                search: torch.Tensor,
                attn_box_mask_z=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                data=None
                ):
        out_dict = []
        for i in range(len(search)):
            x, aux_dict = self.backbone(z=template, x=search[i],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn,
                                        track_query=self.track_query,
                                        training=False
                                        )
            enc_opt = x[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            # x_rgb_feat = aux_dict["x_rgb_feat"]
            # x_dte_feat = aux_dict["x_dte_feat"]
            # rgb_relibility, ir_relibility = self.get_response(x, x_rgb_feat, x_dte_feat)
            # # no drop when inference
            # enc_opt = self.backbone.interface_fusion(enc_opt, x_rgb_feat, x_dte_feat, rgb_relibility, ir_relibility)

            if self.backbone.add_cls_token:
                self.track_query = (x[:, :1].clone()).detach() # stop grad  (B, N, C)
                
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = enc_opt * att
            
            out = self.forward_head(opt, None)
            out_dict.append(out)
        
        return out_dict
    def compute_iou(self, pred_dict, gt_dict, i):
        """
        Compute batch-wise IoU for grounding frames.
        Return: iou: (B,)
        """
        # assert isinstance(pred_dict, list)

        # get GT for this grounding frame
        gt_bbox = gt_dict['grounding_anno'][i]  # (B,4) xywh
        # get pred boxes
        pred_boxes = pred_dict  # assume shape (B,4) cxcywh
        pred_boxes = torch.tensor(pred_boxes, device=gt_bbox.device).float()

        # convert formats
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)      # (B,4)
        gt_boxes_xyxy   = box_xywh_to_xyxy(gt_bbox).clamp(0,1)  # (B,4)

        # compute pairwise IoU matrix
        iou_matrix = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)  # (B,B)
        # pick diagonal -> batch-wise IoU
        indices = torch.arange(pred_boxes_xyxy.size(0), device=pred_boxes_xyxy.device)
        iou = iou_matrix[indices, indices]  # (B,)

        return iou
    # def compute_iou(self, pred_dict, gt_dict):
    #     # currently only support the type of pred_dict is list
    #     assert isinstance(pred_dict, list)
    #     loss_dict = {}
    #
    #     grounding_num = gt_dict['grounding_anno'].size(0)
    #     grounding_ids = list(range(0, len(pred_dict)))[-grounding_num:]
    #     search_num = len(pred_dict) - grounding_num
    #
    #     for i in range(len(pred_dict)):
    #         if i in grounding_ids:  # For grounding frames: calc box loss
    #             # get GT
    #             gt_bbox = gt_dict['grounding_anno'][i - search_num]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
    #
    #             # Get pred boxes
    #             # pred_boxes = pred_dict[i]['pred_boxes']
    #             pred_boxes = torch.asarray(pred_dict)
    #             # if torch.isnan(pred_boxes).any():
    #             #     raise ValueError("Network outputs is NAN! Stop Training")
    #             num_queries = 1 #pred_boxes.size(1)
    #             pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4).cuda()  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
    #             gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(
    #                 min=0.0, max=1.0)
    #             # gt_boxes_vec = box_xywh_to_xyxy(gt_bbox[i]).clamp(
    #             #     min=0.0, max=1.0)
    #             # (B,4) --> (B,1,4) --> (B,N,4)
    #
    #             # compute giou and iou
    #             try:
    #                 giou_loss, iou = giou_func(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    #             except:
    #                 giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
    #             loss_dict['giou'] = giou_loss
    #     return iou
    
        
    def forward_head(self, enc_opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError

def contrastive_loss(rgb_proj, tir_proj, temperature=0.07):
    # rgb_proj, tir_proj: [B, N, D]
    B, N, D = rgb_proj.shape
    rgb_flat = rgb_proj.view(B * N, D)
    tir_flat = tir_proj.view(B * N, D)

    logits = rgb_flat @ tir_flat.T / temperature     # [BN, BN]
    labels = torch.arange(B * N, device=rgb_proj.device)

    # InfoNCE loss from both sides (symmetrical)
    loss_rgb_to_tir = F.cross_entropy(logits, labels)
    loss_tir_to_rgb = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_rgb_to_tir + loss_tir_to_rgb)

def build_esmtrack(cfg, training=True, settings=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                         )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_dropmae_ce':
        backbone = vit_base_dropmae_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        )
        backbone.build_interface_blocks_from_trunk()
    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = ESMTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        cfg=cfg,
        settings=settings,
    )

    return model
