import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from torch import Tensor


class DescriptorMemory:
    """Small JSON-backed descriptor memory.

    The dynamic LLM update loop writes the same JSON structure back later; this
    class only keeps training-time lookup and statistics compact.
    """

    def __init__(self, descriptor_path: Optional[str] = None) -> None:
        self.descriptors: Dict[str, List[str]] = {}
        self.usage: Dict[str, Dict[str, int]] = {}
        self.confusion: Dict[str, Dict[str, int]] = {}
        if descriptor_path:
            self.load(descriptor_path)

    def load(self, descriptor_path: str) -> None:
        path = Path(descriptor_path).expanduser()
        with path.open('r', encoding='utf-8') as f:
            raw = json.load(f)
        for class_name, entries in raw.items():
            texts = []
            for entry in entries:
                if isinstance(entry, str):
                    texts.append(entry)
                elif isinstance(entry, dict) and entry.get('active', True):
                    text = entry.get('text')
                    if text:
                        texts.append(text)
            if texts:
                self.descriptors[class_name] = texts

    def get(self, class_name: str) -> List[str]:
        return self.descriptors.get(class_name, [])

    def record(self, class_name: str, descriptor_texts: Sequence[str],
               predicted_class: Optional[str] = None) -> None:
        class_usage = self.usage.setdefault(class_name, {})
        for text in descriptor_texts:
            class_usage[text] = class_usage.get(text, 0) + 1
        if predicted_class and predicted_class != class_name:
            class_confusion = self.confusion.setdefault(class_name, {})
            class_confusion[predicted_class] = (
                class_confusion.get(predicted_class, 0) + 1)


class KAGEBranch(nn.Module):
    def __init__(self,
                 descriptor_path: Optional[str] = None,
                 embed_dims: int = 256,
                 roi_output_size: int = 7,
                 expand_scale: float = 1.5,
                 topk: int = 3,
                 loss_weight: float = 1.0,
                 score_beta: float = 0.2) -> None:
        super().__init__()
        self.memory = DescriptorMemory(descriptor_path)
        self.roi_output_size = roi_output_size
        self.expand_scale = expand_scale
        self.topk = topk
        self.loss_weight = loss_weight
        self.score_beta = score_beta

        self.roi_proj = nn.Linear(embed_dims, embed_dims)
        self.meta_net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        self.out_norm = nn.LayerNorm(embed_dims)

    def forward_loss(self, visual_feats: Sequence[Tensor],
                     matches: List[dict],
                     descriptor_embeddings: List[Dict[str, Tensor]],
                     text_prompts: Sequence[Sequence[str]]) -> Dict[str, Tensor]:
        device = visual_feats[0].device
        losses = []
        for img_id, match in enumerate(matches):
            pos_inds = match['pos_inds']
            if pos_inds.numel() == 0:
                continue

            query_feat = match['query_feat']
            boxes = match['bbox_pred']
            labels = match['labels']
            img_meta = match['img_meta']
            class_names = list(text_prompts[img_id])
            desc_embeds = descriptor_embeddings[img_id]
            if not desc_embeds:
                continue

            roi_feat = self._pool_boxes(visual_feats, img_id, boxes, img_meta)
            ctx_boxes = self._expand_boxes(boxes, img_meta)
            ctx_feat = self._pool_boxes(visual_feats, img_id, ctx_boxes,
                                        img_meta)
            fused = self.out_norm(query_feat + self.roi_proj(roi_feat) +
                                  self.meta_net(ctx_feat))

            scored = self._score_classes(fused, class_names, desc_embeds)
            if scored is None:
                continue
            scores, selected_desc = scored
            valid = labels < scores.size(1)
            if valid.any():
                loss = F.cross_entropy(scores[valid], labels[valid])
                losses.append(loss)
                selected_valid = {
                    name: inds[valid]
                    for name, inds in selected_desc.items()
                }
                self._record_stats(scores[valid], labels[valid], class_names,
                                   selected_valid)

        if losses:
            loss_desc = torch.stack(losses).mean() * self.loss_weight
        else:
            loss_desc = visual_feats[0].sum() * 0.
        return {'loss_kage_desc': loss_desc}

    def predict_scores(self, visual_feats: Sequence[Tensor],
                       query_infos: List[dict],
                       descriptor_embeddings: List[Dict[str, Tensor]],
                       text_prompts: Sequence[Sequence[str]]) -> List[Tensor]:
        score_maps = []
        for img_id, query_info in enumerate(query_infos):
            query_feat = query_info['query_feat']
            boxes = query_info['bbox_pred']
            img_meta = query_info['img_meta']
            class_names = list(text_prompts[img_id])
            desc_embeds = descriptor_embeddings[img_id]
            if not desc_embeds:
                score_maps.append(
                    query_feat.new_zeros((query_feat.size(0),
                                          len(class_names))))
                continue

            roi_feat = self._pool_boxes(visual_feats, img_id, boxes, img_meta)
            ctx_boxes = self._expand_boxes(boxes, img_meta)
            ctx_feat = self._pool_boxes(visual_feats, img_id, ctx_boxes,
                                        img_meta)
            fused = self.out_norm(query_feat + self.roi_proj(roi_feat) +
                                  self.meta_net(ctx_feat))
            scored = self._score_classes(fused, class_names, desc_embeds)
            if scored is None:
                score_maps.append(
                    query_feat.new_zeros((query_feat.size(0),
                                          len(class_names))))
            else:
                scores, _ = scored
                score_maps.append(scores.sigmoid())
        return score_maps

    def _score_classes(self, fused: Tensor, class_names: List[str],
                       descriptor_embeddings: Dict[str, Tensor]
                       ) -> Optional[Tuple[Tensor, Dict[str, Tensor]]]:
        class_scores = []
        selected_desc = {}
        for class_name in class_names:
            desc = descriptor_embeddings.get(class_name)
            if desc is None or desc.numel() == 0:
                class_scores.append(fused.new_full((fused.size(0), ), -1e4))
                continue
            sim = F.normalize(fused, dim=-1) @ F.normalize(desc, dim=-1).t()
            k = min(self.topk, sim.size(1))
            top = sim.topk(k, dim=1)
            selected_desc[class_name] = top.indices
            class_scores.append(top.values.mean(dim=1))
        if not class_scores:
            return None
        return torch.stack(class_scores, dim=1), selected_desc

    def _record_stats(self, scores: Tensor, labels: Tensor,
                      class_names: List[str],
                      selected_desc: Dict[str, Tensor]) -> None:
        pred_labels = scores.argmax(dim=1)
        for row_idx, (target, pred) in enumerate(
                zip(labels.tolist(), pred_labels.tolist())):
            if target >= len(class_names):
                continue
            class_name = class_names[target]
            desc_texts = self.memory.get(class_name)
            selected_inds = selected_desc.get(class_name)
            if selected_inds is not None:
                desc_texts = [
                    desc_texts[i] for i in selected_inds[row_idx].tolist()
                    if i < len(desc_texts)
                ]
            pred_name = class_names[pred] if pred < len(class_names) else None
            self.memory.record(class_name, desc_texts, pred_name)

    def _pool_boxes(self, visual_feats: Sequence[Tensor], img_id: int,
                    boxes: Tensor, img_meta: dict) -> Tensor:
        feat = visual_feats[0][img_id:img_id + 1]
        _, _, feat_h, feat_w = feat.shape
        img_h, img_w = img_meta['img_shape'][:2]
        scale = boxes.new_tensor([feat_w / img_w, feat_h / img_h,
                                  feat_w / img_w, feat_h / img_h])
        scaled = boxes * scale
        pooled = []
        for box in scaled:
            x1, y1, x2, y2 = box.round().long()
            x1 = int(x1.clamp(0, feat_w - 1).item())
            x2 = int(x2.clamp(0, feat_w - 1).item())
            y1 = int(y1.clamp(0, feat_h - 1).item())
            y2 = int(y2.clamp(0, feat_h - 1).item())
            if x2 <= x1:
                x2 = min(x1 + 1, feat_w)
            if y2 <= y1:
                y2 = min(y1 + 1, feat_h)
            crop = feat[:, :, y1:y2, x1:x2]
            crop = F.adaptive_avg_pool2d(
                crop, (self.roi_output_size, self.roi_output_size))
            pooled.append(crop.mean(dim=(-1, -2)).squeeze(0))
        if not pooled:
            return feat.new_zeros((0, feat.size(1)))
        return torch.stack(pooled, dim=0)

    def _expand_boxes(self, boxes: Tensor, img_meta: dict) -> Tensor:
        img_h, img_w = img_meta['img_shape'][:2]
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        w = (boxes[:, 2] - boxes[:, 0]) * self.expand_scale
        h = (boxes[:, 3] - boxes[:, 1]) * self.expand_scale
        expanded = torch.stack(
            [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5],
            dim=1)
        expanded[:, 0::2].clamp_(min=0, max=img_w)
        expanded[:, 1::2].clamp_(min=0, max=img_h)
        return expanded


def denormalize_cxcywh_boxes(boxes: Tensor, img_meta: dict) -> Tensor:
    img_h, img_w = img_meta['img_shape'][:2]
    factor = boxes.new_tensor([img_w, img_h, img_w, img_h])
    return bbox_cxcywh_to_xyxy(boxes) * factor
