from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn

import torch.nn.functional as F

class ContrastiveLossSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __init__(self, metric, margin, scale):
        super(ContrastiveLossSmoother, self).__init__()
        self.margin = margin
        self.metric = metric
        self.scale = scale

    def _compute_contrastive_loss(self, x0: List[torch.Tensor], x1: List[torch.Tensor], y):
        # euclidian distance
        if self.metric == 'l2':
            # diff = x0 - x1
            # dist_sq = torch.sum(torch.pow(diff, 2), -1) / x0.shape[-1]
            # if torch.any(torch.isnan(dist_sq)):
            #     print("nan error")
            # dist = torch.sqrt(dist_sq)
            # if torch.any(torch.isnan(dist)):
            #     print("nan error")
            dist = F.pairwise_distance(x0, x1, keepdim=True)
        elif self.metric == 'cos':
            dist = 1 - torch.cosine_similarity(x0, x1, dim=-1)
        else:
            raise NotImplementedError(f"Metric {self.metric} not implemented.")

        if dist.dim() == 1:
            loss = y * torch.pow(dist, 2) 
            loss += (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        elif dist.dim() == 2:
            _ , seq_len = dist.shape
            # loss = y.unsqueeze(-1).expand(-1, seq_len) * dist_sq + (1 - y).unsqueeze(-1).expand(-1, seq_len) * torch.pow(dist, 2)
        else:
            raise KeyError
        
        if torch.any(torch.isnan(loss)):
            raise ValueError("nan error")
        
        # loss = torch.sum(loss, dim=-1) / 2.0 / x0.size()[0]
        # if torch.any(torch.isnan(loss)):
        #     raise ValueError("nan error")
        
        return loss

    def _compute_label_smoothing(self, logits, labels):
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)

        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)

        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


    def __call__(self, model_output, labels, **kwargs):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        # Compute Contrastive Loss
        mask = (labels != -100)
        target = labels[mask].reshape(mask.shape[0], -1)

        feat_x = model_output['hidden_states'][-1] #b,seq_len,h
        feat_x = feat_x[..., :-1, :].contiguous()
        feat_x = feat_x[mask]
        feat_x = feat_x[::target.shape[0],:]

        if torch.any(torch.isnan(feat_x)):
            print("nan error")        
        
        index = torch.randperm(feat_x.shape[0]).to(feat_x.device)
        agreement = (target[:, 0] == target[index, 0]).float().to(feat_x.device)
        contrast_loss = self._compute_contrastive_loss(feat_x, feat_x[index, :], agreement)
        contrast_loss = self.scale * torch.mean(contrast_loss)

        # Compute Smooth Loss
        smooth_loss = self._compute_label_smoothing(logits, labels)

        return contrast_loss + smooth_loss