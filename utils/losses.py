import torch
import torch.nn as nn


def score_matrix(txt, img, loss_type):
    """S[i, j] = score between caption i and image j (higher = more similar)."""
    if loss_type == "order":
        # Vendrov et al.: image is the 'specific' item, caption the 'abstract' one (smaller coords).
        # Penalty = || max(0, f_c - f_i) ||^2 ; S = -penalty.
        viol = torch.clamp(txt.unsqueeze(1) - img.unsqueeze(0), min=0)   # (Q, N, D)
        return -(viol ** 2).sum(-1)
    # unit-norm embeddings: -||a-b||^2 = 2*cos - 2, i.e. the report's Eq.(1) up to a constant
    return txt @ img.t()


class ContrastiveLoss(nn.Module):
    """Eq.(2): sum over in-batch negatives of hinge losses in both directions (caption->image, image->caption)."""

    def __init__(self, margin=0.2, loss_type="cosine"):
        super().__init__()
        self.margin, self.loss_type = margin, loss_type

    def forward(self, txt, img):
        s = score_matrix(txt, img, self.loss_type)
        d = s.diag().view(-1, 1)
        cost_img = (self.margin - d + s).clamp(min=0)        # wrong images for caption i
        cost_txt = (self.margin - d.t() + s).clamp(min=0)    # wrong captions for image j
        mask = torch.eye(s.size(0), dtype=torch.bool, device=s.device)
        return (cost_img.masked_fill(mask, 0).sum() + cost_txt.masked_fill(mask, 0).sum()) / s.size(0)
