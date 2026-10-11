import torch
import torch.nn as nn
from torchvision import models


class VGG19Extractor(nn.Module):
    """Frozen VGG-19 -> fc7 (4096-d, after ReLU). Accepts (B,3,H,W) or ten-crop (B,10,3,H,W);
    ten-crop features are averaged into one vector per image (as in the report)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg19(weights=weights)
        self.features, self.avgpool = vgg.features, vgg.avgpool
        # classifier = [fc6, ReLU, Dropout, fc7, ReLU, Dropout, fc8] -> keep up to fc7+ReLU
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:5])
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        ten = x.dim() == 5
        if ten:
            b, n, c, h, w = x.shape
            x = x.reshape(-1, c, h, w)
        x = torch.flatten(self.avgpool(self.features(x)), 1)
        x = self.classifier(x)
        return x.reshape(b, n, -1).mean(1) if ten else x
