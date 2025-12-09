# model_swin_se.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SEExcitation(nn.Module):
    """
    Excitation-only SE block applied to a (B, C) class-token vector.
    """
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        w = self.fc1(x)
        w = F.relu(w, inplace=True)
        w = self.fc2(w)
        w = torch.sigmoid(w)
        return x * w


class SwinSEFER(nn.Module):
    """
    Swin Transformer backbone + SE on global feature + linear classifier.

    Mirrors the paper: SwinT pretrained on ImageNet-1K, input 224x224x3,
    SE applied to the class/global feature, then final FC for FER classes. [file:1]
    """
    def __init__(
        self,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 7,
        reduction: int = 16,
        pretrained: bool = True,
    ):
        super().__init__()
        # num_classes=0 -> timm returns global-pooled features
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.se = SEExcitation(feat_dim, reduction=reduction)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone returns (B, C) for num_classes=0
        feats = self.backbone(x)
        feats = self.se(feats)
        logits = self.fc(feats)
        return logits
