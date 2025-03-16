import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image

MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation='ReLU', batch_norm = True, dropout=0.2, num_classes=1):
        super().__init__()
        self.activation = getattr(nn, activation)()
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        layers = []
        if hidden_dims:
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                if self.batch_norm:
                    layers.append(nn.BatchNorm1d(h_dim))
                layers.append(self.activation)
                layers.append(self.dropout)
                prev_dim = h_dim
            layers.append(nn.Linear(hidden_dims[-1], num_classes))
        else:
            layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.classifier(x)
        return logits

@register_model("ViTClassifier")
class ViTClassifier(nn.Module):
    def __init__(self, backbone_name = 'vit_base_patch16_224', hidden_dims=[], activation='ReLU', batch_norm = True, dropout=0.2, num_classes=1, freeze_backbone=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        embed_dim = self.backbone.num_features

        # Freeze backbone parameters if specified.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
        self.classifier = LinearClassifier(embed_dim, hidden_dims, activation, batch_norm, dropout, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

@register_model("ResNetClassifier")
class ResNetClassifier(nn.Module):
    def __init__(self, backbone_name='resnet50',  hidden_dims=[], activation='ReLU', batch_norm = True, dropout=0.2, num_classes=1, freeze_backbone=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        embed_dim = self.backbone.num_features

        # Freeze backbone parameters if specified.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = LinearClassifier(embed_dim, hidden_dims, activation, batch_norm, dropout, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
if __name__=="__main__":
    model = ViTClassifier(hidden_dims=[32])
    print(model)
