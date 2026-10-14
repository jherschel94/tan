import torch.nn as nn


class ScratchCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)  # random init, NOT pretrained
        self.embedding_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        emb = self.backbone(x)
        return self.classifier(emb), emb
