from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class FoundationEncoderSpec:
    model: object              # torch.nn.Module, in eval mode, no grad needed by caller
    input_size: int            # crops are resized to (input_size, input_size)
    mean: tuple                # per-channel normalization mean, e.g. ImageNet's (0.485, 0.456, 0.406)
    std: tuple                 # per-channel normalization std
    embedding_dim: int
    name: str


def _load_resnet18(device):
    import torch.nn as nn
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    except ImportError:
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=True)
    backbone.fc = nn.Identity()
    backbone.eval()
    return FoundationEncoderSpec(
        model=backbone.to(device), input_size=128,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        embedding_dim=512, name="resnet18",
    )


def _load_uni(device, checkpoint_path=None):
    import os
    import torch
    import timm

    if checkpoint_path is None:
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True,
        )
    else:
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5,
            num_classes=0, dynamic_img_size=True,
        )
        bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        elif os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        else:
            raise FileNotFoundError(
                f"Expected pytorch_model.bin or model.safetensors in {checkpoint_path!r} -- "
                f"did `git clone https://huggingface.co/MahmoodLab/UNI` (with git-lfs and "
                f"prior `huggingface-cli login`) actually pull the large weight file, not "
                f"just a pointer/LFS stub? Run `git lfs pull` inside that directory if unsure."
            )
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    return FoundationEncoderSpec(
        model=model.to(device), input_size=224,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        embedding_dim=1024, name="uni",
    )


def _load_ctranspath(device, checkpoint_path=None):
    import torch

    if checkpoint_path is None:
        raise ValueError(
            "_load_ctranspath needs --embedding-checkpoint pointing at a local "
            "torchscript_model.pt (download from huggingface.co/kaczmarj/CTransPath -- "
            "see this function's docstring)."
        )

    model = torch.jit.load(checkpoint_path, map_location=device)
    model.eval()
    return FoundationEncoderSpec(
        model=model, input_size=224,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        embedding_dim=768, name="ctranspath",
    )


_REGISTRY = {
    "resnet18": lambda device, checkpoint_path=None: _load_resnet18(device),
    "uni": lambda device, checkpoint_path=None: _load_uni(device, checkpoint_path=checkpoint_path),
    "ctranspath": _load_ctranspath,
}


def load_foundation_encoder(name, device=None, checkpoint_path=None):
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name not in _REGISTRY:
        raise ValueError(f"Unknown encoder {name!r}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](device, checkpoint_path=checkpoint_path)
