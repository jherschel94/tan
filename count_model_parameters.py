import argparse

import torch


def count_params(model):
    """Returns (total, trainable) parameter counts for any nn.Module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt(n):
    """Human-readable parameter count, e.g. 408,580 -> '408.6K'."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B ({n:,})"
    if n >= 1e6:
        return f"{n/1e6:.2f}M ({n:,})"
    if n >= 1e3:
        return f"{n/1e3:.1f}K ({n:,})"
    return f"{n:,}"


def main():
    parser = argparse.ArgumentParser(description="Count parameters of every node-feature encoder in this project")
    parser.add_argument("--num-classes", type=int, default=4,
                         help="Number of output classes -- affects the classifier head's size "
                              "slightly for TAN/ScratchCNN's own MIL pretraining heads, and the "
                              "downstream GAT's final layer. Default 4 (BACH); use 7 for BRACS.")
    parser.add_argument("--ctranspath-checkpoint", default=None,
                         help="Path to torchscript_model.pt -- required to count CTransPath's "
                              "parameters. Skipped with a note if not given.")
    parser.add_argument("--hidden-channels", type=int, default=64,
                         help="Same default as train_gnn_kfold.py -- for the GAT head's own "
                              "parameter count, which depends on its input dimension "
                              "(the encoder's embedding_dim) as well as this.")
    parser.add_argument("--num-layers", type=int, default=3)
    args = parser.parse_args()

    print(f"{'Model':<20} {'Total params':<22} {'Pretrained?':<12} {'Trained from scratch?'}")
    print("-" * 80)

    results = {}

    from tissue_aware_network import TissueAwareNetwork
    tan = TissueAwareNetwork(num_classes=args.num_classes)
    total, _ = count_params(tan)
    results["TAN"] = total
    print(f"{'TAN':<20} {fmt(total):<22} {'No':<12} {'Yes (per fold)'}")

    from scratch_cnn import ScratchCNN
    scratch = ScratchCNN(num_classes=args.num_classes)
    total, _ = count_params(scratch)
    results["ScratchCNN"] = total
    print(f"{'ScratchCNN':<20} {fmt(total):<22} {'No':<12} {'Yes (per fold)'}")

    from foundation_encoders import load_foundation_encoder
    resnet_spec = load_foundation_encoder(name="resnet18")
    total, _ = count_params(resnet_spec.model)
    results["ResNet18"] = total
    print(f"{'ResNet18':<20} {fmt(total):<22} {'Yes (ImageNet)':<12} {'No (frozen)'}")

    uni_spec = None
    try:
        uni_spec = load_foundation_encoder(name="uni")
        total, _ = count_params(uni_spec.model)
        results["UNI"] = total
        print(f"{'UNI':<20} {fmt(total):<22} {'Yes (histopath)':<12} {'No (frozen)'}")
    except Exception as e:
        print(f"{'UNI':<20} {'SKIPPED':<22} -- {type(e).__name__}: {e}")
        print(f"{'':20} (needs gated HuggingFace access -- see foundation_encoders.py)")

    ctranspath_spec = None
    if args.ctranspath_checkpoint:
        try:
            ctranspath_spec = load_foundation_encoder(
                name="ctranspath", checkpoint_path=args.ctranspath_checkpoint
            )
            total, _ = count_params(ctranspath_spec.model)
            results["CTransPath"] = total
            print(f"{'CTransPath':<20} {fmt(total):<22} {'Yes (TCGA+PAIP)':<12} {'No (frozen)'}")
        except Exception as e:
            print(f"{'CTransPath':<20} {'SKIPPED':<22} -- {type(e).__name__}: {e}")
    else:
        print(f"{'CTransPath':<20} {'SKIPPED':<22} -- no --ctranspath-checkpoint given")

    print()
    print("GAT head parameter count (varies slightly by encoder, due to input embedding_dim):")
    from train_gnn_kfold import SuperpixelGNN
    embedding_dims = {"TAN": tan.embedding_dim, "ScratchCNN": scratch.embedding_dim,
                       "ResNet18": resnet_spec.embedding_dim}
    if uni_spec is not None:
        embedding_dims["UNI"] = uni_spec.embedding_dim
    if ctranspath_spec is not None:
        embedding_dims["CTransPath"] = ctranspath_spec.embedding_dim
    for name, in_channels in embedding_dims.items():
        gat = SuperpixelGNN(in_channels, num_classes=args.num_classes,
                             hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                             conv_type="gat")
        total, _ = count_params(gat)
        print(f"  GAT head (fed by {name}, in_channels={in_channels}): {fmt(total)}")

    print()
    print("Summary -- backbone-only comparison (excludes the shared GAT head above):")
    if "ScratchCNN" in results and "TAN" in results:
        ratio = results["ScratchCNN"] / results["TAN"]
        print(f"  ScratchCNN has {ratio:.1f}x as many parameters as TAN "
              f"({fmt(results['ScratchCNN'])} vs {fmt(results['TAN'])})")
    if "ResNet18" in results and "TAN" in results:
        ratio = results["ResNet18"] / results["TAN"]
        print(f"  ResNet18 has {ratio:.1f}x as many parameters as TAN "
              f"({fmt(results['ResNet18'])} vs {fmt(results['TAN'])})")
    if "UNI" in results and "TAN" in results:
        ratio = results["UNI"] / results["TAN"]
        print(f"  UNI has {ratio:.1f}x as many parameters as TAN "
              f"({fmt(results['UNI'])} vs {fmt(results['TAN'])})")


if __name__ == "__main__":
    main()
