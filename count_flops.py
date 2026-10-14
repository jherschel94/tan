import argparse
import pickle
import sys

import numpy as np
import torch

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:
    sys.exit("This script needs torch >= 2.1 (torch.utils.flop_counter).")


def count_flops(fn):
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            fn()
    return fc.get_total_flops()


def tan_embed(model, x, mask):
    x, mask = model.tan1(x, mask); x, mask = model.pool1(x, mask)
    x, mask = model.tan2(x, mask); x, mask = model.pool2(x, mask)
    x, mask = model.tan3(x, mask); x, mask = model.pool3(x, mask)
    x, mask = model.tan4(x, mask); x, mask = model.pool4(x, mask)
    return model.attn(x, mask)


def fmt(flops):
    return f"{flops / 1e9:.3f}"


def main():
    p = argparse.ArgumentParser(description="FLOP counts per superpixel and per image for every encoder")
    p.add_argument("dataset", help=".pkl built with --tan-crops (needs tan_crops/tan_masks)")
    p.add_argument("--n-crops", type=int, default=500, help="Number of random superpixels to sample")
    p.add_argument("--num-classes", type=int, default=4, help="4 for BACH, 7 for BRACS")
    p.add_argument("--scratch-crop-size", type=int, default=128,
                   help="Fixed crop of ResNet18-scratch (128 BACH, 64 BRACS)")
    p.add_argument("--ctranspath-checkpoint", default=None)
    p.add_argument("--skip-uni", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cpu")  # deterministic, and does not disturb running GPU jobs
    rng = np.random.default_rng(args.seed)

    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    samples = data["samples"]
    if "tan_crops" not in samples[0]:
        sys.exit("This dataset has no tan_crops -- build it with --tan-crops.")
    mean_nodes = float(np.mean([s["n_nodes"] for s in samples]))
    print(f"{len(samples)} images, mean superpixels per image: {mean_nodes:.0f}")

    pairs = []
    for _ in range(args.n_crops):
        i = int(rng.integers(len(samples)))
        j = int(rng.integers(len(samples[i]["tan_crops"])))
        pairs.append((i, j))

    from tissue_aware_network import TissueAwareNetwork
    from scratch_cnn import ScratchCNN
    tan = TissueAwareNetwork(num_classes=args.num_classes).to(device).eval()
    resnet_scratch = ScratchCNN(num_classes=args.num_classes).to(device).eval()

    results = []  # (name, input description, flops per superpixel mean, std)

    from torchvision.models import resnet18
    ref = resnet18(weights=None).eval()
    ref_flops = count_flops(lambda: ref(torch.zeros(1, 3, 224, 224)))
    print(f"[sanity] torchvision ResNet18 @224: {ref_flops/1e9:.2f} GFLOPs = "
          f"{ref_flops/2e9:.2f} GMACs (expected about 1.82 GMACs)\n")

    tan_f, ada_f, hw, skipped = [], [], [], 0
    for i, j in pairs:
        crop = samples[i]["tan_crops"][j]
        mask = samples[i]["tan_masks"][j]
        x = torch.from_numpy(crop.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
        m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        try:
            tan_f.append(count_flops(lambda: tan_embed(tan, x, m)))
            ada_f.append(count_flops(lambda: resnet_scratch.backbone(x)))
            hw.append(crop.shape[0] * crop.shape[1])
        except Exception:
            skipped += 1
    if skipped:
        print(f"[note] {skipped} crops skipped (too small for a forward pass)")
    print(f"sampled crops: {len(tan_f)}, median bounding box area: {int(np.median(hw))} px, "
          f"mean valid fraction: "
          f"{np.mean([samples[i]['tan_masks'][j].mean() for i, j in pairs]):.2f}")
    results.append(("TAN (embedding only)", "native, masked", np.mean(tan_f), np.std(tan_f)))
    results.append(("AdaptiveCNN (ResNet18)", "native bbox", np.mean(ada_f), np.std(ada_f)))

    s = args.scratch_crop_size
    xs = torch.zeros(1, 3, s, s)
    results.append(("ResNet18-scratch", f"{s}x{s}", count_flops(lambda: resnet_scratch.backbone(xs)), 0.0))

    from foundation_encoders import load_foundation_encoder
    names = [("resnet18", "ResNet18-ImageNet", None)]
    if not args.skip_uni:
        names.append(("uni", "UNI", None))
    if args.ctranspath_checkpoint:
        names.append(("ctranspath", "CTransPath", args.ctranspath_checkpoint))
    for key, label, ckpt in names:
        try:
            spec = load_foundation_encoder(name=key, device=device, checkpoint_path=ckpt)
            xin = torch.zeros(1, 3, spec.input_size, spec.input_size)
            fl = count_flops(lambda: spec.model(xin))
            results.append((label, f"{spec.input_size}x{spec.input_size}", fl, 0.0))
        except Exception as e:
            print(f"[skipped] {label}: {type(e).__name__}: {e}")

    tan_mean = results[0][2]
    print(f"\n{'Encoder':<26}{'input':<16}{'GFLOPs/superpixel':<22}{'GMACs/sp':<12}"
          f"{'GFLOPs/image':<15}{'x TAN'}")
    print("-" * 100)
    for name, inp, mean_f, std_f in results:
        per_sp = fmt(mean_f) + (f" +-{fmt(std_f)}" if std_f > 0 else "")
        print(f"{name:<26}{inp:<16}{per_sp:<22}{fmt(mean_f / 2):<12}"
              f"{mean_f * mean_nodes / 1e9:<15.1f}{mean_f / tan_mean:.1f}x")
    print("\nFLOPs = 2 x MACs; only matmul/conv/attention ops are counted; the GAT is excluded.")
    print("TAN's count covers valid (inside-mask) positions only; AdaptiveCNN processes the whole bounding box.")


if __name__ == "__main__":
    main()
