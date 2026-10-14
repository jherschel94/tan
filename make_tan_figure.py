import argparse
import os
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE_CH = [32, 64, 128, 128]


def _prepare_panel(arr, info, valid_min=1):
    ma = np.ma.masked_where(~info, arr)
    vals = arr[info]
    if vals.size < valid_min:
        return ma, 0.0, 1.0
    lo, hi = np.percentile(vals, [1, 99])
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    return ma, lo, hi


def plot_figure(crop, mask, stages, out_stem, width_in=7.0):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "pdf.fonttype": 42, "font.size": 8,
    })
    H, W = mask.shape
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((0, 0, 0, 0))          # transparent where not informative
    n_cols = 1 + len(stages)
    fig_h = width_in / n_cols * 2 * H / W + 0.5
    fig, axes = plt.subplots(2, n_cols, figsize=(width_in, fig_h))
    extent = (0, W, H, 0)

    axes[0, 0].imshow(crop, extent=extent)
    axes[0, 0].contour(np.arange(W) + 0.5, np.arange(H) + 0.5, mask.astype(float),
                       levels=[0.5], colors="yellow", linewidths=0.8)
    axes[0, 0].set_title("Input crop", fontsize=8)
    axes[1, 0].imshow(mask, cmap="gray", extent=extent, vmin=0, vmax=1, interpolation="nearest")
    axes[1, 0].set_title("Mask", fontsize=8)

    for s, st in enumerate(stages):
        feat, m = st["feat"], st["mask"]
        valid = m > 0.5
        info = st.get("info", valid) & valid
        base = np.full(valid.shape, 0.80)           # outside the mask
        base[valid] = 0.93                           # valid, but edge band (bias only)
        mean_map = feat.mean(axis=0)
        if info.any():
            var = np.array([feat[c][info].var() for c in range(feat.shape[0])])
            top = int(np.argmax(var))
        else:
            top = 0
        for r, arr in enumerate([mean_map, feat[top]]):
            ax = axes[r, s + 1]
            ax.imshow(base, cmap="gray", vmin=0, vmax=1, extent=extent, interpolation="nearest")
            ma, lo, hi = _prepare_panel(arr, info)
            ax.imshow(ma, cmap=cmap, vmin=lo, vmax=hi, extent=extent, interpolation="nearest")
        axes[0, s + 1].set_title(f"Stage {s + 1} ({feat.shape[0]} ch.)", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    short = fig_h < 2.3     # a long rotated label would run over the panel
    fig.text(0.005, 0.72, "mean" if short else "channel mean", rotation=90, va="center", fontsize=8)
    fig.text(0.005, 0.28, "top ch." if short else "strongest channel", rotation=90, va="center", fontsize=8)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    fig.savefig(out_stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_tan(checkpoint, num_classes):
    import torch
    from tissue_aware_network import TissueAwareNetwork
    model = TissueAwareNetwork(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()


def to_tensors(crop, mask):
    import torch
    x = torch.from_numpy(crop.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
    m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    return x, m


def _pool2(a):
    h, w = a.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    return a[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).max(axis=(1, 3))


def run_stagewise(model, crop, mask, show_band=False):
    import torch
    from scipy.ndimage import binary_erosion
    x, m = to_tensors(crop, mask)
    stages = []
    with torch.no_grad():
        for conv_name, pool_name in [("tan1", "pool1"), ("tan2", "pool2"), ("tan3", "pool3"), ("tan4", "pool4")]:
            m_in = m[0].cpu().numpy() > 0.5
            x, m = getattr(model, conv_name)(x, m)
            x, m = getattr(model, pool_name)(x, m)
            st = {"feat": x[0].cpu().numpy(), "mask": m[0].cpu().numpy()}
            if not show_band:
                solid = binary_erosion(m_in, structure=np.ones((3, 3), bool), border_value=0)
                st["info"] = _pool2(solid)
            stages.append(st)
    return stages


def tan_embedding(model, crop, mask):
    import torch
    x, m = to_tensors(crop, mask)
    with torch.no_grad():
        _, emb = model(x, m)
    return emb[0].numpy()


def with_background(crop, mask, mode, rng):
    out = crop.copy()
    out_mask = ~mask
    if mode == "original":
        return out
    if mode == "white":
        out[out_mask] = 255
    elif mode == "mean":
        out[out_mask] = np.round(crop[mask].mean(axis=0)).astype(np.uint8) if mask.any() else 0
    elif mode == "noise":
        out[out_mask] = rng.integers(0, 256, size=(int(out_mask.sum()), 3), dtype=np.uint8)
    else:
        raise ValueError(mode)
    return out


def rank_superpixels(masks):
    from check_node_survival import first_failing_stage
    areas = np.array([m.sum() for m in masks], dtype=float)
    frac = np.array([m.sum() / m.size for m in masks])
    ok = np.array([first_failing_stage(m, 4) is None for m in masks]) & (frac >= 0.55)
    if not ok.any():
        ok = np.array([first_failing_stage(m, 4) is None for m in masks])
    idx = np.where(ok)[0]
    target = 1.3 * np.median(areas[idx])
    order = np.argsort(np.abs(areas[idx] - target) - 1e-6 * frac[idx])
    return [int(i) for i in idx[order]]


def choose_superpixel(masks):
    return rank_superpixels(masks)[0]


def invariance_test(model, samples, n, seed, resnet_mode):
    import torch
    rng = np.random.default_rng(seed)
    modes = ["original", "white", "mean", "noise"]

    # ResNet18 for comparison: the paper's frozen ImageNet encoder if it loads
    net, mean, std, label = None, None, None, ""
    if resnet_mode == "imagenet":
        try:
            from foundation_encoders import load_foundation_encoder
            spec = load_foundation_encoder(name="resnet18", device=torch.device("cpu"))
            net = spec.model.eval()
            mean = torch.tensor(spec.mean).view(3, 1, 1)
            std = torch.tensor(spec.std).view(3, 1, 1)
            label = "ResNet18 (ImageNet, frozen)"
        except Exception as e:
            print(f"[note] could not load the ImageNet ResNet18 ({type(e).__name__}); using a random-init one")
    if net is None:
        from scratch_cnn import ScratchCNN
        sc = ScratchCNN(num_classes=2).eval()
        net = sc.backbone
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        label = "ResNet18 (random init)"

    def resnet_emb(crop):
        t = (torch.from_numpy(crop.astype(np.float32).transpose(2, 0, 1) / 255.0) - mean) / std
        with torch.no_grad():
            return net(t.unsqueeze(0))[0].numpy().ravel()

    pairs = [(a, b) for i, a in enumerate(modes) for b in modes[i + 1:]]
    tan_abs, tan_rel, cnn_rel = [], [], []
    done = 0
    attempts = 0
    while done < n and attempts < 20 * n:
        attempts += 1
        s = samples[int(rng.integers(len(samples)))]
        j = int(rng.integers(len(s["tan_crops"])))
        crop, mask = s["tan_crops"][j], s["tan_masks"][j].astype(bool)
        if mask.sum() < 50:
            continue
        try:
            te = {m: tan_embedding(model, with_background(crop, mask, m, rng), mask) for m in modes}
            ce = {m: resnet_emb(with_background(crop, mask, m, rng)) for m in modes}
        except Exception:
            continue
        for a, b in pairs:
            tan_abs.append(float(np.abs(te[a] - te[b]).max()))
            tan_rel.append(float(np.linalg.norm(te[a] - te[b]) / (np.linalg.norm(te[a]) + 1e-12)))
            cnn_rel.append(float(np.linalg.norm(ce[a] - ce[b]) / (np.linalg.norm(ce[a]) + 1e-12)))
        done += 1
    print(f"\nBackground invariance over {done} superpixels, all {len(pairs)} pairs of "
          f"backgrounds {modes}:")
    print(f"  TAN                : max |delta embedding| = {max(tan_abs):.3e}   "
          f"relative L2 difference: median {np.median(tan_rel):.3e}, max {max(tan_rel):.3e}")
    print(f"  {label:<27}: relative L2 difference: median {np.median(cnn_rel):.3f}, max {max(cnn_rel):.3f}")
    print("  (TAN differences at float precision (~1e-6) or 0 = invariant; "
          "the ResNet18 difference is the sensitivity to the fill.)")


def main():
    p = argparse.ArgumentParser(description="Figure of one superpixel passing through the trained TAN")
    p.add_argument("dataset", help=".pkl built with --tan-crops (a small one is enough)")
    p.add_argument("--checkpoint", required=True, help="state_dict saved by train_tan_bracs_save.py")
    p.add_argument("--num-classes", type=int, required=True, help="7 for BRACS, 4 for BACH (must match the checkpoint)")
    p.add_argument("--split", choices=["any", "train", "val", "test"], default="test",
                   help="Take the image from this official split (default: test, i.e. an image "
                        "TAN never saw in training). --sample-idx then counts within that split. "
                        "BACH files have no split field; 'any' is used automatically.")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--superpixel-idx", type=int, nargs="+", default=[-1],
                   help="One or more superpixel indices (-1 = pick automatically). Several "
                        "indices write one figure each (suffix _sp<idx>) from a single data load.")
    p.add_argument("--top", type=int, default=1,
                   help="With automatic choice: write figures for the K best-ranked superpixels")
    p.add_argument("--out", default="tan_stages.pdf", help="Output; a PNG with the same stem is also written")
    p.add_argument("--width", type=float, default=7.0, help="Figure width in inches (7 = two columns)")
    p.add_argument("--invariance", type=int, default=0, help="N superpixels for the background-invariance test (0 = off)")
    p.add_argument("--resnet", choices=["imagenet", "random"], default="imagenet")
    p.add_argument("--show-band", action="store_true",
                   help="Draw the edge band (valid positions without a fully valid 3x3 "
                        "neighborhood, which carry only the bias) as colors, as in the first version")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = pickle.load(open(args.dataset, "rb"))
    samples = data["samples"]
    if "tan_crops" not in samples[0]:
        raise SystemExit("No tan_crops in this dataset -- build it with --tan-crops.")
    model = load_tan(args.checkpoint, args.num_classes)

    if args.split != "any" and "split" in samples[0]:
        pool_idx = [i for i, x in enumerate(samples) if x["split"] == args.split]
        if not pool_idx:
            raise SystemExit(f"No images with split '{args.split}' in this dataset.")
        img_idx = pool_idx[args.sample_idx]
    else:
        img_idx = args.sample_idx
    s = samples[img_idx]
    masks = [m.astype(bool) for m in s["tan_masks"]]
    if -1 in args.superpixel_idx:
        chosen = rank_superpixels(masks)[:max(1, args.top)]
    else:
        chosen = list(args.superpixel_idx)
    stem0 = os.path.splitext(args.out)[0]
    for j in chosen:
        crop, mask = s["tan_crops"][j], masks[j]
        print(f"Image {img_idx} (split: {s.get('split', 'n/a')}), superpixel {j}: crop "
              f"{crop.shape[1]}x{crop.shape[0]}, {int(mask.sum())} valid pixels "
              f"({100 * mask.mean():.0f}% of the box)")
        stages = run_stagewise(model, crop, mask, show_band=args.show_band)
        for k, st in enumerate(stages):
            print(f"  stage {k + 1}: {st['feat'].shape[0]} channels, {st['feat'].shape[2]}x{st['feat'].shape[1]} "
                  f"positions, {int((st['mask'] > 0.5).sum())} valid")
        stem = stem0 if len(chosen) == 1 else f"{stem0}_sp{j}"
        plot_figure(crop, mask, stages, stem, width_in=args.width)
        print(f"Saved {stem}.pdf and {stem}.png")

    if args.invariance > 0:
        invariance_test(model, samples, args.invariance, args.seed, args.resnet)


if __name__ == "__main__":
    main()
