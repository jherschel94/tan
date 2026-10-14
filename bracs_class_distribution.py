import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def class_names(label_map):
    inv = {v: k for k, v in label_map.items()}
    return [inv[i] for i in range(len(inv))]


def print_split_table(samples, names):
    print("Images per class after the size filter:\n")
    header = f"{'class':<10}" + "".join(f"{s:>10}" for s in ["train", "val", "test"]) + f"{'total':>10}"
    print(header)
    counts = {s: Counter(x["y"] for x in samples if x["split"] == s) for s in ["train", "val", "test"]}
    for i, n in enumerate(names):
        row = [counts[s][i] for s in ["train", "val", "test"]]
        print(f"{n:<10}" + "".join(f"{v:>10}" for v in row) + f"{sum(row):>10}")
    tot = {s: sum(counts[s].values()) for s in counts}
    print(f"{'total':<10}" + "".join(f"{tot[s]:>10}" for s in ["train", "val", "test"]) + f"{sum(tot.values()):>10}")
    print()
    for s in ["train", "val", "test"]:
        c = [counts[s][i] for i in range(len(names))]
        nz = [v for v in c if v > 0]
        print(f"  {s}: max/min class ratio = {max(nz) / min(nz):.2f}  "
              f"(largest {names[int(np.argmax(c))]}, smallest {names[int(np.argmin(c))]}); "
              f"smallest class share = {100 * min(c) / sum(c):.1f}%")
    print()


def print_superpixel_table(samples, names):
    sp = Counter()
    for x in samples:
        if x["split"] == "train":
            sp[x["y"]] += x["n_nodes"]
    total = sum(sp.values())
    print("Superpixels per class in the training split (TAN's MIL training signal):\n")
    for i, n in enumerate(names):
        print(f"  {n:<10}{sp[i]:>10}   {100 * sp[i] / total:5.1f}%")
    vals = [sp[i] for i in range(len(names)) if sp[i] > 0]
    print(f"  max/min ratio = {max(vals) / min(vals):.2f}\n")


def print_filter_effect(root, names, crop_size, samples):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # some BRACS RoIs are huge; avoid the decompression-bomb error
    root = Path(root)
    total = Counter()
    kept = Counter()
    for split in ["train", "val", "test"]:
        for ci, cname in enumerate(names):
            d = root / split / cname
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.suffix.lower() not in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
                    continue
                total[ci] += 1
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                    if w >= crop_size and h >= crop_size:
                        kept[ci] += 1
                except Exception:
                    pass
    n_pkl = Counter(x["y"] for x in samples)
    print(f"Effect of the {crop_size}x{crop_size} size filter per class (all splits):\n")
    print(f"{'class':<10}{'all':>8}{'kept':>8}{'kept %':>9}{'in pkl':>9}")
    for i, n in enumerate(names):
        pct = 100 * kept[i] / total[i] if total[i] else float("nan")
        print(f"{n:<10}{total[i]:>8}{kept[i]:>8}{pct:>8.1f}%{n_pkl[i]:>9}")
    pcts = [100 * kept[i] / total[i] for i in range(len(names)) if total[i]]
    print(f"\n  kept share ranges from {min(pcts):.1f}% to {max(pcts):.1f}% across classes "
          f"(overall {100 * sum(kept.values()) / sum(total.values()):.1f}%)\n")


def print_recall(results, names):
    print("Per-class recall on the test split (mean +- std over seeds), %:\n")
    print(f"{'model':<14}" + "".join(f"{n:>11}" for n in names))
    for label, path in results:
        r = json.load(open(path))
        seeds = r.get("per_seed_results", [r])
        recalls = []
        for s in seeds:
            cm = np.array(s["confusion_matrix"], dtype=float)
            if cm.shape[0] != len(names):
                print(f"{label:<14}[skipped: confusion matrix is {cm.shape}, expected {len(names)} classes]")
                recalls = None
                break
            recalls.append(np.diag(cm) / cm.sum(axis=1).clip(min=1))
        if recalls is None:
            continue
        rec = 100 * np.array(recalls)
        print(f"{label:<14}" + "".join(f"{m:>6.1f}+-{sd:<4.1f}" for m, sd in zip(rec.mean(0), rec.std(0))))
    print()


def main():
    p = argparse.ArgumentParser(description="BRACS class distribution and per-class recall")
    p.add_argument("dataset", help="A BRACS .pkl built with build_bracs_graphs.py")
    p.add_argument("--bracs-root", default=None, help="BRACS_RoI/latest_version, for the filter-effect table")
    p.add_argument("--crop-size", type=int, default=1000)
    p.add_argument("--results", nargs="*", default=[], help="name=path/to/result.json pairs")
    args = p.parse_args()

    data = pickle.load(open(args.dataset, "rb"))
    samples, names = data["samples"], class_names(data["label_map"])
    print(f"{len(samples)} images in {args.dataset}\n")
    print_split_table(samples, names)
    print_superpixel_table(samples, names)
    if args.bracs_root:
        print_filter_effect(args.bracs_root, names, args.crop_size, samples)
    if args.results:
        print_recall([r.split("=", 1) for r in args.results], names)


if __name__ == "__main__":
    main()
