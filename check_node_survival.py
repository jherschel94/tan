import argparse
import pickle

import numpy as np
from scipy.ndimage import binary_erosion

STRUCT = np.ones((3, 3), dtype=bool)


def pool2(mask):
    h, w = mask.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    if h2 == 0 or w2 == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).max(axis=(1, 3))


def first_failing_stage(mask, n_stages=4):
    m = mask.astype(bool)
    for stage in range(1, n_stages + 1):
        if m.size == 0 or not binary_erosion(m, structure=STRUCT, border_value=0).any():
            return stage
        m = pool2(m)
    return None


def main():
    p = argparse.ArgumentParser(description="Fraction of superpixels that collapse inside TAN")
    p.add_argument("dataset", help=".pkl built with --tan-crops")
    p.add_argument("--max-images", type=int, default=None, help="Use only the first N images")
    p.add_argument("--stages", type=int, nargs="+", default=[3, 4])
    args = p.parse_args()

    data = pickle.load(open(args.dataset, "rb"))
    samples = data["samples"]
    if "tan_masks" not in samples[0]:
        raise SystemExit("No tan_masks in this dataset -- build it with --tan-crops.")
    if args.max_images:
        samples = samples[:args.max_images]

    fails = []
    areas = []
    for s in samples:
        for m in s["tan_masks"]:
            fails.append(first_failing_stage(m, max(args.stages)))
            areas.append(int(np.count_nonzero(m)))
    n = len(fails)
    print(f"{len(samples)} images, {n} superpixels; "
          f"valid pixels per superpixel: min {min(areas)}, median {int(np.median(areas))}")
    for k in args.stages:
        collapsed = sum(1 for f in fails if f is not None and f <= k)
        print(f"  {k}-stage configuration: {collapsed} of {n} superpixels collapse "
              f"({100 * collapsed / n:.3f}%)")
    stage_counts = {st: sum(1 for f in fails if f == st) for st in range(1, max(args.stages) + 1)}
    print(f"  first failing stage (among collapsed): {stage_counts}")


if __name__ == "__main__":
    main()
