import argparse
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from skimage import io, color, img_as_float

from main import (run_slic, extract_superpixel_features, build_superpixel_graph,
                    load_embedding_model, extract_cnn_embeddings_per_superpixel,
                    prepare_variable_size_crops, prepare_fixed_scale_crops)

LABEL_MAP = {
    "0_N": 0, "1_PB": 1, "2_UDH": 2, "3_FEA": 3,
    "4_ADH": 4, "5_DCIS": 5, "6_IC": 6,
}

SPLIT_FOLDERS = ["train", "val", "test"]

FEATURE_COLUMNS = [
    "area", "perimeter", "circularity", "eccentricity", "axis_ratio",
    "solidity", "extent", "n_holes", "radius_of_gyration_norm",
    "lacunarity", "boundary_fractal_dim", "n_skeleton_branches",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    "mean_L", "mean_a", "mean_b", "std_L", "std_a", "std_b",
]

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def find_images(bracs_root):
    bracs_root = Path(bracs_root)
    found = []
    for split in SPLIT_FOLDERS:
        split_dir = bracs_root / split
        if not split_dir.is_dir():
            print(f"  [warn] expected split folder not found: {split_dir}")
            continue
        for class_name in LABEL_MAP:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                print(f"  [warn] expected class folder not found: {class_dir}")
                continue
            for p in class_dir.iterdir():
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    found.append((p, class_name, split))
    found.sort(key=lambda item: (item[2], item[1], str(item[0])))
    for path, class_name, split in found:
        yield path, class_name, split


def load_image(path, max_long_edge=None):
    image = io.imread(str(path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = np.ascontiguousarray(image)

    if max_long_edge is not None:
        h, w = image.shape[:2]
        long_edge = max(h, w)
        if long_edge > max_long_edge:
            from skimage import transform
            scale = max_long_edge / long_edge
            new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
            image = transform.resize(image, (new_h, new_w), anti_aliasing=True, preserve_range=True)
            image = np.ascontiguousarray(image.astype(np.uint8))
    return image


def graph_to_arrays_with_edge_attr(G, feature_columns):
    nodes = sorted(G.nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    X = np.array([[G.nodes[n][c] for c in feature_columns] for n in nodes], dtype=np.float32)

    edges, edge_attrs = [], []
    for a, b, data in G.edges(data=True):
        attr = [data.get("shared_boundary", 0.0), data.get("centroid_distance", 0.0)]
        edges.append((pos[a], pos[b])); edge_attrs.append(attr)
        edges.append((pos[b], pos[a])); edge_attrs.append(attr)

    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.array(edge_attrs, dtype=np.float32) if edge_attrs else np.zeros((0, 2), dtype=np.float32)
    return X, edge_index, edge_attr


def crop_center(image, crop_size):
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return None
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return np.ascontiguousarray(image[y0:y0 + crop_size, x0:x0 + crop_size])


def process_one_image(args):
    path, class_name, split_name, params = args
    try:
        image = load_image(path)
        image = crop_center(image, params["crop_size"])
        if image is None:
            return None, (str(path), f"Skipped: smaller than crop_size={params['crop_size']}px "
                                      f"in at least one dimension")
        lab_img = color.rgb2lab(img_as_float(image))

        labels = run_slic(image, n_segments=params["n_superpixels"],
                           compactness=params["slic_compactness"])

        features_df = extract_superpixel_features(image, labels, lab_img=lab_img)
        graph = build_superpixel_graph(labels, features_df)
        X, edge_index, edge_attr = graph_to_arrays_with_edge_attr(graph, FEATURE_COLUMNS)

        n_nan = int(np.isnan(X).sum())
        if n_nan > 0:
            X = np.nan_to_num(X, nan=0.0)

        tan_crops, tan_masks = None, None
        if params.get("tan_crops"):
            ordered_labels, tan_crops, tan_masks = prepare_variable_size_crops(image, labels)

        scratch_crops = None
        if params.get("scratch_crops"):
            ordered_labels_sc, crops_list = prepare_fixed_scale_crops(
                image, labels, crop_size=params["scratch_crop_size"], mask_outside=True,
                fill_mode=params["fill_mode"],
            )
            n_nodes_sc = int(labels.max()) + 1
            scratch_crops = np.zeros((n_nodes_sc, *crops_list[0].shape), dtype=np.uint8)
            for lbl, crop in zip(ordered_labels_sc, crops_list):
                scratch_crops[lbl] = crop

        sample = {
            "x": X,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": LABEL_MAP[class_name],
            "path": str(path),
            "split": split_name,
            "n_nodes": int(X.shape[0]),
            "n_nan_imputed": n_nan,
        }
        if tan_crops is not None:
            sample["tan_crops"] = tan_crops
            sample["tan_masks"] = tan_masks
        if scratch_crops is not None:
            sample["scratch_crops"] = scratch_crops
        if params.get("cnn_embeddings"):
            sample["_image"] = image
            sample["_labels"] = labels
        return sample, None
    except Exception as e:
        return None, (str(path), f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")


def main():
    parser = argparse.ArgumentParser(description="Build SLIC superpixel graphs from BRACS RoI, at native resolution")
    parser.add_argument("bracs_root", help="Path to BRACS_RoI/latest_version "
                                            "(the folder directly containing train/, val/, test/)")
    parser.add_argument("--out", default="bracs_graphs.pkl", help="Output pickle path")
    parser.add_argument("--crop-size", type=int, default=1000,
                         help="Every image is center-cropped to exactly crop_size x crop_size "
                              "before segmentation (default: 1000). BRACS RoIs vary too widely "
                              "in native size (this project's own downloaded copy: ~127px to "
                              "over 17,000px on a side) to segment at native resolution without "
                              "wildly inconsistent physical superpixel granularity across "
                              "images. A fixed crop makes every processed image genuinely "
                              "comparable (same real-world field of view, since BRACS is "
                              "scanned at a consistent 0.25 micron/pixel), at the cost of "
                              "excluding images smaller than crop_size in either dimension -- "
                              "see --crop-size's interaction with --limit-per-class below; "
                              "check what fraction of your own downloaded copy qualifies before "
                              "trusting a given crop_size (see this module's docstring for a "
                              "size-check snippet), and report that fraction in the paper.")
    parser.add_argument("--n-superpixels", type=int, default=400,
                         help="Fixed segment count applied to every (now uniformly-sized, "
                              "post-crop) image. Default (400) targets the same ~70px physical "
                              "superpixel side length this project established for BACH, "
                              "computed for a 1000x1000 crop -- adjust if you change "
                              "--crop-size, since the physical-size-per-segment relationship "
                              "depends on crop_size^2 / n_superpixels.")
    parser.add_argument("--slic-compactness", type=float, default=20.0)
    parser.add_argument("--limit-per-class", type=int, default=None,
                         help="Only process this many images per class, PER SPLIT (for a "
                              "quick smoke test) -- e.g. 15 means up to 15 from train, up to "
                              "15 from val, up to 15 from test, per class.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cnn-embeddings", action="store_true")
    parser.add_argument("--embedding-model", choices=["resnet18", "uni", "ctranspath"],
                         default="resnet18")
    parser.add_argument("--embedding-checkpoint", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--fill-mode", choices=["mean", "white"], default="mean",
                         help="How the masked-out background is filled for frozen encoders "
                              "(ResNet18/UNI/CTransPath, --cnn-embeddings only): 'mean' "
                              "(default, unchanged) uses each superpixel's own mean color; "
                              "'white' uses constant (255,255,255) regardless of content. "
                              "See main.py's compute_fill_value docstring for the tradeoff.")
    parser.add_argument("--tan-crops", action="store_true",
                         help="Extract raw, unresized (crop, mask) pairs per superpixel for "
                              "TissueAwareNetwork (TAN). Stored as sample['tan_crops']/"
                              "['tan_masks'].")
    parser.add_argument("--scratch-crops", action="store_true",
                         help="Extract fixed-size, MASKED crops per superpixel for "
                              "ScratchCNN. Stored as sample['scratch_crops'].")
    parser.add_argument("--scratch-crop-size", type=int, default=128)
    args = parser.parse_args()

    params = {
        "n_superpixels": args.n_superpixels,
        "crop_size": args.crop_size,
        "slic_compactness": args.slic_compactness,
        "cnn_embeddings": args.cnn_embeddings,
        "embedding_model": args.embedding_model,
        "tan_crops": args.tan_crops,
        "scratch_crops": args.scratch_crops,
        "scratch_crop_size": args.scratch_crop_size,
        "fill_mode": args.fill_mode,
    }

    all_images = list(find_images(args.bracs_root))

    from PIL import Image
    kept, excluded_count = [], 0
    for path, class_name, split in all_images:
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception:
            w = h = 0  # unreadable -- let process_one_image's own
                       # try/except handle and report this properly
        if w >= args.crop_size and h >= args.crop_size:
            kept.append((path, class_name, split))
        else:
            excluded_count += 1
    if excluded_count > 0:
        pct = 100 * excluded_count / len(all_images)
        print(f"[note] excluded {excluded_count}/{len(all_images)} images ({pct:.1f}%) "
              f"smaller than --crop-size={args.crop_size}px in at least one dimension -- "
              f"report this fraction in the paper.")
    all_images = kept

    if args.limit_per_class is not None:
        from collections import defaultdict
        counts = defaultdict(int)
        limited = []
        for path, class_name, split in all_images:
            key = (split, class_name)
            if counts[key] < args.limit_per_class:
                limited.append((path, class_name, split))
                counts[key] += 1
        all_images = limited

    if not all_images:
        print(f"No images found under {args.bracs_root} -- check the path and folder names.")
        sys.exit(1)

    print(f"Found {len(all_images)} images across {len(LABEL_MAP)} classes "
          f"(center-cropped to {args.crop_size}x{args.crop_size}, official train/val/test "
          f"split PRESERVED -- not merged, see sample['split']).")
    print(f"Segmentation: SLIC (n_segments={args.n_superpixels}, "
          f"compactness={args.slic_compactness})")
    for split in SPLIT_FOLDERS:
        split_images = [img for img in all_images if img[2] == split]
        print(f"  {split}: {len(split_images)} images total")
        for c in LABEL_MAP:
            n = sum(1 for _, cn, sp in all_images if cn == c and sp == split)
            print(f"    {c}: {n} images")

    work_items = [(path, class_name, split, params) for path, class_name, split in all_images]

    samples, failures = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_image, item): item for item in work_items}
        done = 0
        for future in as_completed(futures):
            sample, error = future.result()
            done += 1
            if sample is not None:
                samples.append(sample)
            else:
                failures.append(error)
            if done % 50 == 0 or done == len(work_items):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{len(work_items)}] {elapsed:.0f}s elapsed, "
                      f"{rate:.2f} img/s, {len(failures)} failed", flush=True)

    print(f"\nDone: {len(samples)} succeeded, {len(failures)} failed, "
          f"{time.time()-t0:.0f}s total.")

    samples.sort(key=lambda s: s["path"])
    if failures:
        print("Failed images:")
        for path, err in failures[:20]:
            print(f"  {path}: {err.splitlines()[0]}")
        if len(failures) > 20:
            print(f"  ... and {len(failures)-20} more")

    node_counts = [s["n_nodes"] for s in samples]
    nan_total = sum(s["n_nan_imputed"] for s in samples)
    print(f"\nSuperpixels per image: min={min(node_counts)}, "
          f"median={int(np.median(node_counts))}, max={max(node_counts)}")
    if min(node_counts) < 0.2 * args.n_superpixels:
        n_collapsed = sum(1 for c in node_counts if c < 0.2 * args.n_superpixels)
        print(f"[warn] {n_collapsed} image(s) produced far fewer superpixels than requested "
              f"({args.n_superpixels}) -- likely SLIC collapse on large low-texture regions "
              f"within the cropped region. Consider raising --slic-compactness.")
    if nan_total > 0:
        print(f"[note] {nan_total} NaN feature values were imputed to 0 across all samples")

    embedding_dim = None
    if args.cnn_embeddings and samples:
        print(f"\nExtracting {args.embedding_model} embeddings for {len(samples)} images "
              f"(single process, always masked, fixed-scale crops)...")
        spec, device = load_embedding_model(encoder_name=args.embedding_model,
                                             checkpoint_path=args.embedding_checkpoint)
        print(f"  embedding model on device: {device} "
              f"(input_size={spec.input_size}, embedding_dim={spec.embedding_dim})")
        t1 = time.time()
        for i, sample in enumerate(samples):
            sample["cnn_embedding"] = extract_cnn_embeddings_per_superpixel(
                sample.pop("_image"), sample.pop("_labels"),
                batch_size=args.embedding_batch_size, spec=spec, device=device,
                crop_mode="fixed_scale", mask_outside=True, fill_mode=args.fill_mode,
            )
            if embedding_dim is None:
                embedding_dim = sample["cnn_embedding"].shape[1]
            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                elapsed = time.time() - t1
                print(f"  [{i+1}/{len(samples)}] {elapsed:.0f}s elapsed, "
                      f"{(i+1)/elapsed:.2f} img/s", flush=True)
        print(f"Done extracting embeddings, {time.time()-t1:.0f}s total.")

    output = {
        "samples": samples,
        "label_map": LABEL_MAP,
        "feature_columns": FEATURE_COLUMNS,
        "params": params,
        "cnn_embedding_dim": embedding_dim,
    }
    with open(args.out, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved {len(samples)} graphs -> {args.out}")


if __name__ == "__main__":
    main()
