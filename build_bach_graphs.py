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


LABEL_MAP = {"Benign": 0, "InSitu": 1, "Invasive": 2, "Normal": 3}

FEATURE_COLUMNS = [
    "area", "perimeter", "circularity", "eccentricity", "axis_ratio",
    "solidity", "extent", "n_holes", "radius_of_gyration_norm",
    "lacunarity", "boundary_fractal_dim", "n_skeleton_branches",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    "mean_L", "mean_a", "mean_b", "std_L", "std_a", "std_b",
]

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def find_images(photos_root):
    photos_root = Path(photos_root)
    for class_name in LABEL_MAP:
        class_dir = photos_root / class_name
        if not class_dir.is_dir():
            print(f"  [warn] expected class folder not found: {class_dir}")
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p, class_name


def load_image(path):
    image = io.imread(str(path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image)


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


def process_one_image(args):
    path, class_name, params = args
    try:
        image = load_image(path)
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
    parser = argparse.ArgumentParser(description="Build SLIC superpixel graphs from the BACH dataset, at native resolution")
    parser.add_argument("bach_root", help="Path to the ICIAR2018_BACH_Challenge folder "
                                           "(the one containing Photos/{Benign,InSitu,Invasive,Normal})")
    parser.add_argument("--out", default="bach_graphs.pkl", help="Output pickle path")
    parser.add_argument("--n-superpixels", type=int, default=600)
    parser.add_argument("--slic-compactness", type=float, default=20.0,
                         help="Set higher than skimage's own default (10) deliberately: testing "
                              "found that on large, low-texture/low-gradient regions, SLIC's "
                              "enforce_connectivity post-processing (skimage's default behavior) "
                              "can catastrophically collapse the requested segment count down to "
                              "as few as 1 connected region at compactness=10 -- raising "
                              "compactness makes the spatial-regularity term dominate enough to "
                              "prevent that, at the cost of somewhat less color-adherent "
                              "boundaries. ALWAYS check the printed 'Superpixels per image' "
                              "min (not just median) after a real run; if it's suspiciously low, "
                              "raise this further.")
    parser.add_argument("--limit-per-class", type=int, default=None,
                         help="Only process this many images per class (for a quick smoke test)")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of parallel worker processes (default: all CPU cores)")
    parser.add_argument("--cnn-embeddings", action="store_true",
                         help="Extract a per-superpixel embedding using a pretrained encoder "
                              "(--embedding-model). Always uses fixed-scale, MASKED crops (see "
                              "module docstring) -- every encoder in this pipeline sees only its "
                              "own superpixel's content. Requires torch+torchvision. Runs as a "
                              "SEPARATE, single-process pass after segmentation -- --workers only "
                              "affects the segmentation stage.")
    parser.add_argument("--embedding-model", choices=["resnet18", "uni", "ctranspath"],
                         default="resnet18",
                         help="Which pretrained encoder to use for --cnn-embeddings (see "
                              "foundation_encoders.py). 'uni' requires prior gated HuggingFace "
                              "access + local authentication; 'ctranspath' requires --embedding-"
                              "checkpoint pointing at a downloaded torchscript_model.pt. Each "
                              "model determines its own crop input size and embedding "
                              "dimensionality automatically.")
    parser.add_argument("--embedding-checkpoint", default=None,
                         help="Path to local model weights. Required for ctranspath. Optional "
                              "for uni (loads from a directory you cloned yourself instead of "
                              "the default dynamic hf-hub download; see foundation_encoders.py's "
                              "_load_uni docstring). Ignored for resnet18.")
    parser.add_argument("--embedding-batch-size", type=int, default=64,
                         help="Batch size for the embedding forward pass (default: 64)")
    parser.add_argument("--fill-mode", choices=["mean", "white"], default="mean",
                         help="How the masked-out background is filled for frozen encoders "
                              "(ResNet18/UNI/CTransPath, --cnn-embeddings only): 'mean' "
                              "(default, unchanged) uses each superpixel's own mean color; "
                              "'white' uses constant (255,255,255) regardless of content. "
                              "See main.py's compute_fill_value docstring for the tradeoff.")
    parser.add_argument("--tan-crops", action="store_true",
                         help="Extract raw, unresized (crop, mask) pairs per superpixel for "
                              "TissueAwareNetwork (TAN) (see tissue_aware_network.py / "
                              "train_tan_extractor.py). Stored as sample['tan_crops']/"
                              "['tan_masks'] (variable-size arrays, kept as Python lists -- NOT "
                              "numpy arrays, since sizes differ per superpixel). No torch needed "
                              "for this extraction step, runs in the parallel segmentation "
                              "workers like the other handcrafted features.")
    parser.add_argument("--scratch-crops", action="store_true",
                         help="Extract fixed-size, MASKED crops per superpixel for training a "
                              "randomly-initialized CNN from scratch (see train_scratch_cnn.py) "
                              "-- the architecture-only comparison point against TissueAwareNetwork. "
                              "Stored as sample['scratch_crops']. No torch needed here either.")
    parser.add_argument("--scratch-crop-size", type=int, default=128,
                         help="Pixel size of the fixed-scale scratch-CNN crops (default: 128)")
    args = parser.parse_args()

    params = {
        "n_superpixels": args.n_superpixels,
        "slic_compactness": args.slic_compactness,
        "cnn_embeddings": args.cnn_embeddings,
        "embedding_model": args.embedding_model,
        "tan_crops": args.tan_crops,
        "scratch_crops": args.scratch_crops,
        "scratch_crop_size": args.scratch_crop_size,
        "fill_mode": args.fill_mode,
    }

    photos_root = Path(args.bach_root) / "Photos"
    if not photos_root.is_dir():
        # Common case: the person points at the outer folder they created
        # when unzipping (e.g. "BACH/") rather than the
        # "ICIAR2018_BACH_Challenge" folder actually containing Photos/.
        candidate = Path(args.bach_root) / "ICIAR2018_BACH_Challenge" / "Photos"
        if candidate.is_dir():
            print(f"[note] {photos_root} not found; using {candidate} instead "
                  f"(found ICIAR2018_BACH_Challenge/Photos one level down).")
            photos_root = candidate
    all_images = list(find_images(photos_root))
    if args.limit_per_class is not None:
        limited, per_class_count = [], {c: 0 for c in LABEL_MAP}
        for path, class_name in all_images:
            if per_class_count[class_name] < args.limit_per_class:
                limited.append((path, class_name))
                per_class_count[class_name] += 1
        all_images = limited

    if not all_images:
        print(f"No images found under {photos_root} -- check the path and folder names.")
        sys.exit(1)

    print(f"Found {len(all_images)} images across {len(LABEL_MAP)} classes (native resolution, no downsampling).")
    print(f"Segmentation: SLIC (n_segments={args.n_superpixels}, compactness={args.slic_compactness})")
    for c in LABEL_MAP:
        n = sum(1 for _, cn in all_images if cn == c)
        print(f"  {c}: {n} images")

    work_items = [(path, class_name, params) for path, class_name in all_images]

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
            if done % 10 == 0 or done == len(work_items):
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
              f"({args.n_superpixels}) -- likely SLIC's enforce_connectivity collapsing on "
              f"large low-texture regions (see run_slic's docstring in main.py). Consider "
              f"raising --slic-compactness (currently {args.slic_compactness}).")
    if nan_total > 0:
        print(f"[note] {nan_total} NaN feature values were imputed to 0 across all samples "
              f"(degenerate tiny regions, e.g. axis_ratio undefined for near-1-pixel superpixels)")

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
            if (i + 1) % 20 == 0 or (i + 1) == len(samples):
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
