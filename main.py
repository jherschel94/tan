import pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy import ndimage as ndi
from skimage import io, color, segmentation, img_as_float, morphology
from PIL import Image
from skimage.filters import rank
from skimage.measure import regionprops
from skimage.feature import graycoprops


def compute_radius_of_gyration_norm(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    cy, cx = ys.mean(), xs.mean()
    rg = np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).mean())
    r_equiv = np.sqrt(len(ys) / np.pi)
    return float(rg / r_equiv) if r_equiv > 0 else 0.0


def compute_lacunarity(mask, box_sizes=(2, 3, 4, 6, 8, 12)):
    mask = mask.astype(float)
    h, w = mask.shape
    valid_sizes = [b for b in box_sizes if b <= min(h, w)]
    vals = []
    for b in valid_sizes:
        integral = np.pad(np.cumsum(np.cumsum(mask, axis=0), axis=1), ((1, 0), (1, 0)))
        box_sums = (integral[b:, b:] - integral[:-b, b:]
                    - integral[b:, :-b] + integral[:-b, :-b]).ravel()
        if box_sums.mean() > 0:
            vals.append(1.0 + box_sums.var() / box_sums.mean() ** 2)
    return float(np.mean(vals)) if vals else 0.0


def compute_boundary_fractal_dimension(mask, box_sizes=(2, 3, 4, 6, 8, 12, 16)):
    boundary = mask & ~ndi.binary_erosion(mask)
    ys, xs = np.nonzero(boundary)
    if len(ys) < 4:
        return 1.0
    h, w = mask.shape
    valid_sizes = [b for b in box_sizes if b <= min(h, w)]
    counts, sizes_used = [], []
    for b in valid_sizes:
        occupied = {(int(y // b), int(x // b)) for y, x in zip(ys, xs)}
        if occupied:
            counts.append(len(occupied))
            sizes_used.append(b)
    if len(counts) < 2:
        return 1.0
    log_inv_size = np.log(1.0 / np.array(sizes_used, dtype=float))
    log_counts = np.log(np.array(counts, dtype=float))
    slope, _ = np.polyfit(log_inv_size, log_counts, 1)
    return float(slope)


def compute_skeleton_branch_points(mask):
    skel = morphology.skeletonize(mask)
    if not skel.any():
        return 0
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndi.convolve(skel.astype(int), kernel, mode='constant', cval=0)
    return int((skel & (neighbor_count >= 3)).sum())


def compute_masked_glcm(gray_levels_patch, mask, levels, distance=1):
    offsets = [(0, distance), (-distance, distance), (distance, 0), (distance, distance)]
    h, w = gray_levels_patch.shape
    glcm = np.zeros((levels, levels, 1, len(offsets)), dtype=np.float64)
    mask = mask.astype(bool)

    for a_idx, (dy, dx) in enumerate(offsets):
        y0a, y1a = max(0, -dy), h - max(0, dy)
        x0a, x1a = max(0, -dx), w - max(0, dx)
        y0b, y1b = max(0, dy), h - max(0, -dy)
        x0b, x1b = max(0, dx), w - max(0, -dx)
        if y1a <= y0a or x1a <= x0a:
            continue

        src, dst = gray_levels_patch[y0a:y1a, x0a:x1a], gray_levels_patch[y0b:y1b, x0b:x1b]
        valid = mask[y0a:y1a, x0a:x1a] & mask[y0b:y1b, x0b:x1b]
        if not valid.any():
            continue

        hist2d, _, _ = np.histogram2d(
            src[valid], dst[valid], bins=levels, range=[[0, levels], [0, levels]]
        )
        glcm[:, :, 0, a_idx] = hist2d + hist2d.T

    return glcm


def compute_glcm_texture_features(L_patch, mask, levels=16, distance=1):
    quantized = np.clip((L_patch / 100.0 * levels), 0, levels - 1).astype(np.int64)
    glcm = compute_masked_glcm(quantized, mask, levels=levels, distance=distance)

    if glcm.sum() == 0:
        return {"glcm_contrast": np.nan, "glcm_homogeneity": np.nan,
                "glcm_energy": np.nan, "glcm_correlation": np.nan}

    return {
        "glcm_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "glcm_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        "glcm_energy": float(np.mean(graycoprops(glcm, "energy"))),
        "glcm_correlation": float(np.nanmean(graycoprops(glcm, "correlation"))),
    }


N_RANDOM_KERNELS = 128
RANDOM_KERNEL_SEED = 42  # fixed on purpose -- see generate_random_kernels' docstring
DEFAULT_KERNEL_SIZES = (3, 5, 7)


def generate_random_kernels(n_kernels=N_RANDOM_KERNELS, size=3, seed=RANDOM_KERNEL_SEED):
    rng = np.random.default_rng(seed)
    kernels = rng.normal(0, 1, (n_kernels, size, size))
    norms = np.linalg.norm(kernels.reshape(n_kernels, -1), axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    kernels = kernels / norms.reshape(n_kernels, 1, 1)
    return kernels.astype(np.float32)


def extract_random_kernel_features(labels, L_channel, kernels=None):
    if kernels is None:
        kernels = generate_random_kernels()
    n_kernels = kernels.shape[0]
    n = int(labels.max()) + 1

    responses = [ndi.correlate(L_channel, k, mode="mirror") for k in kernels]

    flat_labels = labels.ravel()
    counts = np.bincount(flat_labels, minlength=n).astype(np.float64)
    counts_safe = np.where(counts == 0, 1, counts)

    means = np.zeros((n, n_kernels), dtype=np.float32)
    stds = np.zeros((n, n_kernels), dtype=np.float32)
    for k in range(n_kernels):
        resp_flat = responses[k].ravel()
        s1 = np.bincount(flat_labels, weights=resp_flat, minlength=n)
        s2 = np.bincount(flat_labels, weights=resp_flat ** 2, minlength=n)
        mean = s1 / counts_safe
        var = np.clip(s2 / counts_safe - mean ** 2, 0, None)
        means[:, k] = mean
        stds[:, k] = np.sqrt(var)

    return np.concatenate([means, stds], axis=1)  # (n, 2 * n_kernels)


def extract_multiscale_kernel_features(labels, L_channel, kernel_sizes=DEFAULT_KERNEL_SIZES,
                                        n_kernels_per_size=N_RANDOM_KERNELS):
    kernel_sets = {size: generate_random_kernels(n_kernels=n_kernels_per_size, size=size)
                   for size in kernel_sizes}
    blocks = [extract_random_kernel_features(labels, L_channel, kernels=kernel_sets[size])
              for size in kernel_sizes]
    return np.concatenate(blocks, axis=1)


def multiscale_kernel_feature_names(kernel_sizes=DEFAULT_KERNEL_SIZES,
                                     n_kernels_per_size=N_RANDOM_KERNELS):
    names = []
    for size in kernel_sizes:
        names += [f"rk{size}_mean_{i}" for i in range(n_kernels_per_size)]
        names += [f"rk{size}_std_{i}" for i in range(n_kernels_per_size)]
    return names


def compute_fill_value(crop, mask, fill_mode="mean"):
    if fill_mode == "white":
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    elif fill_mode == "mean":
        return crop[mask].mean(axis=0) if mask.any() else crop.reshape(-1, crop.shape[-1]).mean(axis=0)
    else:
        raise ValueError(f"Unknown fill_mode {fill_mode!r} -- expected 'mean' or 'white'.")


def prepare_superpixel_crops(image, labels, crop_size=128, fill_mode="mean"):
    props = regionprops(labels + 1)
    ordered_labels, crops = [], []

    for region in props:
        lbl = region.label - 1
        minr, minc, maxr, maxc = region.bbox
        mask = region.image
        crop = image[minr:maxr, minc:maxc].astype(np.float32)
        if crop.ndim == 2:
            crop = np.stack([crop] * 3, axis=-1)

        fill = compute_fill_value(crop, mask, fill_mode)
        masked_crop = crop.copy()
        masked_crop[~mask] = fill
        masked_crop = np.clip(masked_crop, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(masked_crop).resize((crop_size, crop_size), Image.BILINEAR)
        ordered_labels.append(lbl)
        crops.append(np.array(pil_img))  # np.array (not asarray) forces a writable copy

    return ordered_labels, crops


def prepare_fixed_scale_crops(image, labels, crop_size=224, mask_outside=False, fill_mode="mean"):
    h, w = image.shape[:2]
    props = regionprops(labels + 1)
    ordered_labels, crops = [], []
    half = crop_size // 2

    for region in props:
        lbl = region.label - 1
        cy, cx = region.centroid  # skimage convention: (row, col) = (y, x)
        cy, cx = int(round(cy)), int(round(cx))

        if h < crop_size or w < crop_size:
            pad_h, pad_w = max(0, crop_size - h), max(0, crop_size - w)
            padded = np.pad(image, ((pad_h // 2, pad_h - pad_h // 2),
                                     (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode="edge")
            crop = padded[:crop_size, :crop_size].copy()
            y0 = x0 = 0  # for the mask_outside branch below; degenerate, rarely exercised
        else:
            y0 = max(0, min(cy - half, h - crop_size))
            x0 = max(0, min(cx - half, w - crop_size))
            crop = image[y0:y0 + crop_size, x0:x0 + crop_size].copy()

        if crop.ndim == 2:
            crop = np.stack([crop] * 3, axis=-1)

        if mask_outside:
            minr, minc, maxr, maxc = region.bbox
            mask_full = np.zeros((h, w), dtype=bool)
            mask_full[minr:maxr, minc:maxc] = region.image
            mask_crop = mask_full[y0:y0 + crop_size, x0:x0 + crop_size]
            if mask_crop.shape != crop.shape[:2]:
                mask_crop = np.ones(crop.shape[:2], dtype=bool)  # degenerate small-image fallback
            fill = compute_fill_value(crop, mask_crop, fill_mode)
            crop = crop.copy()
            crop[~mask_crop] = fill

        ordered_labels.append(lbl)
        crops.append(np.array(crop, dtype=np.uint8))

    return ordered_labels, crops


def prepare_variable_size_crops(image, labels):
    props = regionprops(labels + 1)
    ordered_labels, crops, masks = [], [], []

    for region in props:
        lbl = region.label - 1
        minr, minc, maxr, maxc = region.bbox
        mask = region.image
        crop = image[minr:maxr, minc:maxc]
        if crop.ndim == 2:
            crop = np.stack([crop] * 3, axis=-1)

        ordered_labels.append(lbl)
        crops.append(np.array(crop, dtype=np.uint8))
        masks.append(np.array(mask, dtype=bool))

    return ordered_labels, crops, masks


def load_embedding_model(device=None, encoder_name="resnet18", checkpoint_path=None):
    import torch
    from foundation_encoders import load_foundation_encoder

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_foundation_encoder(encoder_name, device=device, checkpoint_path=checkpoint_path)
    return spec, device


def extract_cnn_embeddings_per_superpixel(image, labels, batch_size=64, spec=None, device=None,
                                           encoder_name="resnet18", checkpoint_path=None,
                                           crop_mode="fixed_scale", mask_outside=False,
                                           fill_mode="mean"):
    import torch
    import torchvision.transforms.functional as TF

    if spec is None or device is None:
        spec, device = load_embedding_model(device, encoder_name=encoder_name,
                                             checkpoint_path=checkpoint_path)

    if crop_mode == "fixed_scale":
        ordered_labels, crops = prepare_fixed_scale_crops(
            image, labels, crop_size=spec.input_size, mask_outside=mask_outside, fill_mode=fill_mode
        )
    elif crop_mode == "bbox_resize":
        ordered_labels, crops = prepare_superpixel_crops(image, labels, crop_size=spec.input_size,
                                                          fill_mode=fill_mode)
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode!r} (expected 'fixed_scale' or 'bbox_resize')")

    n = int(labels.max()) + 1
    embeddings = np.zeros((n, spec.embedding_dim), dtype=np.float32)
    if not crops:
        return embeddings

    mean = torch.tensor(spec.mean).view(3, 1, 1)
    std = torch.tensor(spec.std).view(3, 1, 1)

    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch_crops = crops[start:start + batch_size]
            batch_labels = ordered_labels[start:start + batch_size]
            tensors = [(TF.to_tensor(c) - mean) / std for c in batch_crops]
            batch_tensor = torch.stack(tensors).to(device)
            out = spec.model(batch_tensor).cpu().numpy()
            for i, lbl in enumerate(batch_labels):
                embeddings[lbl] = out[i]

    return embeddings


# ==========================================================================
# 6. PER-SUPERPIXEL FEATURE EXTRACTION (shape + color)
# ==========================================================================
def extract_superpixel_features(image, labels, lab_img=None, glcm_levels=16, glcm_distance=1):
    if lab_img is None:
        lab_img = color.rgb2lab(img_as_float(image))

    n = int(labels.max()) + 1
    flat_labels = labels.ravel()

    # --- color features (mean + std per channel), via bincount, same
    # pattern used elsewhere in this script for center recomputation ---
    counts = np.bincount(flat_labels, minlength=n).astype(float)
    records_color = {}
    for ci, name in enumerate(['L', 'a', 'b']):
        channel = lab_img[:, :, ci].ravel()
        s1 = np.bincount(flat_labels, weights=channel, minlength=n)
        s2 = np.bincount(flat_labels, weights=channel ** 2, minlength=n)
        mean = s1 / counts
        var = np.clip(s2 / counts - mean ** 2, 0, None)
        records_color[f'mean_{name}'] = mean
        records_color[f'std_{name}'] = np.sqrt(var)

    # --- shape features, via regionprops on labels+1 (regionprops treats
    # label 0 as background and silently drops it -- every pixel in this
    # segmentation belongs to a real superpixel, so we shift by 1 to avoid
    # accidentally losing label 0's region) ---
    props = regionprops(labels + 1)
    rows = []
    L_channel = lab_img[:, :, 0]

    for region in props:
        lbl = region.label - 1
        mask = region.image
        minr, minc, maxr, maxc = region.bbox
        L_patch = L_channel[minr:maxr, minc:maxc]
        texture = compute_glcm_texture_features(L_patch, mask, levels=glcm_levels,
                                                 distance=glcm_distance)
        area = region.area
        perimeter = region.perimeter if region.perimeter > 0 else 1.0
        minor = region.axis_minor_length if region.axis_minor_length > 0 else np.nan
        row = {
            'label': lbl,
            'area': area,
            'perimeter': perimeter,
            'circularity': 4 * np.pi * area / (perimeter ** 2),
            'eccentricity': region.eccentricity,
            'axis_ratio': (region.axis_major_length / minor) if minor and not np.isnan(minor) else np.nan,
            'solidity': region.solidity,
            'extent': region.extent,
            'n_holes': max(0, 1 - region.euler_number),
            'radius_of_gyration_norm': compute_radius_of_gyration_norm(mask),
            'lacunarity': compute_lacunarity(mask),
            'boundary_fractal_dim': compute_boundary_fractal_dimension(mask),
            'n_skeleton_branches': compute_skeleton_branch_points(mask),
            'glcm_contrast': texture['glcm_contrast'],
            'glcm_homogeneity': texture['glcm_homogeneity'],
            'glcm_energy': texture['glcm_energy'],
            'glcm_correlation': texture['glcm_correlation'],
            'centroid_y': region.centroid[0],
            'centroid_x': region.centroid[1],
            'mean_L': records_color['mean_L'][lbl],
            'mean_a': records_color['mean_a'][lbl],
            'mean_b': records_color['mean_b'][lbl],
            'std_L': records_color['std_L'][lbl],
            'std_a': records_color['std_a'][lbl],
            'std_b': records_color['std_b'][lbl],
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index('label').sort_index()
    return df


def build_superpixel_graph(labels, features_df):
    G = nx.Graph()
    for lbl, row in features_df.iterrows():
        G.add_node(int(lbl), **row.to_dict())

    right_diff = labels[:, :-1] != labels[:, 1:]
    down_diff = labels[:-1, :] != labels[1:, :]
    ys_r, xs_r = np.nonzero(right_diff)
    ys_d, xs_d = np.nonzero(down_diff)

    boundary_counts = {}
    for y, x in zip(ys_r, xs_r):
        a, b = int(labels[y, x]), int(labels[y, x + 1])
        key = (a, b) if a < b else (b, a)
        boundary_counts[key] = boundary_counts.get(key, 0) + 1
    for y, x in zip(ys_d, xs_d):
        a, b = int(labels[y, x]), int(labels[y + 1, x])
        key = (a, b) if a < b else (b, a)
        boundary_counts[key] = boundary_counts.get(key, 0) + 1

    for (a, b), shared in boundary_counts.items():
        ca = np.array([features_df.loc[a, 'centroid_y'], features_df.loc[a, 'centroid_x']])
        cb = np.array([features_df.loc[b, 'centroid_y'], features_df.loc[b, 'centroid_x']])
        G.add_edge(a, b, shared_boundary=shared, centroid_distance=float(np.linalg.norm(ca - cb)))

    return G


def graph_to_arrays(G, feature_columns):
    nodes = sorted(G.nodes)
    node_pos = {n: i for i, n in enumerate(nodes)}
    X = np.array([[G.nodes[n][c] for c in feature_columns] for n in nodes], dtype=float)
    edges = []
    for a, b in G.edges:
        edges.append((node_pos[a], node_pos[b]))
        edges.append((node_pos[b], node_pos[a]))
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return X, edge_index


def colorize_labels(labels, seed=0):
    n_labels = labels.max() + 1
    rng = np.random.default_rng(seed)
    palette = rng.uniform(0.15, 1.0, size=(n_labels, 3))  # avoid near-black colors
    return palette[labels]


def run_slic(image, n_segments, compactness):
    return segmentation.slic(
        img_as_float(image), n_segments=n_segments, compactness=compactness,
        start_label=0,
    )

