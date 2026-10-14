import argparse
import math
import pickle
import random

import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from tissue_aware_network import TissueAwareNetwork

ROTATE_90_PROB = 1.0     
ARBITRARY_ROT_PROB = 0.0 
ARBITRARY_ROT_MAX_DEG = 20


def random_intensity_aug(img, mask):
    m = mask > 0
    if not m.any():
        return img
    if random.random() < 0.7:
        contrast = random.uniform(0.85, 1.15)
        brightness = random.uniform(-0.08, 0.08)
        img[m] = np.clip(img[m] * contrast + brightness, 0.0, 1.0)
    if random.random() < 0.5:
        scale = np.random.uniform(0.9, 1.1, size=(1, img.shape[-1])).astype(np.float32)
        img[m] = np.clip(img[m] * scale, 0.0, 1.0)
    if random.random() < 0.5:
        noise = np.random.normal(0.0, 0.02, size=img[m].shape).astype(np.float32)
        img[m] = np.clip(img[m] + noise, 0.0, 1.0)
    return img


def train_aug(img, mask):
    if random.random() > 0.5:
        img = img[:, ::-1, :].copy()
        mask = mask[:, ::-1].copy()
    if random.random() > 0.5:
        img = img[::-1, :, :].copy()
        mask = mask[::-1, :].copy()

    if random.random() < ROTATE_90_PROB:
        k = random.randint(0, 3)
        img = np.rot90(img, k, axes=(0, 1)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()

    if random.random() < ARBITRARY_ROT_PROB:
        angle = random.uniform(-ARBITRARY_ROT_MAX_DEG, ARBITRARY_ROT_MAX_DEG)
        img_rot = np.stack([
            ndimage.rotate(img[:, :, c], angle, reshape=False, order=1,
                            mode="constant", cval=0.0)
            for c in range(img.shape[2])
        ], axis=-1)
        mask_rot = ndimage.rotate(mask.astype(np.float32), angle, reshape=False, order=0,
                                   mode="constant", cval=0.0)
        mask_rot = (mask_rot > 0.5).astype(np.float32)
        img = (img_rot * mask_rot[:, :, None]).astype(np.float32)
        mask = mask_rot

    img = random_intensity_aug(img, mask)
    return img.astype(np.float32), mask.astype(np.float32)



class SuperpixelCropDataset(Dataset):

    def __init__(self, samples, sample_indices, augment=False):
        self.items = []  # (crop, mask, label)
        self.sizes = []  # crop area, for the size-aware batch sampler below
        self.augment = augment
        for i in sample_indices:
            s = samples[i]
            for crop, mask in zip(s["tan_crops"], s["tan_masks"]):
                self.items.append((crop, mask, s["y"]))
                self.sizes.append(crop.shape[0] * crop.shape[1])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        crop, mask, label = self.items[idx]
        img = crop.astype(np.float32) / 255.0  # HWC, [0, 1]
        m = mask.astype(np.float32)
        if self.augment:
            img, m = train_aug(img, m)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # -> CHW for the model
        m = np.ascontiguousarray(m)
        return torch.from_numpy(img), torch.from_numpy(m), label


class SizeBatchSampler(Sampler):

    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sorted_indices = sorted(range(len(dataset)), key=lambda i: dataset.sizes[i])

    def __iter__(self):
        batches = [self.sorted_indices[i:i + self.batch_size]
                   for i in range(0, len(self.sorted_indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self):
        return math.ceil(len(self.sorted_indices) / self.batch_size)


def pad_collate(batch):
    imgs, masks, labels = zip(*batch)
    max_h = max(im.shape[1] for im in imgs)
    max_w = max(im.shape[2] for im in imgs)
    imgs_p = torch.stack([F.pad(im, (0, max_w - im.shape[2], 0, max_h - im.shape[1])) for im in imgs])
    masks_p = torch.stack([F.pad(m, (0, max_w - m.shape[1], 0, max_h - m.shape[0])) for m in masks])
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    return imgs_p, masks_p, labels_t


def extract_embeddings_for_sample(model, sample, device, batch_size=64):
    crops, masks = sample["tan_crops"], sample["tan_masks"]
    n = len(crops)
    order = sorted(range(n), key=lambda i: crops[i].shape[0] * crops[i].shape[1])
    embs = np.zeros((n, model.embedding_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk_idx = order[start:start + batch_size]
            imgs = [torch.from_numpy(crops[i].astype(np.float32).transpose(2, 0, 1) / 255.0)
                    for i in chunk_idx]
            msks = [torch.from_numpy(masks[i].astype(np.float32)) for i in chunk_idx]
            imgs_t, masks_t, _ = pad_collate(list(zip(imgs, msks, [0] * len(chunk_idx))))
            imgs_t, masks_t = imgs_t.to(device), masks_t.to(device)
            _, emb = model(imgs_t, masks_t)
            emb = emb.cpu().numpy()
            for j, i in enumerate(chunk_idx):
                embs[i] = emb[j]

    return embs


def train_tan_encoder(samples, tan_train_idx, tan_val_idx, num_classes, device,
                        epochs=30, patience=12, batch_size=16, lr=1e-4, weight_decay=1e-3,
                        verbose=True, class_balance=False):
    train_ds = SuperpixelCropDataset(samples, tan_train_idx, augment=True)
    val_ds = SuperpixelCropDataset(samples, tan_val_idx, augment=False)
    if verbose:
        print(f"  TissueAwareNetwork crops: {len(train_ds)} train, {len(val_ds)} val "
              f"(from {len(tan_train_idx)} train images, {len(tan_val_idx)} val images)")

    train_loader = DataLoader(
        train_ds, batch_sampler=SizeBatchSampler(train_ds, batch_size, shuffle=True),
        collate_fn=pad_collate, num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=SizeBatchSampler(val_ds, batch_size, shuffle=False),
        collate_fn=pad_collate, num_workers=2, pin_memory=True, persistent_workers=True,
    )

    class_weights = None
    if class_balance:
        counts = np.zeros(num_classes, dtype=np.float64)
        for _, _, lbl in train_ds.items:
            counts[lbl] += 1
        present = counts > 0
        w = np.ones(num_classes, dtype=np.float64)
        w[present] = counts.sum() / (present.sum() * counts[present])
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
        if verbose:
            print("  class-balanced loss: superpixels per class "
                  f"{counts.astype(int).tolist()} -> weights {np.round(w, 2).tolist()}")

    model = TissueAwareNetwork(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        for imgs, masks, labels in train_loader:
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs, masks)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)
        train_loss = running_loss / max(n_seen, 1)
        scheduler.step()

        model.eval()
        val_loss, n_val, correct = 0.0, 0, 0
        with torch.no_grad():
            for imgs, masks, labels in val_loader:
                imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
                logits, _ = model(imgs, masks)
                val_loss += F.cross_entropy(logits, labels, weight=class_weights, reduction="sum").item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                n_val += labels.size(0)
        val_loss /= max(n_val, 1)
        val_acc = correct / max(n_val, 1)

        if verbose:
            print(f"  [epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_node_acc={val_acc:.3f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train 4-stage TissueAwareNetwork via MIL pseudo-labels, then extract per-superpixel features")
    parser.add_argument("dataset", help="Path to a .pkl from build_bach_graphs.py --tan-crops")
    parser.add_argument("--out", default="bach_graphs_tan.pkl",
                         help="Output .pkl with tan_features added (tan_crops/tan_masks dropped)")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                         help="Fraction of IMAGES (not superpixels) held out for validation/"
                              "early stopping -- splitting at the superpixel level instead "
                              "would leak correlated tissue across the split")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12,
                         help="Early-stopping patience (default: 12, raised from an earlier "
                              "default of 7 -- a real training run showed val_loss oscillating "
                              "noticeably for several epochs after its initial improvement "
                              "phase before settling, so a low patience risked stopping before "
                              "a genuinely better optimum was found; combined with the LR "
                              "schedule below, this gives training more room to find it.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    samples, label_map = data["samples"], data["label_map"]
    num_classes = len(label_map)

    if "tan_crops" not in samples[0]:
        raise ValueError("This dataset has no 'tan_crops' field -- rebuild it with "
                          "build_bach_graphs.py --tan-crops.")

    labels_all = np.array([s["y"] for s in samples])
    all_idx = np.arange(len(samples))
    train_idx, val_idx = train_test_split(all_idx, test_size=args.val_fraction,
                                           stratify=labels_all, random_state=args.seed)
    print(f"Image-level split: {len(train_idx)} train images, {len(val_idx)} val images")

    val_paths_file = f"{args.out}.val_paths.txt"
    train_paths_file = f"{args.out}.train_paths.txt"
    with open(val_paths_file, "w") as f:
        f.write("\n".join(samples[i]["path"] for i in val_idx) + "\n")
    with open(train_paths_file, "w") as f:
        f.write("\n".join(samples[i]["path"] for i in train_idx) + "\n")
    print(f"Wrote val/train image paths -> {val_paths_file}, {train_paths_file}")

    model = train_tan_encoder(
        samples, train_idx, val_idx, num_classes, device,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, verbose=True,
    )

    print("\nExtracting tan_features for every superpixel in the full dataset "
          "(train + val images, using the trained model)...")
    model.eval()
    for i, s in enumerate(samples):
        s["tan_features"] = extract_embeddings_for_sample(model, s, device)
        del s["tan_crops"]
        del s["tan_masks"]
        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved {len(samples)} samples with tan_features "
          f"(dim={model.embedding_dim}) -> {args.out}")


if __name__ == "__main__":
    main()
