import argparse
import pickle
import random

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scratch_cnn import ScratchCNN


def mask_fill(crop, mask, fill_mode="mean"):
    crop = crop.astype(np.float32)
    if fill_mode == "white":
        fill = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    elif fill_mode == "mean":
        if mask.any():
            fill = crop[mask].mean(axis=0)
        else:
            fill = crop.reshape(-1, crop.shape[-1]).mean(axis=0)
    else:
        raise ValueError(f"Unknown fill_mode {fill_mode!r} -- expected 'mean' or 'white'.")
    out = crop.copy()
    out[~mask] = fill
    return np.clip(out, 0, 255).astype(np.uint8)


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


def adaptive_aug(img, mask):
    if random.random() > 0.5:
        img = img[:, ::-1, :].copy()
        mask = mask[:, ::-1].copy()
    if random.random() > 0.5:
        img = img[::-1, :, :].copy()
        mask = mask[::-1, :].copy()
    if random.random() < 1.0:
        k = random.randint(0, 3)
        img = np.rot90(img, k, axes=(0, 1)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
    img = random_intensity_aug(img, mask)
    return img.astype(np.float32), mask.astype(np.float32)


class NativeSizeCropDataset(Dataset):

    def __init__(self, samples, sample_indices, augment=False, fill_mode="mean"):
        self.items = []  # (crop, mask, label)
        self.augment = augment
        self.fill_mode = fill_mode
        for i in sample_indices:
            s = samples[i]
            for crop, mask in zip(s["tan_crops"], s["tan_masks"]):
                self.items.append((crop, mask, s["y"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        crop, mask, label = self.items[idx]
        filled = mask_fill(crop, mask, self.fill_mode)
        img = filled.astype(np.float32) / 255.0
        m = mask.astype(np.float32)
        if self.augment:
            img, m = adaptive_aug(img, m)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(img), label


def collate_batch_size_1(batch):
    assert len(batch) == 1, "NativeSizeCropDataset must be used with batch_size=1 -- see module docstring."
    img, label = batch[0]
    return img.unsqueeze(0), torch.tensor([label], dtype=torch.long)


def train_adaptive_cnn(samples, train_idx, val_idx, num_classes, device,
                        epochs=30, patience=12, lr=1e-4, weight_decay=1e-3, verbose=True,
                        fill_mode="mean"):
    train_ds = NativeSizeCropDataset(samples, train_idx, augment=True, fill_mode=fill_mode)
    val_ds = NativeSizeCropDataset(samples, val_idx, augment=False, fill_mode=fill_mode)
    if verbose:
        print(f"  AdaptiveCNN crops: {len(train_ds)} train, {len(val_ds)} val "
              f"(from {len(train_idx)} train images, {len(val_idx)} val images, batch_size=1)")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                               collate_fn=collate_batch_size_1,
                               num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_batch_size_1,
                             num_workers=2, pin_memory=True, persistent_workers=True)

    model = ScratchCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)
        train_loss = running_loss / max(n_seen, 1)
        scheduler.step()

        model.eval()
        val_loss, n_val, correct = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs)
                val_loss += F.cross_entropy(logits, labels, reduction="sum").item()
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


def extract_embeddings_for_sample(model, sample, device, fill_mode="mean"):
    crops, masks = sample["tan_crops"], sample["tan_masks"]
    n = len(crops)
    embs = np.zeros((n, model.embedding_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(n):
            filled = mask_fill(crops[i], masks[i], fill_mode)
            img = torch.from_numpy(
                np.ascontiguousarray(filled.astype(np.float32).transpose(2, 0, 1) / 255.0)
            ).unsqueeze(0).to(device)
            _, emb = model(img)
            embs[i] = emb.cpu().numpy()[0]
    return embs


def main():
    parser = argparse.ArgumentParser(
        description="Train a standard ResNet18 (ScratchCNN architecture) at native "
                    "bounding-box size, batch size 1, then extract per-superpixel features")
    parser.add_argument("dataset", help="Path to a .pkl from build_bach_graphs.py/"
                                         "build_bracs_graphs.py --tan-crops")
    parser.add_argument("--out", default="bach_graphs_adaptive.pkl",
                         help="Output .pkl with cnn_embedding added (tan_crops/tan_masks dropped)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fill-mode", choices=["mean", "white"], default="mean",
                         help="How the masked-out background is filled -- see mask_fill's "
                              "docstring. Unlike TAN, AdaptiveCNN's standard convolution has "
                              "no guarantee that this value is excluded from computation, so "
                              "this flag genuinely tests its effect.")
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
                          "build_bach_graphs.py/build_bracs_graphs.py --tan-crops.")

    labels_all = np.array([s["y"] for s in samples])
    all_idx = np.arange(len(samples))
    train_idx, val_idx = train_test_split(all_idx, test_size=args.val_fraction,
                                           stratify=labels_all, random_state=args.seed)
    print(f"Image-level split: {len(train_idx)} train images, {len(val_idx)} val images")

    model = train_adaptive_cnn(
        samples, train_idx, val_idx, num_classes, device,
        epochs=args.epochs, patience=args.patience,
        lr=args.lr, weight_decay=args.weight_decay, verbose=True, fill_mode=args.fill_mode,
    )

    print("\nExtracting cnn_embedding for every superpixel in the full dataset "
          "(train + val images, using the trained model)...")
    model.eval()
    for i, s in enumerate(samples):
        s["cnn_embedding"] = extract_embeddings_for_sample(model, s, device, fill_mode=args.fill_mode)
        del s["tan_crops"]
        del s["tan_masks"]
        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved {len(samples)} samples with cnn_embedding "
          f"(dim={model.embedding_dim}) -> {args.out}")


if __name__ == "__main__":
    main()
