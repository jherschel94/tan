import argparse
import pickle
import random

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scratch_cnn import ScratchCNN
from train_tan_extractor import train_aug  # reused as-is, works on any HWC array


class FixedCropDataset(Dataset):

    def __init__(self, samples, sample_indices, augment=False):
        self.crops = []
        self.labels = []
        self.augment = augment
        for i in sample_indices:
            s = samples[i]
            for crop in s["scratch_crops"]:
                self.crops.append(crop)
                self.labels.append(s["y"])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        img = self.crops[idx].astype(np.float32) / 255.0
        label = self.labels[idx]
        if self.augment:
            mask = np.ones(img.shape[:2], dtype=np.float32)
            img, _ = train_aug(img, mask)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return torch.from_numpy(img), label


def train_scratch_cnn(samples, train_idx, val_idx, num_classes, device,
                       epochs=30, patience=12, batch_size=32, lr=1e-4, weight_decay=1e-3,
                       verbose=True):
    train_ds = FixedCropDataset(samples, train_idx, augment=True)
    val_ds = FixedCropDataset(samples, val_idx, augment=False)
    if verbose:
        print(f"  ScratchCNN crops: {len(train_ds)} train, {len(val_ds)} val "
              f"(from {len(train_idx)} train images, {len(val_idx)} val images)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
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


def extract_embeddings_for_sample(model, sample, device, batch_size=64):
    crops = sample["scratch_crops"]
    n = len(crops)
    embs = np.zeros((n, model.embedding_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = crops[start:start + batch_size]
            imgs = torch.from_numpy(
                np.ascontiguousarray(batch.astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
            ).to(device)
            _, emb = model(imgs)
            embs[start:start + batch.shape[0]] = emb.cpu().numpy()
    return embs


def main():
    parser = argparse.ArgumentParser(
        description="Train a randomly-initialized ResNet18 via MIL pseudo-labels, then extract per-superpixel features")
    parser.add_argument("dataset", help="Path to a .pkl from build_bach_graphs.py --scratch-crops")
    parser.add_argument("--out", default="bach_graphs_scratch.pkl",
                         help="Output .pkl with cnn_embedding added (scratch_crops dropped)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
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

    if "scratch_crops" not in samples[0]:
        raise ValueError("This dataset has no 'scratch_crops' field -- rebuild it with "
                          "build_bach_graphs.py --scratch-crops.")

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

    model = train_scratch_cnn(
        samples, train_idx, val_idx, num_classes, device,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, verbose=True,
    )

    print("\nExtracting cnn_embedding for every superpixel in the full dataset "
          "(train + val images, using the trained model)...")
    model.eval()
    for i, s in enumerate(samples):
        s["cnn_embedding"] = extract_embeddings_for_sample(model, s, device)
        del s["scratch_crops"]
        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved {len(samples)} samples with cnn_embedding "
          f"(dim={model.embedding_dim}) -> {args.out}")


if __name__ == "__main__":
    main()
