import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tissue_aware_network import (TAN_CHANNELS, NUM_ATTN_HEADS, ATTN_HIDDEN_DIM,
                                   GROUPNORM_GROUPS, TAN_DROPOUT_P, CLASSIFIER_HIDDEN_DIM)
from train_tan_extractor import train_aug, SizeBatchSampler


def fill_outside(img, mask, fill_mode):
    valid = mask > 0
    if fill_mode == "white":
        color = np.ones(3, dtype=np.float32)
    elif fill_mode == "mean":
        color = (img[valid].mean(axis=0) if valid.any() else img.reshape(-1, 3).mean(axis=0)).astype(np.float32)
    else:
        raise ValueError(f"fill_mode must be 'mean' or 'white', got {fill_mode!r}")
    out = img.copy()
    out[~valid] = color
    return out, color


class StandardStage(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_p=TAN_DROPOUT_P):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        num_groups = min(GROUPNORM_GROUPS, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout2d(p=dropout_p)
        bound = 1.0 / math.sqrt(out_channels * in_channels * 9)   # TAN's init scale, see module docstring
        nn.init.uniform_(self.conv.weight, -bound, bound)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.dropout(F.leaky_relu(self.norm(self.conv(x))))


class UnmaskedAttentionPooling(nn.Module):

    def __init__(self, tan_dim, num_heads=NUM_ATTN_HEADS, hidden_dim=ATTN_HIDDEN_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.Sequential(
            nn.Linear(tan_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        scores = self.attn(x_flat).permute(0, 2, 1)          # (B, heads, H*W)
        attn_w = F.softmax(scores, dim=-1)
        return torch.bmm(attn_w, x_flat).reshape(B, self.num_heads * C)


class TANUnmasked(nn.Module):
    """Returns (logits, embedding), like TissueAwareNetwork. `mask` is accepted
    and ignored so the call signature matches TAN's."""

    def __init__(self, num_classes):
        super().__init__()
        ch = TAN_CHANNELS
        self.stage1 = StandardStage(3, ch[0])
        self.stage2 = StandardStage(ch[0], ch[1])
        self.stage3 = StandardStage(ch[1], ch[2])
        self.stage4 = StandardStage(ch[2], ch[3])
        self.pool = nn.MaxPool2d(2)
        self.attn = UnmaskedAttentionPooling(tan_dim=ch[-1])
        attn_out = ch[-1] * NUM_ATTN_HEADS
        self.embedding_dim = attn_out
        self.classifier = nn.Sequential(
            nn.Linear(attn_out, CLASSIFIER_HIDDEN_DIM),
            nn.LayerNorm(CLASSIFIER_HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(CLASSIFIER_HIDDEN_DIM, num_classes),
        )

    def forward(self, x, mask=None):
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.pool(self.stage3(x))
        x = self.pool(self.stage4(x))
        emb = self.attn(x)
        return self.classifier(emb), emb


class UnmaskedCropDataset(Dataset):
    """Same items as train_tan_extractor.SuperpixelCropDataset (native-size raw crop,
    mask, image label), but each item is returned FILLED outside the mask, together
    with its fill color (used to pad batches). Augmentation (training split only) is
    train_aug on the raw crop + mask, applied before the fill."""

    def __init__(self, samples, sample_indices, augment=False, fill_mode="mean"):
        self.items = []
        self.sizes = []
        self.augment = augment
        self.fill_mode = fill_mode
        for i in sample_indices:
            s = samples[i]
            for crop, mask in zip(s["tan_crops"], s["tan_masks"]):
                self.items.append((crop, mask, s["y"]))
                self.sizes.append(crop.shape[0] * crop.shape[1])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        crop, mask, label = self.items[idx]
        img = crop.astype(np.float32) / 255.0
        m = mask.astype(np.float32)
        if self.augment:
            img, m = train_aug(img, m)
        img, color = fill_outside(img, m, self.fill_mode)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(color), label


def pad_collate_fill(batch):
    """Pads every crop (bottom/right, like TAN's pad_collate) to the batch maximum
    with ITS OWN fill color."""
    imgs, colors, labels = zip(*batch)
    max_h = max(im.shape[1] for im in imgs)
    max_w = max(im.shape[2] for im in imgs)
    canvases = []
    for im, c in zip(imgs, colors):
        canvas = c.view(3, 1, 1).expand(3, max_h, max_w).clone()
        canvas[:, :im.shape[1], :im.shape[2]] = im
        canvases.append(canvas)
    return torch.stack(canvases), torch.as_tensor(labels, dtype=torch.long)


def train_tan_unmasked(samples, train_idx, val_idx, num_classes, device,
                       epochs=30, patience=12, batch_size=16, lr=1e-4, weight_decay=1e-3,
                       verbose=True, fill_mode="mean"):
    """Identical loop to train_tan_extractor.train_tan_encoder (AdamW,
    CosineAnnealingLR(T_max=epochs), early stopping on validation loss, best
    state restored) with TANUnmasked and fill-padded batches."""
    train_ds = UnmaskedCropDataset(samples, train_idx, augment=True, fill_mode=fill_mode)
    val_ds = UnmaskedCropDataset(samples, val_idx, augment=False, fill_mode=fill_mode)
    if verbose:
        print(f"  TAN-unmasked ({fill_mode} fill) crops: {len(train_ds)} train, {len(val_ds)} val "
              f"(from {len(train_idx)} train images, {len(val_idx)} val images)")
    train_loader = DataLoader(
        train_ds, batch_sampler=SizeBatchSampler(train_ds, batch_size, shuffle=True),
        collate_fn=pad_collate_fill, num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=SizeBatchSampler(val_ds, batch_size, shuffle=False),
        collate_fn=pad_collate_fill, num_workers=2, pin_memory=True, persistent_workers=True,
    )
    model = TANUnmasked(num_classes=num_classes).to(device)
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


def extract_embeddings_for_sample(model, sample, device, fill_mode="mean", batch_size=64):
    """Embeddings of every superpixel of one image, size-sorted and fill-padded in
    chunks exactly like training (stable sort -> deterministic batching), returned
    in the original superpixel order. fill_mode MUST equal the one used for training."""
    crops, masks = sample["tan_crops"], sample["tan_masks"]
    n = len(crops)
    order = sorted(range(n), key=lambda i: (crops[i].shape[0] * crops[i].shape[1], i))
    embs = np.zeros((n, model.embedding_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = order[start:start + batch_size]
            items = []
            for i in chunk:
                img = crops[i].astype(np.float32) / 255.0
                img, color = fill_outside(img, masks[i].astype(np.float32), fill_mode)
                items.append((torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                              torch.from_numpy(color), 0))
            imgs, _ = pad_collate_fill(items)
            _, emb = model(imgs.to(device))
            emb = emb.cpu().numpy()
            for j, i in enumerate(chunk):
                embs[i] = emb[j]
    return embs


if __name__ == "__main__":
    n_cls = 4
    m = TANUnmasked(n_cls)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"TANUnmasked parameters: {n_params:,}")
    try:
        from tissue_aware_network import TissueAwareNetwork
        n_tan = sum(p.numel() for p in TissueAwareNetwork(n_cls).parameters())
        print(f"TissueAwareNetwork parameters: {n_tan:,}  ->  identical: {n_params == n_tan}")
        assert n_params == n_tan, "parameter counts differ -- the ablation would not be like-for-like"
    except ImportError:
        print("(tissue_aware_network not importable here; expected 408,580 for 4 classes)")

    x = torch.rand(2, 3, 70, 90)
    logits, emb = m.eval()(x)
    print("output shapes:", tuple(logits.shape), tuple(emb.shape), "(expected (2, 4) and (2, 1024))")

    img = np.random.rand(60, 60, 3).astype(np.float32)
    mask = np.zeros((60, 60), np.float32)
    mask[15:45, 10:50] = 1
    outs = {}
    for mode in ("mean", "white"):
        filled, _ = fill_outside(img, mask, mode)
        with torch.no_grad():
            outs[mode] = m(torch.from_numpy(filled.transpose(2, 0, 1)).unsqueeze(0))[1][0]
    diff = (outs["mean"] - outs["white"]).abs().max().item()
    print(f"max |embedding(mean fill) - embedding(white fill)| = {diff:.4f}  (TAN gives exactly 0 here)")
