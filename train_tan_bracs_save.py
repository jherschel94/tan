import argparse
import json
import pickle
import random

import numpy as np
import torch

from train_tan_extractor import train_tan_encoder


def official_split(samples):
    train_idx = [i for i, s in enumerate(samples) if s["split"] == "train"]
    val_idx = [i for i, s in enumerate(samples) if s["split"] == "val"]
    test_idx = [i for i, s in enumerate(samples) if s["split"] == "test"]
    if not train_idx or not val_idx:
        raise SystemExit("Need a BRACS .pkl built with build_bracs_graphs.py (sample['split'] missing/empty).")
    return train_idx, val_idx, test_idx


def main():
    p = argparse.ArgumentParser(description="Train one TAN on BRACS and save the weights")
    p.add_argument("dataset", help="BRACS .pkl built with --tan-crops")
    p.add_argument("--out", default="tan_bracs_seed0.pt", help="Weights file (plain state_dict)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--class-balance", action="store_true",
                   help="Class-balanced backbone loss (off = the main TAN of the paper)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    random.seed(args.seed)       # train_aug uses Python's random
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = pickle.load(open(args.dataset, "rb"))
    samples, label_map = data["samples"], data["label_map"]
    if "tan_crops" not in samples[0]:
        raise SystemExit("No tan_crops in this dataset -- build it with --tan-crops.")
    num_classes = len(label_map)
    train_idx, val_idx, test_idx = official_split(samples)
    print(f"Official split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test; "
          f"{num_classes} classes. Training on train, early stopping on val (up to {args.epochs} epochs)...")

    model = train_tan_encoder(
        samples, train_idx, val_idx, num_classes, device,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, verbose=True,
        class_balance=args.class_balance,
    )

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, args.out)
    meta = {
        "seed": args.seed, "num_classes": num_classes, "label_map": label_map,
        "epochs_max": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay, "class_balance": args.class_balance,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "dataset": args.dataset,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved weights -> {args.out}  (+ {args.out}.json)")


if __name__ == "__main__":
    main()
