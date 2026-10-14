import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from train_gnn_kfold import train_model, evaluate_model
from train_rf_baseline import pool_sample, STAT_FUNCS


def split_indices(samples):
    train_idx = [i for i, s in enumerate(samples) if s["split"] == "train"]
    val_idx = [i for i, s in enumerate(samples) if s["split"] == "val"]
    test_idx = [i for i, s in enumerate(samples) if s["split"] == "test"]
    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            f"Expected non-empty train/val/test splits, got "
            f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)} -- was this dataset built with "
            f"build_bracs_graphs.py (which preserves sample['split']), not build_bach_graphs.py "
            f"(which has no such field)?"
        )
    return train_idx, val_idx, test_idx


def run_one_seed(samples, label_map, train_idx, val_idx, test_idx, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = len(label_map)
    labels_all = np.array([s["y"] for s in samples])

    X_pooled = np.array([pool_sample(s, None, args.stats) for s in samples])
    clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                  random_state=seed, n_jobs=-1)
    clf.fit(X_pooled[train_idx], labels_all[train_idx])
    rf_preds = clf.predict(X_pooled[test_idx])
    rf_acc = accuracy_score(labels_all[test_idx], rf_preds)
    rf_f1 = f1_score(labels_all[test_idx], rf_preds, average="macro")
    print(f"\n[seed {seed}] Random Forest (test set): accuracy={rf_acc:.3f}  macro-F1={rf_f1:.3f}")

    fold_samples = samples
    if args.mode == "tan":
        from train_tan_extractor import train_tan_encoder, extract_embeddings_for_sample
        print(f"[seed {seed}] Training TissueAwareNetwork ONCE on official train "
              f"(val for early stopping, up to {args.tan_epochs} epochs)...")
        backbone_model = train_tan_encoder(
            samples, train_idx, val_idx, num_classes, device,
            epochs=args.tan_epochs, patience=args.tan_patience, batch_size=args.tan_batch_size,
            lr=args.tan_lr, weight_decay=args.tan_weight_decay, verbose=True,
            class_balance=args.tan_class_balance,
        )
        backbone_model.eval()
        print(f"[seed {seed}] Extracting TAN embeddings for all train+val+test images...")
        fold_samples = list(samples)
        for i in train_idx + val_idx + test_idx:
            emb = extract_embeddings_for_sample(backbone_model, samples[i], device)
            fold_samples[i] = {**samples[i], "cnn_embedding": emb}
        gnn_use_cnn_embeddings = True
    elif args.mode == "scratchcnn":
        from train_scratch_cnn import train_scratch_cnn, extract_embeddings_for_sample
        print(f"[seed {seed}] Training ScratchCNN ONCE on official train "
              f"(val for early stopping, up to {args.scratch_epochs} epochs)...")
        backbone_model = train_scratch_cnn(
            samples, train_idx, val_idx, num_classes, device,
            epochs=args.scratch_epochs, patience=args.scratch_patience,
            batch_size=args.scratch_batch_size, lr=args.scratch_lr,
            weight_decay=args.scratch_weight_decay, verbose=True,
        )
        backbone_model.eval()
        print(f"[seed {seed}] Extracting ScratchCNN embeddings for all train+val+test images...")
        fold_samples = list(samples)
        for i in train_idx + val_idx + test_idx:
            emb = extract_embeddings_for_sample(backbone_model, samples[i], device)
            fold_samples[i] = {**samples[i], "cnn_embedding": emb}
        gnn_use_cnn_embeddings = True
    elif args.mode == "adaptivecnn":
        from train_adaptive_cnn import train_adaptive_cnn, extract_embeddings_for_sample
        if "tan_crops" not in samples[0]:
            raise ValueError("--mode adaptivecnn requires this dataset to have been built "
                              "with --tan-crops (needs tan_crops/tan_masks).")
        print(f"[seed {seed}] Training AdaptiveCNN ONCE on official train "
              f"(native size, batch_size=1, val for early stopping, up to "
              f"{args.adaptive_epochs} epochs)...")
        backbone_model = train_adaptive_cnn(
            samples, train_idx, val_idx, num_classes, device,
            epochs=args.adaptive_epochs, patience=args.adaptive_patience,
            lr=args.adaptive_lr, weight_decay=args.adaptive_weight_decay, verbose=True,
            fill_mode=args.fill_mode,
        )
        backbone_model.eval()
        print(f"[seed {seed}] Extracting AdaptiveCNN embeddings for all train+val+test images...")
        fold_samples = list(samples)
        for i in train_idx + val_idx + test_idx:
            emb = extract_embeddings_for_sample(backbone_model, samples[i], device, fill_mode=args.fill_mode)
            fold_samples[i] = {**samples[i], "cnn_embedding": emb}
        gnn_use_cnn_embeddings = True
    elif args.mode == "tanunmasked":
        from tan_unmasked import train_tan_unmasked, extract_embeddings_for_sample
        if "tan_crops" not in samples[0]:
            raise ValueError("--mode tanunmasked requires this dataset to have been built "
                              "with --tan-crops (needs tan_crops/tan_masks).")
        print(f"[seed {seed}] Training TAN-unmasked ({args.fill_mode} fill) ONCE on official train "
              f"(val for early stopping, up to {args.tan_epochs} epochs)...")
        backbone_model = train_tan_unmasked(
            samples, train_idx, val_idx, num_classes, device,
            epochs=args.tan_epochs, patience=args.tan_patience, batch_size=args.tan_batch_size,
            lr=args.tan_lr, weight_decay=args.tan_weight_decay, verbose=True,
            fill_mode=args.fill_mode,
        )
        backbone_model.eval()
        print(f"[seed {seed}] Extracting TAN-unmasked embeddings for all train+val+test images...")
        fold_samples = list(samples)
        for i in train_idx + val_idx + test_idx:
            emb = extract_embeddings_for_sample(backbone_model, samples[i], device, fill_mode=args.fill_mode)
            fold_samples[i] = {**samples[i], "cnn_embedding": emb}
        gnn_use_cnn_embeddings = True
    else:  # cnn_embeddings
        if "cnn_embedding" not in samples[0]:
            raise ValueError("--mode cnn_embeddings requires this dataset to have been built "
                              "with build_bracs_graphs.py --cnn-embeddings.")
        gnn_use_cnn_embeddings = True

    model_kwargs = dict(hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                         dropout=args.dropout, conv_type=args.conv_type)
    train_kwargs = dict(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
                         patience=args.patience, use_cnn_embeddings=gnn_use_cnn_embeddings,
                         use_tan_features=False)
    print(f"[seed {seed}] Training {args.conv_type.upper()} GNN on official train "
          f"(val for early stopping, up to {args.epochs} epochs)...")
    model, embedding_reducer, tan_reducer, _, epochs_run = train_model(
        fold_samples, train_idx, val_idx, num_classes, model_kwargs, train_kwargs, device
    )
    trues, preds = evaluate_model(model, fold_samples, test_idx, device, args.batch_size,
                                   embedding_reducer=embedding_reducer, tan_reducer=tan_reducer)
    gnn_acc = accuracy_score(trues, preds)
    gnn_f1 = f1_score(trues, preds, average="macro")
    cm = confusion_matrix(trues, preds)

    print(f"[seed {seed}] RF: accuracy={rf_acc:.3f}  macro-F1={rf_f1:.3f}  |  "
          f"{args.conv_type.upper()}+{args.mode} ({epochs_run} epochs): "
          f"accuracy={gnn_acc:.3f}  macro-F1={gnn_f1:.3f}")

    return {
        "seed": seed, "mode": args.mode, "conv_type": args.conv_type,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "gnn_epochs_run": epochs_run,
        "rf_accuracy": rf_acc, "rf_macro_f1": rf_f1,
        "gnn_accuracy": gnn_acc, "gnn_macro_f1": gnn_f1,
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train/val/test evaluation on BRACS's official split (no k-fold CV)")
    parser.add_argument("dataset", help="Path to a .pkl from build_bracs_graphs.py")
    parser.add_argument("--mode", choices=["cnn_embeddings", "tan", "scratchcnn", "adaptivecnn", "tanunmasked"],
                         required=True)
    parser.add_argument("--seed", type=int, default=0,
                         help="Starting seed. With --n-seeds > 1, seeds seed, seed+1, ..., "
                              "seed+n_seeds-1 are all run, each a full fresh pipeline on the "
                              "SAME official split (see module docstring on what this does and "
                              "does not measure).")
    parser.add_argument("--n-seeds", type=int, default=1,
                         help="Number of random seeds to run (default: 1, identical to prior "
                              "behavior). > 1 reports mean +/- std across seeds IN ADDITION to "
                              "each individual seed's result -- see module docstring: this "
                              "measures training-run-to-run noise, NOT sensitivity to the "
                              "train/test partition itself (which never changes here).")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", default="results_bracs_official", help="Output directory")

    parser.add_argument("--conv-type", choices=["sage", "gin", "gat"], default="gat")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--tan-epochs", type=int, default=75)
    parser.add_argument("--tan-patience", type=int, default=15)
    parser.add_argument("--tan-batch-size", type=int, default=16)
    parser.add_argument("--tan-lr", type=float, default=1e-4)
    parser.add_argument("--tan-weight-decay", type=float, default=1e-3)
    parser.add_argument("--tan-class-balance", action="store_true",
                         help="Inverse-frequency class weights (from superpixel counts in the "
                              "training split) in TAN's backbone training loss and in its "
                              "early-stopping validation loss. Off by default (all earlier "
                              "runs). Does not affect the GAT or any other mode.")

    parser.add_argument("--scratch-epochs", type=int, default=75)
    parser.add_argument("--scratch-patience", type=int, default=15)
    parser.add_argument("--scratch-batch-size", type=int, default=32)
    parser.add_argument("--scratch-lr", type=float, default=1e-4)
    parser.add_argument("--scratch-weight-decay", type=float, default=1e-3)

    parser.add_argument("--adaptive-epochs", type=int, default=75)
    parser.add_argument("--adaptive-patience", type=int, default=15)
    parser.add_argument("--adaptive-lr", type=float, default=1e-4)
    parser.add_argument("--adaptive-weight-decay", type=float, default=1e-3)
    parser.add_argument("--fill-mode", choices=["mean", "white"], default="mean",
                         help="How the masked-out background is filled for AdaptiveCNN and "
                              "TAN-unmasked (--mode adaptivecnn / tanunmasked only) -- see "
                              "train_adaptive_cnn.py's mask_fill and tan_unmasked.py.")

    parser.add_argument("--stats", nargs="+", default=["mean", "std", "min", "max", "median"],
                         choices=list(STAT_FUNCS.keys()))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    samples, label_map = data["samples"], data["label_map"]

    train_idx, val_idx, test_idx = split_indices(samples)
    print(f"Official split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test "
          f"(train+val used for training/early-stopping, test evaluated once per seed)")
    if args.n_seeds > 1:
        print(f"Multi-seed run: {args.n_seeds} seeds ({args.seed}..{args.seed + args.n_seeds - 1}), "
              f"SAME official split every time -- only training randomness varies.")

    seed_results = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        result = run_one_seed(samples, label_map, train_idx, val_idx, test_idx, args, device, seed)
        seed_results.append(result)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    if args.n_seeds == 1:
        r = seed_results[0]
        print(f"Random Forest (test set)              : accuracy={r['rf_accuracy']:.3f}  "
              f"macro-F1={r['rf_macro_f1']:.3f}")
        print(f"{args.conv_type.upper()}+{args.mode} (test set, {r['gnn_epochs_run']} epochs run): "
              f"accuracy={r['gnn_accuracy']:.3f}  macro-F1={r['gnn_macro_f1']:.3f}")
        result_out = {**seed_results[0], "label_map": label_map}
    else:
        gnn_accs = np.array([r["gnn_accuracy"] for r in seed_results])
        gnn_f1s = np.array([r["gnn_macro_f1"] for r in seed_results])
        rf_accs = np.array([r["rf_accuracy"] for r in seed_results])
        rf_f1s = np.array([r["rf_macro_f1"] for r in seed_results])
        print(f"Across {args.n_seeds} seeds (same official split, training randomness only):")
        print(f"  Random Forest      : accuracy={rf_accs.mean():.3f}+/-{rf_accs.std():.3f}  "
              f"macro-F1={rf_f1s.mean():.3f}+/-{rf_f1s.std():.3f}")
        print(f"  {args.conv_type.upper()}+{args.mode:<12}: accuracy={gnn_accs.mean():.3f}+/-{gnn_accs.std():.3f}  "
              f"macro-F1={gnn_f1s.mean():.3f}+/-{gnn_f1s.std():.3f}")
        gnn_epochs = np.array([r["gnn_epochs_run"] for r in seed_results])
        result_out = {
            "mode": args.mode, "conv_type": args.conv_type, "n_seeds": args.n_seeds,
            "n_train": seed_results[0]["n_train"], "n_val": seed_results[0]["n_val"],
            "n_test": seed_results[0]["n_test"],
            "gnn_epochs_run": float(gnn_epochs.mean()),  # mean across seeds, for
                                                          # schema-compatibility with
                                                          # gather_bracs_results.py
            "rf_accuracy": float(rf_accs.mean()), "rf_accuracy_std": float(rf_accs.std()),
            "rf_macro_f1": float(rf_f1s.mean()), "rf_macro_f1_std": float(rf_f1s.std()),
            "gnn_accuracy": float(gnn_accs.mean()), "gnn_accuracy_std": float(gnn_accs.std()),
            "gnn_macro_f1": float(gnn_f1s.mean()), "gnn_macro_f1_std": float(gnn_f1s.std()),
            "per_seed_results": seed_results,
            "label_map": label_map,
        }

    with open(out_dir / "result.json", "w") as f:
        json.dump(result_out, f, indent=2)
    print(f"\nSaved result.json -> {out_dir}/")


if __name__ == "__main__":
    main()
