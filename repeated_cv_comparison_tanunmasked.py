import argparse
import hashlib
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import wilcoxon  # kept for reference/comparison only
from stats_utils import nadeau_bengio_ttest

import torch

from train_gnn_kfold import train_model, evaluate_model
from train_rf_baseline import pool_sample, STAT_FUNCS
from tan_unmasked import train_tan_unmasked, extract_embeddings_for_sample


def main():
    parser = argparse.ArgumentParser(
        description="Fully leakage-free RF vs. GNN comparison with a fresh TAN-unmasked trained per CV fold")
    parser.add_argument("dataset", help="Path to a .pkl from build_bach_graphs.py --tan-crops "
                                         "(must still have raw tan_crops/tan_masks -- NOT the "
                                         "output of train_tan_extractor.py, which strips them)")
    parser.add_argument("--n-splits", type=int, default=5, help="Folds per repeat (default: 5)")
    parser.add_argument("--n-repeats", type=int, default=1,
                         help="Repeats (default: 1 -> 5 total TAN-unmasked trainings at the "
                              "default --n-splits. Raise cautiously -- see module docstring.)")
    parser.add_argument("--seed", type=int, default=0)

    # TAN-unmasked (per-fold, fresh) hyperparameters
    parser.add_argument("--tan-val-fraction", type=float, default=0.15,
                         help="Within each fold's training images, fraction held out for the "
                              "TAN-unmasked's OWN early stopping (default: 0.15)")
    parser.add_argument("--tan-epochs", type=int, default=30)
    parser.add_argument("--tan-patience", type=int, default=12)
    parser.add_argument("--tan-batch-size", type=int, default=16)
    parser.add_argument("--tan-lr", type=float, default=1e-4)
    parser.add_argument("--tan-weight-decay", type=float, default=1e-3)
    parser.add_argument("--fill-mode", choices=["mean", "white"], required=True,
                         help="Fill value for the pixels outside each superpixel's mask. TAN-unmasked "
                              "processes the fill like any other pixel (standard conv/norm/pool/attention).")

    # GNN hyperparameters (same defaults as train_gnn_kfold.py)
    parser.add_argument("--conv-type", choices=["sage", "gin", "gat"], default="gat")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                         help="Within each fold's training images, fraction held out for the "
                              "GNN's OWN early stopping (separate from --tan-val-fraction)")

    # RF hyperparameters
    parser.add_argument("--stats", nargs="+", default=["mean", "std", "min", "max", "median"],
                         choices=list(STAT_FUNCS.keys()))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)

    parser.add_argument("--out", default="repeated_cv_tan_unmasked", help="Output directory")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--restart", action="store_true",
                         help="Ignore any existing paired_results.csv at --out and start over "
                              "from scratch, instead of resuming from where a previous "
                              "(possibly crashed/interrupted) run left off (default behavior).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    samples, label_map = data["samples"], data["label_map"]
    num_classes = len(label_map)

    if "tan_crops" not in samples[0]:
        raise ValueError("This dataset has no 'tan_crops' field -- rebuild it with "
                          "build_bach_graphs.py --tan-crops (and don't run "
                          "train_tan_extractor.py on it first -- that strips tan_crops).")

    labels_all = np.array([s["y"] for s in samples])
    n_total_splits = args.n_splits * args.n_repeats
    print(f"Loaded {len(samples)} images. {args.n_splits}-fold x {args.n_repeats} repeats "
          f"= {n_total_splits} folds, each requiring a FRESH TAN-unmasked training "
          f"(up to {args.tan_epochs} epochs, patience {args.tan_patience}). This will take a while.")

    X_pooled = np.array([pool_sample(s, None, args.stats) for s in samples])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "paired_results.csv"

    rskf = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats,
                                    random_state=args.seed)
    splits = list(rskf.split(np.zeros(len(labels_all)), labels_all))

    model_kwargs = dict(hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                         dropout=args.dropout, conv_type=args.conv_type)

    rows = []
    completed_splits = set()
    if results_csv.exists() and not args.restart:
        existing_df = pd.read_csv(results_csv)
        rows = existing_df.to_dict("records")
        completed_splits = set(existing_df["split"].tolist())
        print(f"[resume] found {len(completed_splits)} already-completed fold(s) in "
              f"{results_csv} -- skipping those, continuing with the rest. "
              f"Use --restart to ignore this and start over.")

    t0 = time.time()
    for split_idx, (train_idx, test_idx) in enumerate(splits):
        if split_idx in completed_splits:
            continue
        repeat_idx, fold_idx = divmod(split_idx, args.n_splits)
        fold_t0 = time.time()

        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                      random_state=args.seed, n_jobs=-1)
        clf.fit(X_pooled[train_idx], labels_all[train_idx])
        rf_preds = clf.predict(X_pooled[test_idx])
        rf_acc = accuracy_score(labels_all[test_idx], rf_preds)
        rf_f1 = f1_score(labels_all[test_idx], rf_preds, average="macro")

        train_labels = labels_all[train_idx]
        tan_train_idx, tan_val_idx = train_test_split(
            train_idx, test_size=args.tan_val_fraction, stratify=train_labels,
            random_state=args.seed + split_idx,
        )
        print(f"\n[split {split_idx+1}/{n_total_splits}] (repeat {repeat_idx}, fold {fold_idx}) "
              f"training fresh TAN-unmasked...")
        tan_model = train_tan_unmasked(
            samples, tan_train_idx, tan_val_idx, num_classes, device,
            epochs=args.tan_epochs, patience=args.tan_patience, batch_size=args.tan_batch_size,
            lr=args.tan_lr, weight_decay=args.tan_weight_decay, verbose=False,
            fill_mode=args.fill_mode,
        )

        tan_model.eval()
        fold_samples = list(samples)  # shallow copy of the list itself
        for i in np.concatenate([train_idx, test_idx]):
            emb = extract_embeddings_for_sample(tan_model, samples[i], device, fill_mode=args.fill_mode)
            fold_samples[i] = {**samples[i], "tan_features": emb}

        gnn_train_idx, gnn_val_idx = train_test_split(
            train_idx, test_size=args.val_fraction, stratify=train_labels,
            random_state=args.seed + split_idx,
        )
        train_kwargs = dict(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
                             patience=args.patience, use_cnn_embeddings=False,
                             use_tan_features=True)
        model, embedding_reducer, tan_reducer, _, epochs_run = train_model(
            fold_samples, gnn_train_idx, gnn_val_idx, num_classes, model_kwargs, train_kwargs, device
        )
        trues, preds = evaluate_model(model, fold_samples, test_idx, device,
                                       args.batch_size, embedding_reducer=embedding_reducer,
                                       tan_reducer=tan_reducer)
        gnn_acc = accuracy_score(trues, preds)
        gnn_f1 = f1_score(trues, preds, average="macro")

        rows.append({
            "split": split_idx, "repeat": repeat_idx, "fold": fold_idx,
            "n_train": len(train_idx), "n_test": len(test_idx), "gnn_epochs_run": epochs_run,
            "rf_accuracy": rf_acc, "rf_macro_f1": rf_f1,
            "gnn_accuracy": gnn_acc, "gnn_macro_f1": gnn_f1,
        
            "test_paths_hash": hashlib.md5(
                "|".join(sorted(samples[i]["path"] for i in test_idx)).encode()
            ).hexdigest(),
        })
        pd.DataFrame(rows).sort_values("split").to_csv(results_csv, index=False)

        fold_elapsed = time.time() - fold_t0
        total_elapsed = time.time() - t0
        print(f"[split {split_idx+1}/{n_total_splits}] RF acc={rf_acc:.3f}  "
              f"{args.conv_type.upper()}+tan-unmasked acc={gnn_acc:.3f}  "
              f"(this fold: {fold_elapsed:.0f}s, total: {total_elapsed:.0f}s)")

    results_df = pd.DataFrame(rows).sort_values("split").reset_index(drop=True)

    print("\n" + "=" * 70)
    print(f"Random Forest       : accuracy {results_df['rf_accuracy'].mean():.3f} +/- "
          f"{results_df['rf_accuracy'].std():.3f}   macro-F1 {results_df['rf_macro_f1'].mean():.3f} "
          f"+/- {results_df['rf_macro_f1'].std():.3f}")
    print(f"{args.conv_type.upper()}+tan-unmasked (clean) : accuracy {results_df['gnn_accuracy'].mean():.3f} +/- "
          f"{results_df['gnn_accuracy'].std():.3f}   macro-F1 {results_df['gnn_macro_f1'].mean():.3f} "
          f"+/- {results_df['gnn_macro_f1'].std():.3f}")

    diff_acc = results_df["gnn_accuracy"] - results_df["rf_accuracy"]
    n_gnn_better = int((diff_acc > 0).sum())
    n_rf_better = int((diff_acc < 0).sum())
    n_tied = int((diff_acc == 0).sum())
    print(f"\nSign count (accuracy): {args.conv_type.upper()}+tan-unmasked better in {n_gnn_better}/{n_total_splits}, "
          f"RF better in {n_rf_better}/{n_total_splits}, tied in {n_tied}/{n_total_splits}")

    for metric in ["accuracy", "macro_f1"]:
        rf_vals = results_df[f"rf_{metric}"].values
        gnn_vals = results_df[f"gnn_{metric}"].values
        diffs = gnn_vals - rf_vals
        if np.all(diffs == 0):
            print(f"\n[{metric}] all paired differences are exactly 0 -- test undefined.")
            continue
        stat, p = nadeau_bengio_ttest(diffs, results_df["n_train"].values, results_df["n_test"].values)
        direction = f"{args.conv_type.upper()}+tan-unmasked" if diffs.mean() > 0 else "RF"
        print(f"\n[{metric}] Nadeau-Bengio corrected t-test: statistic={stat:.2f}, p={p:.4f}  "
              f"(mean diff = {diffs.mean():+.4f}, favoring {direction})")
        if p < 0.05:
            print(f"  -> statistically significant at alpha=0.05.")
        else:
            print(f"  -> NOT statistically significant at alpha=0.05 -- treat any apparent "
                  f"advantage as noise given current sample size.")

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in results_df.iterrows():
        color = "tab:green" if row["gnn_accuracy"] > row["rf_accuracy"] else "tab:red"
        ax.plot([0, 1], [row["rf_accuracy"], row["gnn_accuracy"]], color=color, alpha=0.5, marker="o")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random Forest", f"{args.conv_type.upper()}+tan-unmasked (clean)"])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Paired accuracy across {n_total_splits} folds (fresh TAN-unmasked per fold)\n"
                 f"(green = GNN+tan-unmasked better, red = RF better)")
    plt.tight_layout()
    plt.savefig(out_dir / "paired_comparison.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved paired_results.csv, paired_comparison.png -> {out_dir}/")


if __name__ == "__main__":
    main()
