import argparse
import hashlib
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

from train_gnn_kfold import load_dataset, train_model, evaluate_model
from train_rf_baseline import pool_sample, pool_feature_names, STAT_FUNCS


def main():
    parser = argparse.ArgumentParser(description="Paired repeated-CV comparison: RF vs. GNN")
    parser.add_argument("dataset", help="Path to the .pkl produced by build_bach_graphs.py")
    parser.add_argument("--n-splits", type=int, default=5, help="Folds per repeat (default: 5)")
    parser.add_argument("--n-repeats", type=int, default=5,
                         help="Number of repeats -> total comparisons = n_splits * n_repeats "
                              "(default: 5 -> 25 paired comparisons)")
    parser.add_argument("--seed", type=int, default=0, help="Master seed for split generation")

    parser.add_argument("--conv-type", choices=["sage", "gin", "gat"], default="gat")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                         help="Fraction of each split's training portion held out for GNN "
                              "early stopping (default: 0.15). Not used by RF.")

    parser.add_argument("--stats", nargs="+", default=["mean", "std", "min", "max", "median"],
                         choices=list(STAT_FUNCS.keys()))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)

    parser.add_argument("--use-cnn-embeddings", action="store_true",
                         help="GNN side only: node features come from pretrained-CNN "
                              "embeddings alone (standardized, full dimensionality, no PCA). "
                              "RF baseline is unaffected -- it always pools the 22 handcrafted "
                              "features (see pool_sample), so RF and GNN are not measuring the "
                              "same underlying feature source; interpret the RF column "
                              "accordingly. Exactly one of --use-cnn-embeddings/"
                              "--use-tan-features should be set.")
    parser.add_argument("--use-tan-features", action="store_true",
                         help="GNN side only: node features come from TissueAwareNetwork "
                              "embeddings alone (standardized, full dimensionality, no PCA).")
    parser.add_argument("--pretrain-node-epochs", type=int, default=0,
                         help="GNN side only: if > 0, pretrains the backbone via MIL-style "
                              "node-level pseudo-label classification before graph-level "
                              "training. 0 (default) disables it, identical to prior behavior.")
    parser.add_argument("--pretrain-lr", type=float, default=None,
                         help="Learning rate for node-pretraining (default: same as --lr)")
    parser.add_argument("--pretrain-patience", type=int, default=10)

    parser.add_argument("--out", default="repeated_cv_results", help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for the GNN even if CUDA "
                                                             "is available")
    parser.add_argument("--exclude-file", default=None,
                         help="Text file of sample paths (one per line) to exclude before CV")
    parser.add_argument("--include-only-file", default=None,
                         help="Text file of sample paths (one per line) to KEEP, dropping "
                              "everything else -- e.g. train_tan_extractor.py's val-paths "
                              "output, to sanity-check on only the images a MIL-trained "
                              "feature extractor never received gradient updates on.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    exclude_paths = None
    if args.exclude_file:
        with open(args.exclude_file) as f:
            exclude_paths = [line.strip() for line in f if line.strip()]
    include_only_paths = None
    if args.include_only_file:
        with open(args.include_only_file) as f:
            include_only_paths = [line.strip() for line in f if line.strip()]

    samples, label_map, feature_columns = load_dataset(
        args.dataset, exclude_paths=exclude_paths, include_only_paths=include_only_paths
    )
    labels_all = np.array([s["y"] for s in samples])
    num_classes = len(label_map)
    n_total_splits = args.n_splits * args.n_repeats
    print(f"Loaded {len(samples)} graphs, {len(feature_columns)} features/node, "
          f"{num_classes} classes.")
    print(f"Repeated CV: {args.n_splits}-fold x {args.n_repeats} repeats = "
          f"{n_total_splits} paired comparisons.\n")

    X_pooled = np.array([pool_sample(s, feature_columns, args.stats) for s in samples])

    rskf = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats,
                                    random_state=args.seed)
    splits = list(rskf.split(np.zeros(len(labels_all)), labels_all))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = dict(hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                         dropout=args.dropout, conv_type=args.conv_type)
    train_kwargs = dict(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
                         patience=args.patience, use_cnn_embeddings=args.use_cnn_embeddings,
                         use_tan_features=args.use_tan_features,
                         pretrain_node_epochs=args.pretrain_node_epochs,
                         pretrain_lr=args.pretrain_lr or args.lr,
                         pretrain_patience=args.pretrain_patience)

    rows = []
    t0 = time.time()
    for split_idx, (train_idx, test_idx) in enumerate(splits):
        repeat_idx, fold_idx = divmod(split_idx, args.n_splits)

        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                      random_state=args.seed, n_jobs=-1)
        clf.fit(X_pooled[train_idx], labels_all[train_idx])
        rf_preds = clf.predict(X_pooled[test_idx])
        rf_acc = accuracy_score(labels_all[test_idx], rf_preds)
        rf_f1 = f1_score(labels_all[test_idx], rf_preds, average="macro")

        train_labels = labels_all[train_idx]
        gnn_train_idx, gnn_val_idx = train_test_split(
            train_idx, test_size=args.val_fraction, stratify=train_labels,
            random_state=args.seed + split_idx,
        )
        model, embedding_reducer, tan_reducer, _, epochs_run = train_model(
            samples, gnn_train_idx, gnn_val_idx, num_classes, model_kwargs, train_kwargs, device
        )
        trues, preds = evaluate_model(model, samples, test_idx, device, args.batch_size,
                                       embedding_reducer=embedding_reducer, tan_reducer=tan_reducer)
        gnn_acc = accuracy_score(trues, preds)
        gnn_f1 = f1_score(trues, preds, average="macro")

        rows.append({
            "split": split_idx, "repeat": repeat_idx, "fold": fold_idx,
            "n_train": len(train_idx), "n_test": len(test_idx), "epochs_run": epochs_run,
            "rf_accuracy": rf_acc, "rf_macro_f1": rf_f1,
            "gnn_accuracy": gnn_acc, "gnn_macro_f1": gnn_f1,
            "test_paths_hash": hashlib.md5(
                "|".join(sorted(samples[i]["path"] for i in test_idx)).encode()
            ).hexdigest(),
        })
        elapsed = time.time() - t0
        print(f"[split {split_idx+1}/{n_total_splits}] (repeat {repeat_idx}, fold {fold_idx}) "
              f"RF acc={rf_acc:.3f}  {args.conv_type.upper()} acc={gnn_acc:.3f}  "
              f"({elapsed:.0f}s elapsed)")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "paired_results.csv", index=False)

    print("\n" + "=" * 70)
    print(f"Random Forest : accuracy {results_df['rf_accuracy'].mean():.3f} +/- "
          f"{results_df['rf_accuracy'].std():.3f}   macro-F1 {results_df['rf_macro_f1'].mean():.3f} "
          f"+/- {results_df['rf_macro_f1'].std():.3f}")
    print(f"{args.conv_type.upper():<13} : accuracy {results_df['gnn_accuracy'].mean():.3f} +/- "
          f"{results_df['gnn_accuracy'].std():.3f}   macro-F1 {results_df['gnn_macro_f1'].mean():.3f} "
          f"+/- {results_df['gnn_macro_f1'].std():.3f}")

    diff_acc = results_df["gnn_accuracy"] - results_df["rf_accuracy"]
    n_gnn_better = int((diff_acc > 0).sum())
    n_rf_better = int((diff_acc < 0).sum())
    n_tied = int((diff_acc == 0).sum())
    print(f"\nSign count (accuracy): {args.conv_type.upper()} better in {n_gnn_better}/{n_total_splits} "
          f"splits, RF better in {n_rf_better}/{n_total_splits}, tied in {n_tied}/{n_total_splits}")

    for metric in ["accuracy", "macro_f1"]:
        rf_vals = results_df[f"rf_{metric}"].values
        gnn_vals = results_df[f"gnn_{metric}"].values
        diffs = gnn_vals - rf_vals
        if np.all(diffs == 0):
            print(f"\n[{metric}] all paired differences are exactly 0 -- test undefined.")
            continue
        stat, p = nadeau_bengio_ttest(diffs, results_df["n_train"].values, results_df["n_test"].values)
        direction = args.conv_type.upper() if diffs.mean() > 0 else "RF"
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
    ax.set_xticklabels(["Random Forest", args.conv_type.upper()])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Paired accuracy across {n_total_splits} splits\n"
                 f"(green = {args.conv_type.upper()} better, red = RF better)")
    plt.tight_layout()
    plt.savefig(out_dir / "paired_comparison.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved paired_results.csv, paired_comparison.png -> {out_dir}/")


if __name__ == "__main__":
    main()
