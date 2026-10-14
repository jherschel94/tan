import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              confusion_matrix, classification_report)


STAT_FUNCS = {
    "mean": lambda x: np.mean(x, axis=0),
    "std": lambda x: np.std(x, axis=0),
    "min": lambda x: np.min(x, axis=0),
    "max": lambda x: np.max(x, axis=0),
    "median": lambda x: np.median(x, axis=0),
}


def pool_sample(sample, feature_columns, stats):
    x = sample["x"]
    pooled = np.concatenate([STAT_FUNCS[s](x) for s in stats])

    n_nodes = x.shape[0]
    n_directed_edges = sample["edge_index"].shape[1]
    mean_degree = n_directed_edges / n_nodes if n_nodes > 0 else 0.0

    return np.concatenate([pooled, [n_nodes, mean_degree]])


def pool_feature_names(feature_columns, stats):
    names = [f"{stat}_{col}" for stat in stats for col in feature_columns]
    return names + ["n_superpixels", "mean_degree"]


def main():
    parser = argparse.ArgumentParser(description="Random Forest baseline on pooled superpixel features")
    parser.add_argument("dataset", help="Path to the .pkl produced by build_bach_graphs.py")
    parser.add_argument("--k", type=int, default=5, help="Number of CV folds (default: 5, "
                                                           "matching train_gnn_kfold.py's default)")
    parser.add_argument("--stats", nargs="+", default=["mean", "std", "min", "max", "median"],
                         choices=list(STAT_FUNCS.keys()),
                         help="Which summary statistics to pool per feature column")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0,
                         help="Must match train_gnn_kfold.py's --seed for identical folds "
                              "(default: 0, matching that script's default)")
    parser.add_argument("--out", default="results_rf", help="Output directory")
    args = parser.parse_args()

    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    samples, label_map, feature_columns = data["samples"], data["label_map"], data["feature_columns"]
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in range(len(label_map))]

    pooled_names = pool_feature_names(feature_columns, args.stats)
    X = np.array([pool_sample(s, feature_columns, args.stats) for s in samples])
    y = np.array([s["y"] for s in samples])
    print(f"Loaded {len(samples)} graphs -> pooled feature matrix {X.shape} "
          f"({len(pooled_names)} pooled features/image)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    fold_metrics = []
    all_true, all_pred = [], []
    importances = np.zeros(len(pooled_names))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            random_state=args.seed, n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        trues = y[test_idx]

        acc = accuracy_score(trues, preds)
        bal_acc = balanced_accuracy_score(trues, preds)
        macro_f1 = f1_score(trues, preds, average="macro")
        fold_metrics.append({
            "fold": fold, "n_train": len(train_idx), "n_test": len(test_idx),
            "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1,
        })
        all_true.extend(trues.tolist())
        all_pred.extend(preds.tolist())
        importances += clf.feature_importances_
        print(f"[fold {fold}] n_train={len(train_idx)} n_test={len(test_idx)} -> "
              f"acc={acc:.3f} bal_acc={bal_acc:.3f} macro_f1={macro_f1:.3f}")

    importances /= args.k

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Cross-validated results over {args.k} folds:")
    for col in ["accuracy", "balanced_accuracy", "macro_f1"]:
        print(f"  {col}: {metrics_df[col].mean():.3f} +/- {metrics_df[col].std():.3f}")

    print("\nPooled classification report (all folds' test predictions combined):")
    report = classification_report(all_true, all_pred, target_names=class_names, digits=3)
    print(report)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (Random Forest, pooled across {args.k} folds)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    importance_df = pd.DataFrame({"feature": pooled_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(out_dir / "feature_importances.csv", index=False)

    top_n = min(20, len(importance_df))
    fig2, ax2 = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    top = importance_df.head(top_n).iloc[::-1]
    ax2.barh(top["feature"], top["importance"])
    ax2.set_xlabel("Mean decrease in impurity (averaged across folds)")
    ax2.set_title(f"Top {top_n} pooled features")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importances.png", dpi=150, bbox_inches="tight")

    print(f"\nTop 10 most important pooled features:")
    print(importance_df.head(10).to_string(index=False))

    print(f"\nSaved fold_metrics.csv, classification_report.txt, confusion_matrix.png, "
          f"feature_importances.csv/.png -> {out_dir}/")


if __name__ == "__main__":
    main()
