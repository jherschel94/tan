import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from stats_utils import nadeau_bengio_ttest, holm_bonferroni



def main():
    parser = argparse.ArgumentParser(description="Direct pairwise encoder comparison, Holm-Bonferroni corrected")
    parser.add_argument("--results", nargs="+", required=True,
                         help="name=path pairs, e.g. tan=results_tan/paired_results.csv "
                              "resnet18=results_resnet18/paired_results.csv ...")
    parser.add_argument("--primary", required=True,
                         help="Which name (from --results) is the reference encoder every "
                              "other one gets compared against, e.g. 'tan'")
    parser.add_argument("--metric", default="gnn_accuracy",
                         choices=["gnn_accuracy", "gnn_macro_f1"],
                         help="Which column to compare (default: gnn_accuracy)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--test", choices=["nb", "wilcoxon"], default="nb",
                         help="Significance test for paired fold comparisons (default: nb -- "
                              "the Nadeau-Bengio corrected resampled t-test, which accounts for "
                              "the correlation between repeated-CV folds induced by overlapping "
                              "training/test sets; a naive test, including Wilcoxon, treats the "
                              "folds as independent and understates the true variance. See "
                              "nadeau_bengio_ttest's docstring. 'wilcoxon' is kept for "
                              "comparison/robustness reporting, not recommended as the primary "
                              "test for repeated k-fold CV results.")
    parser.add_argument("--out", default="final_comparison")
    args = parser.parse_args()

    encoders = {}
    for item in args.results:
        name, path = item.split("=", 1)
        encoders[name] = pd.read_csv(path)

    if args.primary not in encoders:
        raise ValueError(f"--primary {args.primary!r} not found among --results names: "
                          f"{list(encoders.keys())}")

    ref_df = encoders[args.primary]
    has_hash = all("test_paths_hash" in df.columns for df in encoders.values())
    n_folds = len(ref_df)  # set unconditionally -- needed later for the plot title
                            # regardless of which check path (strong/weak) runs below

    if has_hash:
        ref_hashes = ref_df[["split", "test_paths_hash"]].sort_values("split")
        for name, df in encoders.items():
            hashes = df[["split", "test_paths_hash"]].sort_values("split")
            if not hashes.reset_index(drop=True).equals(ref_hashes.reset_index(drop=True)):
                raise ValueError(
                    f"[FATAL] '{name}' has different test_paths_hash values than "
                    f"'{args.primary}' for one or more splits -- these runs used "
                    f"DIFFERENT actual test images for the same split index and cannot "
                    f"be validly paired, even if the fold sizes happen to match. Did they "
                    f"use the same --seed/--n-splits/--n-repeats and the same underlying "
                    f"dataset (same images, same order)?"
                )
        print(f"[STRONG check] test_paths_hash verified identical across {len(encoders)} "
              f"encoders for all {len(ref_hashes)} splits -- these are genuinely the same "
              f"train/test partitions, not just the same sizes.")
    else:
        missing = [n for n, df in encoders.items() if "test_paths_hash" not in df.columns]
        print(f"[WARNING] test_paths_hash missing from: {missing} (from a run predating "
              f"this check, or an older version of repeated_cv_comparison*.py). Falling "
              f"back to a WEAK check (fold sizes only) -- this does NOT prove the same "
              f"images were used, only that the fold sizes match, which StratifiedKFold "
              f"gives you almost regardless of --seed. Re-run the affected encoder(s) with "
              f"the current script version if you need the strong guarantee for a paper.")
        ref_sizes = ref_df[["split", "n_train", "n_test"]].sort_values("split")
        for name, df in encoders.items():
            sizes = df[["split", "n_train", "n_test"]].sort_values("split")
            if not sizes.reset_index(drop=True).equals(ref_sizes.reset_index(drop=True)):
                raise ValueError(
                    f"[FATAL] '{name}' has different fold split/n_train/n_test than "
                    f"'{args.primary}' -- these runs are not on the same folds and cannot be "
                    f"paired. Did they use the same --seed/--n-splits/--n-repeats and the same "
                    f"underlying image set?"
                )
        print(f"[WEAK check only] Fold-size consistency passed across {len(encoders)} "
              f"encoders ({len(ref_sizes)} folds each) -- see warning above.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary table: mean +/- std per encoder
    summary_rows = []
    for name, df in encoders.items():
        summary_rows.append({
            "encoder": name,
            f"{args.metric}_mean": df[args.metric].mean(),
            f"{args.metric}_std": df[args.metric].std(),
            "n_folds": len(df),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(f"{args.metric}_mean", ascending=False)
    summary_df.to_csv(out_dir / "encoder_summary.csv", index=False)
    print("\nSummary (sorted best to worst):")
    print(summary_df.to_string(index=False))

    # Pairwise: primary vs every other encoder, merged on 'split' so
    # comparisons are truly paired (not just row-order-aligned).
    primary_df = encoders[args.primary][["split", args.metric, "n_train", "n_test"]].rename(
        columns={args.metric: "primary_val"}
    )
    other_names = [n for n in encoders if n != args.primary]

    comparison_rows = []
    all_pvalues = []
    for name in other_names:
        other_df = encoders[name][["split", args.metric]].rename(columns={args.metric: "other_val"})
        merged = primary_df.merge(other_df, on="split", how="inner")
        if len(merged) != len(primary_df):
            raise ValueError(f"[FATAL] merge on 'split' between '{args.primary}' and '{name}' "
                              f"dropped rows -- their split indices don't fully match.")

        diffs = merged["primary_val"].values - merged["other_val"].values
        n_primary_better = int((diffs > 0).sum())
        n_other_better = int((diffs < 0).sum())
        n_tied = int((diffs == 0).sum())

        if np.all(diffs == 0):
            stat, p = np.nan, 1.0
        elif args.test == "nb":
            stat, p = nadeau_bengio_ttest(diffs, merged["n_train"].values, merged["n_test"].values)
        else:
            stat, p = wilcoxon(merged["primary_val"], merged["other_val"])

        comparison_rows.append({
            "comparison": f"{args.primary} vs {name}",
            "other_encoder": name,
            "primary_mean": merged["primary_val"].mean(),
            "other_mean": merged["other_val"].mean(),
            "mean_diff": diffs.mean(),
            "primary_better_n": n_primary_better,
            "other_better_n": n_other_better,
            "tied_n": n_tied,
            "test_used": args.test,
            "test_stat": stat,
            "p_value": p,
        })
        all_pvalues.append(p)

    comparison_df = pd.DataFrame(comparison_rows)
    valid = ~comparison_df["p_value"].isna()
    p_holm = np.full(len(comparison_df), np.nan)
    sig_holm = np.full(len(comparison_df), False)
    if valid.any():
        adj, rej = holm_bonferroni(comparison_df.loc[valid, "p_value"].values, alpha=args.alpha)
        p_holm[valid.values] = adj
        sig_holm[valid.values] = rej
    comparison_df["p_holm"] = p_holm
    comparison_df[f"significant_holm_{args.alpha}"] = sig_holm
    comparison_df[f"significant_uncorrected_{args.alpha}"] = comparison_df["p_value"] < args.alpha

    comparison_df.to_csv(out_dir / "pairwise_comparisons.csv", index=False)
    test_label = ("Nadeau-Bengio corrected resampled t-test" if args.test == "nb"
                  else "paired Wilcoxon signed-rank")
    print(f"\n[{args.metric}] '{args.primary}' vs every other encoder "
          f"({test_label}, Holm-Bonferroni corrected across "
          f"{len(other_names)} comparisons):")
    print(comparison_df.to_string(index=False))

    n_sig = int(sig_holm.sum())
    print(f"\n{args.primary!r} is significantly different from {n_sig}/{len(other_names)} "
          f"other encoders after Holm-Bonferroni correction (alpha={args.alpha}).")

    fig, ax = plt.subplots(figsize=(1.6 * len(encoders) + 2, 5))
    names_sorted = summary_df["encoder"].tolist()
    data_for_plot = [encoders[n][args.metric].values for n in names_sorted]
    bp = ax.boxplot(data_for_plot, tick_labels=names_sorted, patch_artist=True)
    for patch, name in zip(bp["boxes"], names_sorted):
        patch.set_facecolor("tab:orange" if name == args.primary else "lightgray")
    for i, name in enumerate(names_sorted):
        y = encoders[name][args.metric].values
        x = np.random.default_rng(0).normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, s=15, color="black", alpha=0.6, zorder=3)
    ax.set_ylabel(args.metric)
    ax.set_title(f"Encoder comparison ({n_folds} paired folds each)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "encoder_comparison.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved encoder_summary.csv, pairwise_comparisons.csv, encoder_comparison.png -> {out_dir}/")


if __name__ == "__main__":
    main()
