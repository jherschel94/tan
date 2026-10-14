import argparse
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Gather BRACS official-split results into one table")
    parser.add_argument("--results", nargs="+", required=True,
                         help="name=path/to/result.json pairs, e.g. tan=results_bracs_tan/result.json")
    parser.add_argument("--out", default="bracs_summary.csv")
    args = parser.parse_args()

    rows = []
    for item in args.results:
        name, path = item.split("=", 1)
        with open(path) as f:
            r = json.load(f)
        rows.append({
            "encoder": name,
            "rf_accuracy": r["rf_accuracy"], "rf_macro_f1": r["rf_macro_f1"],
            "gnn_accuracy": r["gnn_accuracy"], "gnn_macro_f1": r["gnn_macro_f1"],
            "n_train": r["n_train"], "n_val": r["n_val"], "n_test": r["n_test"],
            "gnn_epochs_run": r["gnn_epochs_run"],
        })

    df = pd.DataFrame(rows).sort_values("gnn_accuracy", ascending=False).reset_index(drop=True)
    print("BRACS official-split results (single train/val/test, no CV -- point estimates only):\n")
    print(df.to_string(index=False))
    df.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
