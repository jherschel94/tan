import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              confusion_matrix, classification_report)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GINConv, GATConv, global_mean_pool, global_max_pool


def make_conv(conv_type, in_channels, out_channels):
    if conv_type == "sage":
        return SAGEConv(in_channels, out_channels)
    elif conv_type == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        return GINConv(mlp)
    elif conv_type == "gat":
        return GATConv(in_channels, out_channels, heads=4, concat=False, dropout=0.0)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type!r} (expected 'sage', 'gin', or 'gat')")


class GNNBackbone(nn.Module):

    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.3, conv_type="sage"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        c = in_channels
        for _ in range(num_layers):
            self.convs.append(make_conv(conv_type, c, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            c = hidden_channels
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SuperpixelGNN(nn.Module):

    def __init__(self, in_channels, hidden_channels=64, num_classes=4,
                 num_layers=3, dropout=0.3, conv_type="sage", backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else GNNBackbone(
            in_channels, hidden_channels, num_layers, dropout, conv_type
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = self.backbone(x, edge_index)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.classifier(g)


class NodeClassifierModel(nn.Module):

    def __init__(self, backbone, hidden_channels, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.node_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index):
        x = self.backbone(x, edge_index)
        return self.node_classifier(x)


def load_dataset(path, exclude_paths=None, include_only_paths=None):
    with open(path, "rb") as f:
        data = pickle.load(f)
    samples = data["samples"]
    if exclude_paths:
        exclude_paths = set(exclude_paths)
        before = len(samples)
        samples = [s for s in samples if s["path"] not in exclude_paths]
        print(f"[exclude] dropped {before - len(samples)} samples listed in the exclude file "
              f"({len(samples)} remaining)")
    if include_only_paths:
        include_only_paths = set(include_only_paths)
        before = len(samples)
        samples = [s for s in samples if s["path"] in include_only_paths]
        print(f"[include-only] kept {len(samples)} of {before} samples listed in the "
              f"include-only file")
    return samples, data["label_map"], data["feature_columns"]


def make_pyg_data(sample, embedding_reducer=None, tan_reducer=None):
    x = None
    if embedding_reducer is not None:
        x = embedding_reducer.transform(sample["cnn_embedding"]).astype(np.float32)
    if tan_reducer is not None:
        tf = tan_reducer.transform(sample["tan_features"]).astype(np.float32)
        x = tf if x is None else np.concatenate([x, tf], axis=1)
    if x is None:
        raise ValueError("make_pyg_data: no embedding_reducer/tan_reducer was active -- "
                          "there would be zero node features. Activate exactly one.")
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(sample["edge_index"], dtype=torch.long),
        y=torch.tensor([sample["y"]], dtype=torch.long),
    )


def fit_feature_reducer(samples, indices, field_name):
    from sklearn.preprocessing import StandardScaler

    all_feat = np.concatenate([samples[i][field_name] for i in indices], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_feat)
    return scaler


def fit_embedding_reducer(samples, indices):
    return fit_feature_reducer(samples, indices, "cnn_embedding")


def fit_tan_reducer(samples, indices):
    return fit_feature_reducer(samples, indices, "tan_features")


def make_node_pretrain_data(sample, embedding_reducer=None, tan_reducer=None):
    x = None
    if embedding_reducer is not None:
        x = embedding_reducer.transform(sample["cnn_embedding"]).astype(np.float32)
    if tan_reducer is not None:
        tf = tan_reducer.transform(sample["tan_features"]).astype(np.float32)
        x = tf if x is None else np.concatenate([x, tf], axis=1)
    if x is None:
        raise ValueError("make_node_pretrain_data: no embedding_reducer/tan_reducer was "
                          "active -- there would be zero node features.")
    n_nodes = x.shape[0]
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(sample["edge_index"], dtype=torch.long),
        y=torch.full((n_nodes,), sample["y"], dtype=torch.long),
    )


def pretrain_node_backbone(samples, train_idx, val_idx, in_channels, num_classes, model_kwargs,
                            pretrain_kwargs, device, embedding_reducer, tan_reducer=None):
    backbone = GNNBackbone(in_channels, hidden_channels=model_kwargs["hidden_channels"],
                            num_layers=model_kwargs["num_layers"], dropout=model_kwargs["dropout"],
                            conv_type=model_kwargs["conv_type"]).to(device)
    node_model = NodeClassifierModel(backbone, model_kwargs["hidden_channels"], num_classes,
                                      dropout=model_kwargs["dropout"]).to(device)
    optimizer = torch.optim.Adam(node_model.parameters(), lr=pretrain_kwargs["lr"], weight_decay=1e-4)

    train_data = [make_node_pretrain_data(samples[i], embedding_reducer, tan_reducer) for i in train_idx]
    val_data = [make_node_pretrain_data(samples[i], embedding_reducer, tan_reducer) for i in val_idx]
    train_loader = DataLoader(train_data, batch_size=pretrain_kwargs["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=pretrain_kwargs["batch_size"])

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(pretrain_kwargs["epochs"]):
        node_model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = node_model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()

        node_model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = node_model(batch.x, batch.edge_index)
                val_loss += F.cross_entropy(out, batch.y, reduction="sum").item()
                n_val += batch.y.size(0)
        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in node_model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= pretrain_kwargs["patience"]:
                break

    if best_state is not None:
        node_model.load_state_dict(best_state)

    return node_model.backbone


def train_model(samples, train_idx, val_idx, num_classes, model_kwargs, train_kwargs, device):
    use_embeddings = train_kwargs.get("use_cnn_embeddings", False)
    use_tan = train_kwargs.get("use_tan_features", False)

    embedding_reducer = fit_embedding_reducer(samples, train_idx) if use_embeddings else None
    tan_reducer = fit_tan_reducer(samples, train_idx) if use_tan else None

    train_data = [make_pyg_data(samples[i], embedding_reducer, tan_reducer) for i in train_idx]
    val_data = [make_pyg_data(samples[i], embedding_reducer, tan_reducer) for i in val_idx]
    train_loader = DataLoader(train_data, batch_size=train_kwargs["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=train_kwargs["batch_size"])

    in_channels = train_data[0].x.shape[1]

    pretrain_epochs = train_kwargs.get("pretrain_node_epochs", 0)
    backbone = None
    if pretrain_epochs > 0:
        pretrain_kwargs = dict(
            epochs=pretrain_epochs,
            lr=train_kwargs.get("pretrain_lr", train_kwargs["lr"]),
            batch_size=train_kwargs["batch_size"],
            patience=train_kwargs.get("pretrain_patience", 10),
        )
        backbone = pretrain_node_backbone(
            samples, train_idx, val_idx, in_channels, num_classes, model_kwargs,
            pretrain_kwargs, device, embedding_reducer, tan_reducer,
        )

    model = SuperpixelGNN(in_channels, num_classes=num_classes, backbone=backbone,
                           **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_kwargs["lr"], weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epoch = 0

    for epoch in range(train_kwargs["epochs"]):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                val_loss += F.cross_entropy(out, batch.y, reduction="sum").item()
                n_val += batch.y.size(0)
        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_kwargs["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, embedding_reducer, tan_reducer, best_val_loss, epoch + 1


def evaluate_model(model, samples, indices, device, batch_size=16,
                    embedding_reducer=None, tan_reducer=None):
    data = [make_pyg_data(samples[i], embedding_reducer, tan_reducer) for i in indices]
    loader = DataLoader(data, batch_size=batch_size)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds.extend(out.argmax(dim=1).cpu().tolist())
            trues.extend(batch.y.cpu().tolist())
    return trues, preds


def run_one_fold(samples, train_idx, val_idx, test_idx, num_classes, args, device):
    model_kwargs = dict(hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                         dropout=args.dropout, conv_type=args.conv_type)
    train_kwargs = dict(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
                         patience=args.patience,
                         use_cnn_embeddings=getattr(args, "use_cnn_embeddings", False),
                         use_tan_features=getattr(args, "use_tan_features", False),
                         pretrain_node_epochs=getattr(args, "pretrain_node_epochs", 0),
                         pretrain_lr=getattr(args, "pretrain_lr", None) or args.lr,
                         pretrain_patience=getattr(args, "pretrain_patience", 10))
    model, embedding_reducer, tan_reducer, _, n_epochs_run = train_model(
        samples, train_idx, val_idx, num_classes, model_kwargs, train_kwargs, device
    )
    trues, preds = evaluate_model(model, samples, test_idx, device, args.batch_size,
                                   embedding_reducer=embedding_reducer, tan_reducer=tan_reducer)
    return trues, preds, n_epochs_run


def main():
    parser = argparse.ArgumentParser(description="K-fold GNN training on BACH superpixel graphs")
    parser.add_argument("dataset", help="Path to the .pkl produced by build_bach_graphs.py")
    parser.add_argument("--k", type=int, default=5, help="Number of outer CV folds (default: 5)")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                         help="Fraction of each fold's training portion held out for early "
                              "stopping (default: 0.15)")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per fold (default: 100)")
    parser.add_argument("--patience", type=int, default=15,
                         help="Early-stopping patience in epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--conv-type", choices=["sage", "gin", "gat"], default="sage",
                         help="Message-passing layer type (default: sage). See make_conv's "
                              "docstring for the reasoning behind each option.")
    parser.add_argument("--use-cnn-embeddings", action="store_true",
                         help="Node features come from pretrained-CNN superpixel embeddings "
                              "(from build_bach_graphs.py --cnn-embeddings), standardized, full "
                              "dimensionality, no PCA. Requires the dataset .pkl to contain "
                              "'cnn_embedding' per sample. Exactly one of --use-cnn-embeddings/"
                              "--use-tan-features should be set -- the 22 handcrafted "
                              "morphometric/texture features are never included (this pipeline "
                              "evaluates each node-feature encoder in isolation).")
    parser.add_argument("--use-tan-features", action="store_true",
                         help="Node features come from TissueAwareNetwork embeddings (from "
                              "train_tan_extractor.py), standardized, full dimensionality, no "
                              "PCA. Requires the dataset .pkl to contain 'tan_features' per "
                              "sample.")
    parser.add_argument("--pretrain-node-epochs", type=int, default=0,
                         help="If > 0, pretrains the GNN backbone via MIL-style node-level "
                              "pseudo-label classification (every superpixel inherits its "
                              "image's class label) for up to this many epochs before the "
                              "normal graph-level training loop -- 0 (default) disables "
                              "pretraining entirely, identical to prior behavior. See "
                              "NodeClassifierModel's docstring for the MIL assumption and its "
                              "caveats.")
    parser.add_argument("--pretrain-lr", type=float, default=None,
                         help="Learning rate for the node-pretraining phase (default: same as "
                              "--lr)")
    parser.add_argument("--pretrain-patience", type=int, default=10,
                         help="Early-stopping patience for the node-pretraining phase "
                              "(default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="gnn_results", help="Output directory for results")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--exclude-file", default=None,
                         help="Text file of sample paths (one per line) to exclude before "
                              "running CV -- e.g. to hold out a set of images from this "
                              "evaluation for any reason, without needing a separate dataset.")
    parser.add_argument("--include-only-file", default=None,
                         help="Text file of sample paths (one per line) to KEEP, dropping "
                              "everything else -- use this with train_tan_extractor.py's "
                              "val-paths output to restrict evaluation to images the "
                              "TissueAwareNetwork never received gradient updates on.")
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
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in range(len(label_map))]
    labels_all = np.array([s["y"] for s in samples])
    print(f"Loaded {len(samples)} graphs, {len(feature_columns)} handcrafted features/node "
          f"(NOT used -- node features come only from the active embedding source, see below), "
          f"{len(label_map)} classes: {class_names}")
    if not (args.use_cnn_embeddings or args.use_tan_features):
        print("[error] exactly one of --use-cnn-embeddings/--use-tan-features must be set -- "
              "there would be zero node features otherwise.")
        return
    if args.use_cnn_embeddings:
        if "cnn_embedding" not in samples[0]:
            print("[error] --use-cnn-embeddings was set, but this dataset has no "
                  "'cnn_embedding' field -- rebuild it with build_bach_graphs.py --cnn-embeddings.")
            return
        raw_dim = samples[0]["cnn_embedding"].shape[1]
        print(f"Node features: CNN embeddings only, {raw_dim}-dim, standardized, no PCA")
    if args.use_tan_features:
        if "tan_features" not in samples[0]:
            print("[error] --use-tan-features was set, but this dataset has no "
                  "'tan_features' field -- run train_tan_extractor.py on a dataset built "
                  "with build_bach_graphs.py --tan-crops first.")
            return
        raw_dim = samples[0]["tan_features"].shape[1]
        print(f"Node features: TissueAwareNetwork embeddings only, {raw_dim}-dim, standardized, no PCA")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    fold_metrics = []
    all_true, all_pred = [], []

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels_all)), labels_all)):
        trainval_labels = labels_all[trainval_idx]
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=args.val_fraction, stratify=trainval_labels,
            random_state=args.seed,
        )

        trues, preds, n_epochs_run = run_one_fold(
            samples, train_idx, val_idx, test_idx, len(label_map), args, device
        )

        acc = accuracy_score(trues, preds)
        bal_acc = balanced_accuracy_score(trues, preds)
        macro_f1 = f1_score(trues, preds, average="macro")
        fold_metrics.append({
            "fold": fold, "n_train": len(train_idx), "n_val": len(val_idx),
            "n_test": len(test_idx), "epochs_run": n_epochs_run,
            "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1,
        })
        all_true.extend(trues)
        all_pred.extend(preds)
        print(f"[fold {fold}] n_train={len(train_idx)} n_val={len(val_idx)} n_test={len(test_idx)} "
              f"epochs={n_epochs_run} -> acc={acc:.3f} bal_acc={bal_acc:.3f} macro_f1={macro_f1:.3f}")

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
    ax.set_title(f"Confusion matrix (pooled across {args.k} folds)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved fold_metrics.csv, classification_report.txt, confusion_matrix.png -> {out_dir}/")


if __name__ == "__main__":
    main()
