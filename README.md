# TAN: A Size- and Form-Adaptive Masked-Convolution Encoder for Tissue Graphs

Code for the paper *TAN: A Size- and Form-Adaptive Masked-Convolution Encoder for Tissue Graphs* (submitted to ISBI 2027).

**TAN (Tissue-Aware Network)** is a compact convolutional encoder (0.41M parameters) for the nodes of superpixel tissue graphs. It processes each superpixel at its native shape and size and excludes everything outside the superpixel from every operation (masked convolution, masked GroupNorm, masked max-pooling, masked attention pooling), so its embedding does not depend on how the background is filled.

The repository contains the complete comparison pipeline on **BACH** and **BRACS**: SLIC superpixel graphs, node encoders with an identical GAT classifier and identical folds, a TAN-unmasked ablation, corrected significance tests, and FLOP and parameter counts.

| Encoder | Description |
|---|---|
| `TAN` | masked convolution, trained from scratch (this work) |
| `TAN-unmasked` | same architecture and training, standard operations on a fill-padded crop (ablation) |
| `ResNet18-scratch` | ResNet18 trained from scratch on a 128x128 masked crop |
| `AdaptiveCNN` | the same ResNet18 at native bounding-box size, batch size 1 (BACH only) |
| `ResNet18-ImageNet`, `CTransPath`, `UNI` | frozen pretrained encoders |

## Repository layout

| File | Purpose |
|---|---|
| `tissue_aware_network.py` | TAN (SolidConv, masked GroupNorm, masked max-pooling, masked attention pooling) |
| `tan_unmasked.py` | TAN-unmasked ablation (model, fill-padded data, training, extraction) |
| `train_tan_extractor.py` | TAN training, augmentation, size-sorted batching, embedding extraction |
| `scratch_cnn.py`, `train_scratch_cnn.py` | ResNet18-scratch (model and trainer) |
| `train_adaptive_cnn.py` | AdaptiveCNN |
| `foundation_encoders.py` | frozen ResNet18-ImageNet, UNI, CTransPath |
| `main.py` | SLIC graph construction, crop extraction, embedding utilities |
| `build_bach_graphs.py`, `build_bracs_graphs.py` | build the graph pickles for each dataset |
| `train_gnn_kfold.py`, `train_rf_baseline.py` | GAT classifier and Random Forest baseline |
| `repeated_cv_comparison*.py` | BACH 5x3 repeated cross-validation, one script per encoder family |
| `train_eval_bracs_official_split.py` | BRACS official-split evaluation (`--mode cnn_embeddings, tan, scratchcnn, adaptivecnn, tanunmasked`) |
| `compare_all_encoders.py`, `stats_utils.py` | paired comparison, Nadeau-Bengio corrected t-test, Holm-Bonferroni |
| `gather_bracs_results.py` | collects BRACS `result.json` files into one table |
| `count_flops.py`, `count_model_parameters.py` | FLOPs per superpixel and parameter counts |
| `check_node_survival.py` | how often a superpixel has no fully valid 3x3 neighborhood at some stage |
| `bracs_class_distribution.py` | BRACS class counts and the effect of the size filter |
| `train_tan_bracs_save.py`, `make_tan_figure.py` | train and save one TAN; stage visualization, background-invariance and batch-padding tests |

## Setup
## Setup

```bash
conda create -n tan python=3.9 && conda activate tan
pip install -r requirements.txt
```
`requirements.txt` is a frozen snapshot of the environment the experiments were run in: Python 3.9.25, PyTorch 2.8.0 (CUDA 12.8), torchvision 0.23.0, PyTorch Geometric 2.6.1, timm 1.0.29, on a single NVIDIA RTX 5090 (32 GB) with 61 GB RAM. The graph pickles are large (several GB for BACH, more for BRACS); plan for at least 48 GB RAM.

## Data and pretrained weights (not included)

Neither the datasets nor the weights are part of this repository. Obtain them from the original sources and follow their terms.

- **BACH** (ICIAR 2018 Grand Challenge): https://iciar2018-challenge.grand-challenge.org/Dataset/
- **BRACS** (RoI images, `BRACS_RoI/latest_version`): https://www.bracs.icar.cnr.it/
- **UNI** is gated on Hugging Face (`MahmoodLab/UNI`): request access yourself and authenticate (`hf auth login` or the `HF_TOKEN` environment variable).
- **CTransPath**: download `torchscript_model.pt` from `huggingface.co/kaczmarj/CTransPath` and pass its path with `--embedding-checkpoint`.
- **ResNet18-ImageNet** weights are downloaded by torchvision.

## Reproducing the experiments

Paths are placeholders: `BACH=<path>/ICIAR2018_BACH_Challenge`, `BRACS=<path>/BRACS_RoI/latest_version`.

### 1. Build the graphs

```bash
# BACH, 800 superpixels per image at native resolution (mean fill is the default)
python build_bach_graphs.py $BACH --out bach_graphs_tan_raw.pkl     --n-superpixels 800 --tan-crops     --workers 8
python build_bach_graphs.py $BACH --out bach_graphs_scratch_raw.pkl --n-superpixels 800 --scratch-crops --workers 8
python build_bach_graphs.py $BACH --out bach_graphs_resnet18.pkl    --n-superpixels 800 --cnn-embeddings --embedding-model resnet18 --workers 8
python build_bach_graphs.py $BACH --out bach_graphs_uni.pkl         --n-superpixels 800 --cnn-embeddings --embedding-model uni --workers 8
python build_bach_graphs.py $BACH --out bach_graphs_ctranspath.pkl  --n-superpixels 800 --cnn-embeddings --embedding-model ctranspath \
        --embedding-checkpoint ./ctranspath_weights/torchscript_model.pt --workers 8
# white fill for the baselines: add  --fill-mode white  (TAN does not use a fill value)

# BRACS, 250 superpixels per 1000x1000 center crop
python build_bracs_graphs.py $BRACS --out bracs_graphs_tan_raw.pkl     --n-superpixels 250 --tan-crops --workers 8
python build_bracs_graphs.py $BRACS --out bracs_graphs_scratch_raw.pkl --n-superpixels 250 --scratch-crops --scratch-crop-size 128 --workers 2
# frozen encoders: --cnn-embeddings --embedding-model {resnet18,uni,ctranspath}, as for BACH
```

### 2. BACH: 5-fold cross-validation, repeated 3 times (15 folds)

```bash
python repeated_cv_comparison_tan.py         bach_graphs_tan_raw.pkl     --n-splits 5 --n-repeats 3 --tan-epochs 75 --tan-patience 15 --out results_tan/
python repeated_cv_comparison_tanunmasked.py bach_graphs_tan_raw.pkl     --n-splits 5 --n-repeats 3 --tan-epochs 75 --tan-patience 15 --fill-mode mean --out results_tanunmasked_mean/
python repeated_cv_comparison_scratchcnn.py  bach_graphs_scratch_raw.pkl --n-splits 5 --n-repeats 3 --scratch-epochs 75 --scratch-patience 15 --out results_scratch/
python repeated_cv_comparison_adaptivecnn.py bach_graphs_tan_raw.pkl     --n-splits 5 --n-repeats 3 --adaptive-epochs 75 --adaptive-patience 15 --out results_adaptivecnn/
python repeated_cv_comparison.py             bach_graphs_uni.pkl         --n-splits 5 --n-repeats 3 --conv-type gat --use-cnn-embeddings --out results_uni/
# likewise for resnet18 and ctranspath; --fill-mode white for the white-fill runs

python compare_all_encoders.py \
    --results tan=results_tan/paired_results.csv tan_unmasked=results_tanunmasked_mean/paired_results.csv \
              scratch=results_scratch/paired_results.csv uni=results_uni/paired_results.csv \
    --primary tan --metric gnn_accuracy --test nb --out comparison/
```
`compare_all_encoders.py` checks that every encoder was evaluated on identical train/test partitions and applies the Nadeau-Bengio corrected resampled t-test with Holm-Bonferroni correction.

### 3. BRACS: official split, 5 seeds

```bash
python train_eval_bracs_official_split.py bracs_graphs_tan_raw.pkl --mode tan --tan-epochs 75 --tan-patience 15 --n-seeds 5 --out results_bracs_tan/
python train_eval_bracs_official_split.py bracs_graphs_tan_raw.pkl --mode tanunmasked --fill-mode mean --tan-epochs 75 --tan-patience 15 --n-seeds 5 --out results_bracs_tanunmasked_mean/
python train_eval_bracs_official_split.py bracs_graphs_scratch_raw.pkl --mode scratchcnn --scratch-epochs 75 --scratch-patience 15 --n-seeds 5 --out results_bracs_scratch/
python train_eval_bracs_official_split.py bracs_graphs_uni.pkl --mode cnn_embeddings --conv-type gat --n-seeds 5 --out results_bracs_uni/
python gather_bracs_results.py --results tan=results_bracs_tan/result.json uni=results_bracs_uni/result.json --out bracs_summary.csv
```
BRACS has a single fixed split, so results are reported as mean and standard deviation over seeds, without significance tests.

### 4. Measurements and figures

```bash
python count_flops.py bach_graphs_tan_raw.pkl --n-crops 300 --num-classes 4 --scratch-crop-size 128 \
        --ctranspath-checkpoint ./ctranspath_weights/torchscript_model.pt     # add --skip-uni to skip UNI
python count_model_parameters.py --num-classes 4
python check_node_survival.py bach_graphs_tan_raw.pkl --stages 3 4

python train_tan_bracs_save.py bracs_graphs_tan_raw.pkl --seed 0 --out tan_bracs_seed0.pt
python make_tan_figure.py bracs_graphs_tan_raw.pkl --checkpoint tan_bracs_seed0.pt --num-classes 7 \
        --sample-idx 0 --superpixel-idx <idx> --out tan_stages.pdf            # stage visualization
python make_tan_figure.py bracs_graphs_tan_raw.pkl --checkpoint tan_bracs_seed0.pt --num-classes 7 \
        --invariance 100 --padding-test 100 --out /tmp/inv.pdf               # background-invariance and padding tests
```
Run any script with `--help` for its options.

## AI assistance

Claude (Anthropic) was used as an assistant under the authors' direction: it implemented most of the code from the authors' specifications (the authors ran, checked and corrected it) and wrote and revised wording for the paper from the authors' instructions, results and edits.

## License

See `LICENSE`.


