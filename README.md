# Diffused ViT

Reference code for a Diffused Vision Transformer framework for radiograph-based
classification of deep caries and caries-induced irreversible pulpitis.

The implementation covers preprocessing, ViT feature extraction, feature
refinement, binary classification, training, evaluation, and an ablation entry
point.

This is a reference implementation of the method described in the article. It
reproduces the architecture, the training objective, and the evaluation path.

## Installation

```bash
pip install -r requirements.txt
```

## Data layout

```text
data/
  train/
    deep_caries/
    pulpitis/
  val/
    deep_caries/
    pulpitis/
```

The clinical images are not distributed. They remain under the institutional
data-use agreement described in the article's Data Availability section.

## Usage

```bash
python train.py --data-root data --output-dir outputs
python evaluate.py --data-dir data/val --checkpoint outputs/best.pt
python ablation.py --data-root data
```

`ablation.py` starts the parameter-matched setting with the diffusion
reconstruction loss weighted to zero.

## Timestep sampler

The sampler draws the no-additional-noise endpoint `t = 0` with probability
`ENDPOINT_PROB` and otherwise samples uniformly over `{1, ..., T_MAX}`. Inference
evaluates the endpoint branch.

## Partition manifest

`tools/make_fold_manifest.py` builds a patient-disjoint, class-stratified k-fold
partition from a directory of images and writes a de-identified manifest.

```bash
python tools/make_fold_manifest.py --root data/all --out out/manifest \
       --private-dir private --folds 5 --seed 42
```

The patient identifier defaults to the leading segment of each filename up to
the first underscore, hyphen, or dot, so `1234567_L1.png` maps to patient
`1234567`. Pass `--patient-regex` with a named group `pid` for other
conventions, or `--patient-map` with a two-column CSV headed
`filename,patient_id`. `--dry-run` prints the fold sizes and writes nothing.

The run writes `fold_manifest.csv`, holding one row per patient with
`anon_id, fold, label, class, n_images`; `split.json`, holding the per-fold
train and validation lists of `anon_id` that the training driver consumes; and
`mapping_private.csv`, written to `--private-dir`, which is the only file
linking an `anon_id` back to its source filename.

Anonymous identifiers take the form `P0001` and are assigned from an order
shuffled under `--seed`, so the numbering carries no acquisition order, record
number, or date.

## Metric utilities

`tools/stats_recompute.py` computes fold-level metrics, Wilson intervals, and
paired AUC comparisons from per-fold prediction files.

```bash
python tools/stats_recompute.py --root runs/oof --out out/ \
       --models diffused_vit vit resnet18 --threshold 0.5 --score-type prob
```

It reads `<root>/fold{k}/preds.csv`, one file per fold, each with columns:

```text
patient_id, y_true, <model_1>, <model_2>, ...
```

`y_true` is 0/1. Each model column holds a continuous score: a probability or a
logit, where only the ordering within one model matters and `--score-type`
determines how the threshold is applied. Each `patient_id` appears in exactly one
fold file.

The run writes `fold_metrics.csv` with the per-fold accuracy, sensitivity,
specificity, precision, AUC, and integer confusion counts; `summary.csv` with the
mean and standard deviation across folds; `pooled.csv` with the pooled
out-of-fold metrics and predictive values; `delong.csv` with the pairwise AUC
comparison and its Holm adjustment; `wilson_ci.csv` with the denominator-matched
Wilson intervals per fold; and `stats.json`, which collects the above in one
object.

Denominators, threshold, and interval method are printed and stored alongside the
results.
