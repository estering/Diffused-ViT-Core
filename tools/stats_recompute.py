from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

Z975 = 1.959963984540054


def load_folds(root: Path, models: list[str]):
    if not root.is_dir():
        raise SystemExit(
            f"[FATAL] directory not found: {root}\n"
            f"        Point --root at a directory holding fold*/preds.csv.\n"
            f"        Prepare <root>/fold1/preds.csv through <root>/fold5/preds.csv "
            f"with columns\n"
            f"        patient_id, y_true, {', '.join(models)}\n"
            f"        See the module docstring for the exact layout."
        )
    files = sorted(root.glob("fold*/preds.csv"))
    if not files:
        raise SystemExit(
            f"[FATAL] no fold*/preds.csv under {root}\n"
            f"        Expected <root>/fold1/preds.csv, <root>/fold2/preds.csv, ...\n"
            f"        with columns patient_id, y_true, {', '.join(models)}"
        )

    pid, y, fold = [], [], []
    sc = {m: [] for m in models}
    for f in files:
        fold_name = f.parent.name
        with f.open(encoding="utf-8-sig", newline="") as fh:
            rdr = csv.DictReader(fh)
            missing = [c for c in ["patient_id", "y_true"] + models
                       if c not in (rdr.fieldnames or [])]
            if missing:
                raise SystemExit(f"[FATAL] {f} missing columns: {missing}")
            for row in rdr:
                pid.append(row["patient_id"])
                y.append(int(float(row["y_true"])))
                fold.append(fold_name)
                for m in models:
                    sc[m].append(float(row[m]))

    if len(set(pid)) != len(pid):
        raise SystemExit("[FATAL] duplicate patient_id across fold files")

    return pid, np.array(y, int), {m: np.array(v, float) for m, v in sc.items()}, np.array(fold)


def confusion(y, pred):
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    return tp, fn, fp, tn


def ratios(tp, fn, fp, tn):
    n_pos, n_neg = tp + fn, tn + fp
    out = {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n_pos": n_pos, "n_neg": n_neg, "n_pred_pos": tp + fp,
        "errors": fn + fp,
    }
    out["accuracy"] = 100.0 * (tp + tn) / (n_pos + n_neg) if n_pos + n_neg else float("nan")
    out["sensitivity"] = 100.0 * tp / n_pos if n_pos else float("nan")
    out["specificity"] = 100.0 * tn / n_neg if n_neg else float("nan")
    out["precision"] = 100.0 * tp / (tp + fp) if (tp + fp) else float("nan")
    out["npv"] = 100.0 * tn / (tn + fn) if (tn + fn) else float("nan")
    return out


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def wilson(k, n, z=Z975):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return (100.0 * (c - h) / d, 100.0 * (c + h) / d)


def mcnemar_bc(y, pa, pb):
    ca, cb = (pa == y), (pb == y)
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    chi2 = (b - c) ** 2 / n if n else float("nan")
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0)) if n else float("nan")
    return b, c, chi2, p


def _placement(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and v[j + 1] == v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos, r_neg = ranks[:m], ranks[m:]
    v10 = (r_pos - (m + 1) / 2.0) / n
    v01 = 1.0 - (r_neg - (n + 1) / 2.0) / m
    return v10, v01


def delong_paired(y, sa, sb):
    v10a, v01a = _placement(y, sa)
    v10b, v01b = _placement(y, sb)
    m, n = len(v10a), len(v01a)

    def var(x):
        return np.var(x, ddof=1) if len(x) > 1 else 0.0

    s10_aa, s01_aa = var(v10a), var(v01a)
    s10_bb, s01_bb = var(v10b), var(v01b)
    s10_ab = np.cov(v10a, v10b, ddof=1)[0, 1] if m > 1 else 0.0
    s01_ab = np.cov(v01a, v01b, ddof=1)[0, 1] if n > 1 else 0.0

    var_diff = (s10_aa / m + s01_aa / n + s10_bb / m + s01_bb / n
                - 2 * s10_ab / m - 2 * s01_ab / n)
    diff = auc(y, sa) - auc(y, sb)
    if var_diff <= 0:
        return diff, float("nan"), float("nan")
    z = diff / math.sqrt(var_diff)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return diff, z, p


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        run = max(run, adj)
        out[k] = run
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="directory holding fold*/preds.csv")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--models", nargs="+", default=["diffused_vit", "vit", "resnet18"])
    ap.add_argument("--primary", default="diffused_vit")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--score-type", choices=["prob", "logit"], default="prob")
    a = ap.parse_args(argv)

    if not a.root:
        raise SystemExit("[FATAL] --root is required")
    return run(a)


def run(a) -> int:
    pid, y, sc, foldnames = load_folds(a.root, a.models)
    a.out.mkdir(parents=True, exist_ok=True)
    folds = [str(f) for f in sorted(set(foldnames))]

    print(f"[info] {len(y)} patients, folds {folds}, models {a.models}")
    print(f"[info] threshold={a.threshold}  score-type={a.score_type}  "
          f"intervals=Wilson(z={Z975:.6f})  McNemar=uncorrected chi-square")

    rows = []
    for f in folds:
        mask = foldnames == f
        for m in a.models:
            s = sc[m][mask]
            yy = y[mask]
            pred = (s >= a.threshold) if a.score_type == "prob" else (s >= 0.0)
            pred = pred.astype(int)
            r = ratios(*confusion(yy, pred))
            r.update({"fold": f, "model": m, "auc": auc(yy, s)})
            rows.append(r)

    cols = ["fold", "model", "tp", "fn", "fp", "tn", "n_pos", "n_neg",
            "n_pred_pos", "errors", "accuracy", "sensitivity", "specificity",
            "precision", "npv", "auc"]
    with (a.out / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = []
    for m in a.models:
        sub = [r for r in rows if r["model"] == m]
        rec = {"model": m, "folds": len(sub)}
        for k in ["accuracy", "auc", "precision", "sensitivity", "specificity"]:
            v = np.array([r[k] for r in sub], float)
            rec[f"{k}_mean"] = float(v.mean())
            rec[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        summary.append(rec)
        print(f"\n  {m}: n_folds={len(sub)}")
        for k in ["accuracy", "auc", "precision", "sensitivity", "specificity"]:
            dec = 3 if k == "auc" else 1
            print(f"    {k:<12s} {rec[f'{k}_mean']:.{dec}f} +- {rec[f'{k}_sd']:.{dec}f}")

    with (a.out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    pooled = []
    for m in a.models:
        pred = ((sc[m] >= a.threshold) if a.score_type == "prob"
                else (sc[m] >= 0.0)).astype(int)
        r = ratios(*confusion(y, pred))
        r.update({"model": m, "auc": auc(y, sc[m])})
        pooled.append(r)

    with (a.out / "pooled.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pooled[0].keys()))
        w.writeheader()
        w.writerows(pooled)
    p0 = [r for r in pooled if r["model"] == a.primary][0]
    print(f"\n  pooled {a.primary}: acc={p0['accuracy']:.2f}% "
          f"sen={p0['sensitivity']:.2f}% spec={p0['specificity']:.2f}% "
          f"prec={p0['precision']:.2f}% auc={p0['auc']:.3f}")

    mc_rows = []
    for ma, mb in combinations(a.models, 2):
        pa = ((sc[ma] >= a.threshold) if a.score_type == "prob" else (sc[ma] >= 0.0)).astype(int)
        pb = ((sc[mb] >= a.threshold) if a.score_type == "prob" else (sc[mb] >= 0.0)).astype(int)
        b, c, chi2, p = mcnemar_bc(y, pa, pb)
        cap = int(np.sum(pa != y) + np.sum(pb != y))
        mc_rows.append({"model_a": ma, "model_b": mb, "b": b, "c": c,
                        "b_minus_c": b - c, "b_plus_c": b + c,
                        "discordant_ceiling": cap,
                        "both_wrong": (cap - (b + c)) // 2,
                        "chi2": chi2, "p": p})
        print(f"\n  McNemar {ma} vs {mb}: b={b} c={c} b+c={b + c} (ceiling {cap}) "
              f"chi2={chi2:.1f} p={p:.2e}")

    dl_rows, praw = [], {}
    for ma, mb in combinations(a.models, 2):
        diff, z, p = delong_paired(y, sc[ma], sc[mb])
        key = f"{ma}_vs_{mb}"
        praw[key] = p if not math.isnan(p) else 1.0
        dl_rows.append({"pair": key, "delta_auc": diff, "z": z, "p": p})
    hadj = holm(praw)
    for r in dl_rows:
        r["p_holm"] = hadj.get(r["pair"], float("nan"))
    with (a.out / "delong.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "delta_auc", "z", "p", "p_holm"])
        w.writeheader()
        w.writerows(dl_rows)
    for r in dl_rows:
        print(f"  DeLong {r['pair']}: dAUC={r['delta_auc']:+.3f} z={r['z']:.2f} "
              f"p={r['p']:.2e} p_Holm={r['p_holm']:.2e}")

    with (a.out / "mcnemar.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mc_rows[0].keys()))
        w.writeheader()
        w.writerows(mc_rows)

    ci_rows = []
    for r in rows:
        for name, k, n in [("accuracy", r["tp"] + r["tn"], r["n_pos"] + r["n_neg"]),
                           ("sensitivity", r["tp"], r["n_pos"]),
                           ("specificity", r["tn"], r["n_neg"]),
                           ("precision", r["tp"], r["n_pred_pos"])]:
            lo, hi = wilson(k, n)
            ci_rows.append({"fold": r["fold"], "model": r["model"], "metric": name,
                            "k": k, "n": n, "lo": lo, "hi": hi})
    with (a.out / "wilson_ci.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["fold", "model", "metric", "k", "n", "lo", "hi"])
        w.writeheader()
        w.writerows(ci_rows)

    (a.out / "stats.json").write_text(
        json.dumps({"n_patients": int(len(y)), "folds": folds,
                    "threshold": a.threshold, "score_type": a.score_type,
                    "wilson_z": Z975, "fold_metrics": rows,
                    "summary": summary, "pooled": pooled,
                    "mcnemar": mc_rows, "delong": dl_rows},
                   indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\n[ok] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
