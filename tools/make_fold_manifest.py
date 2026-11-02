from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect(root: Path, classes: list[str], regex: re.Pattern) -> list[dict]:
    rows = []
    for label, cname in enumerate(classes):
        d = root / cname
        if not d.is_dir():
            print(f"[WARN] class directory not found: {d}", file=sys.stderr)
            continue
        for p in sorted(d.glob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            m = regex.search(p.name)
            pid = (
                m.group("pid")
                if (m and "pid" in (m.groupdict() or {}))
                else p.stem.split("_")[0].split("-")[0]
            )
            rows.append(
                {"file": p.name, "path": str(p), "label": label,
                 "class": cname, "patient": pid}
            )
    return rows


def load_map(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            keys = {k.lower().strip(): v.strip() for k, v in r.items()}
            fn = keys.get("filename") or keys.get("file")
            pid = keys.get("patient_id") or keys.get("patient") or keys.get("pid")
            if fn and pid:
                out[fn] = pid
    return out


def stratified_patient_folds(rows, k, seed):
    by_pat = defaultdict(list)
    for r in rows:
        by_pat[r["patient"]].append(r)

    pat_class = {}
    for pid, rs in by_pat.items():
        labs = {r["label"] for r in rs}
        pat_class[pid] = labs.pop() if len(labs) == 1 else -1

    rng = random.Random(seed)
    folds = {}
    for lab in sorted({v for v in pat_class.values()}):
        pats = [p for p, c in pat_class.items() if c == lab]
        rng.shuffle(pats)
        for i, p in enumerate(pats):
            folds[p] = i % k
    return folds, pat_class, by_pat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="directory holding one subdirectory per class")
    ap.add_argument("--out", type=Path, default=Path("out/manifest"))
    ap.add_argument("--private-dir", type=Path, default=Path("private"))
    ap.add_argument("--classes", nargs="+", default=["deep_caries", "pulpitis"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patient-regex", default=r"^(?P<pid>[^_\-\.]+)")
    ap.add_argument("--patient-map", type=Path,
                    help="CSV with columns filename,patient_id")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if not a.root.is_dir():
        raise SystemExit(
            f"[FATAL] directory not found: {a.root}\n"
            f"        Expected one subdirectory per class, for example\n"
            f"        {a.root}/deep_caries/  and  {a.root}/pulpitis/\n"
            f"        each holding images whose filenames begin with a patient "
            f"identifier."
        )

    regex = re.compile(a.patient_regex)
    rows = collect(a.root, a.classes, regex)
    if not rows:
        raise SystemExit(
            f"[FATAL] no images collected under {a.root}\n"
            f"        Looked for class subdirectories {a.classes} containing\n"
            f"        extensions {sorted(IMG_EXT)}. Use --classes to name the "
            f"directories you have."
        )

    if a.patient_map:
        mp = load_map(a.patient_map)
        for r in rows:
            if r["file"] in mp:
                r["patient"] = mp[r["file"]]

    n_pat = len({r["patient"] for r in rows})
    print(f"[info] {len(rows)} images, {n_pat} patients, classes {a.classes}")

    folds, pat_class, by_pat = stratified_patient_folds(rows, a.folds, a.seed)
    mixed = [p for p, c in pat_class.items() if c == -1]
    if mixed:
        print(f"[WARN] {len(mixed)} patients appear under more than one class; "
              f"each is folded by its first file's class", file=sys.stderr)

    rng = random.Random(a.seed)
    pats = sorted(folds)
    rng.shuffle(pats)
    anon = {p: f"P{i + 1:04d}" for i, p in enumerate(pats)}

    per_fold = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per_fold[folds[r["patient"]]][r["label"]] += 1
    print("\n[fold] images per class per fold (patient-disjoint)")
    for f in range(a.folds):
        seg = "  ".join(
            f"{cls}={per_fold[f][i]:4d}" for i, cls in enumerate(a.classes)
        )
        npat = sum(1 for p, k in folds.items() if k == f)
        print(f"  Fold {f + 1}: {seg}   patients={npat}")

    assert len({r["patient"] for r in rows}) == len(folds)

    if a.dry_run:
        print("\n[dry-run] no files written")
        return 0

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "fold_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anon_id", "fold", "label", "class", "n_images"])
        cnt = defaultdict(int)
        for r in rows:
            cnt[r["patient"]] += 1
        for p in pats:
            cls = a.classes[pat_class[p]] if pat_class[p] >= 0 else "mixed"
            w.writerow([anon[p], folds[p] + 1, pat_class[p], cls, cnt[p]])

    splits = {}
    for f in range(a.folds):
        splits[f"fold{f + 1}"] = {
            "val": sorted({anon[r["patient"]] for r in rows
                           if folds[r["patient"]] == f}),
            "train": sorted({anon[r["patient"]] for r in rows
                             if folds[r["patient"]] != f}),
        }
    (a.out / "split.json").write_text(
        json.dumps(
            {"folds": a.folds, "seed": a.seed, "classes": a.classes,
             "patient_disjoint": True, "assignments": splits},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    a.private_dir.mkdir(parents=True, exist_ok=True)
    with (a.private_dir / "mapping_private.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["filename", "patient_id", "anon_id", "fold"])
        for r in sorted(rows, key=lambda x: x["file"]):
            w.writerow([r["file"], r["patient"], anon[r["patient"]],
                        folds[r["patient"]] + 1])

    print(f"\n[ok] public manifest: {a.out / 'fold_manifest.csv'} (anon_id only)")
    print(f"[ok] driver input:    {a.out / 'split.json'}")
    print(f"[ok] private mapping: {a.private_dir / 'mapping_private.csv'} "
          f"(local use only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
