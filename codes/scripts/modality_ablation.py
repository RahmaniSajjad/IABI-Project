#!/usr/bin/env python3
"""
تحلیل پایداری مدل با حذف/جایگزینی نظام‌مند مدالیتی‌ها.
- حذف تکی: هر مدالیته با Black, Gaussian Noise, Mean باقی‌مانده, Copy یکی دیگر, Interpolation.
- حذف دوگانه: ترکیب‌های دو مدالیته با همان جایگزین‌ها.
خروجی: PNG برای هر سناریو + متریک‌های Dice, HD95, Sensitivity.
"""

import csv
import gzip
import io
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict

import nibabel as nib
import numpy as np
import requests

API_BASE = "http://216.126.237.218:8086"
PREDICT_URL = f"{API_BASE}/predict"
MODALITY_ORDER = ["flair", "t1ce", "t1", "t2"]
LABELS = [1, 2, 4]  # BraTS: TC, WT, ET
LABEL_NAMES = {1: "TC", 2: "WT", 4: "ET"}
LABEL_RGB = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 4: (0, 0, 1)}


def unzip_nii(zip_path: str, out_dir: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.endswith(".nii") and not n.endswith(".nii.gz")]
        if not names:
            raise FileNotFoundError(f"No .nii in {zip_path}")
        z.extract(names[0], out_dir)
    return os.path.join(out_dir, os.path.basename(names[0]))


def load_case_modalities(data_dir: str, case_id: str, tmp_dir: str) -> tuple[dict[str, np.ndarray], object]:
    """Load 4 modalities as numpy arrays; return dict and affine from first."""
    zips = {
        "flair": f"{case_id}_brain_flair.nii.zip",
        "t1ce": f"{case_id}_brain_t1ce.nii.zip",
        "t1": f"{case_id}_brain_t1.nii.zip",
        "t2": f"{case_id}_brain_t2.nii.zip",
    }
    mods = {}
    affine = None
    for key in MODALITY_ORDER:
        zip_path = os.path.join(data_dir, zips[key])
        nii_path = unzip_nii(zip_path, tmp_dir)
        img = nib.load(nii_path)
        mods[key] = np.asarray(img.dataobj, dtype=np.float32)
        if affine is None:
            affine = img.affine
    return mods, affine


# --- جایگزین‌ها ---
def replace_black(vol: np.ndarray) -> np.ndarray:
    return np.zeros_like(vol, dtype=vol.dtype)


def replace_gaussian(vol: np.ndarray, ref_vol: np.ndarray | None = None) -> np.ndarray:
    ref = ref_vol if ref_vol is not None else vol
    r = np.nanmax(ref) - np.nanmin(ref)
    if r <= 0:
        r = 1.0
    std = 0.2 * r
    noise = np.random.default_rng(42).normal(0, std, size=vol.shape).astype(vol.dtype)
    return np.clip(vol + noise, ref.min(), ref.max())


def replace_mean(volumes: list[np.ndarray]) -> np.ndarray:
    return np.mean(volumes, axis=0).astype(volumes[0].dtype)


def replace_copy(other: np.ndarray) -> np.ndarray:
    return np.asarray(other, dtype=other.dtype).copy()


def replace_interpolation(removed_key: str, mods: dict[str, np.ndarray]) -> np.ndarray:
    idx = MODALITY_ORDER.index(removed_key)
    if idx == 0:
        return replace_copy(mods["t1ce"])
    if idx == len(MODALITY_ORDER) - 1:
        return replace_copy(mods[MODALITY_ORDER[idx - 1]])
    left = mods[MODALITY_ORDER[idx - 1]]
    right = mods[MODALITY_ORDER[idx + 1]]
    return (0.5 * left + 0.5 * right).astype(left.dtype)


def build_single_removal_scenarios(mods: dict[str, np.ndarray]) -> list[tuple[str, dict[str, np.ndarray]]]:
    """سناریوهای حذف تکی: هر مدالیته با 5 نوع جایگزین."""
    scenarios = []
    for removed in MODALITY_ORDER:
        remaining = [k for k in MODALITY_ORDER if k != removed]
        others = [mods[k] for k in remaining]
        # Black
        out = {k: replace_copy(mods[k]) for k in MODALITY_ORDER}
        out[removed] = replace_black(mods[removed])
        scenarios.append((f"single_{removed}_black", out))
        # Gaussian
        out = {k: replace_copy(mods[k]) for k in MODALITY_ORDER}
        out[removed] = replace_gaussian(mods[removed])
        scenarios.append((f"single_{removed}_gaussian", out))
        # Mean remaining
        out = {k: replace_copy(mods[k]) for k in MODALITY_ORDER}
        out[removed] = replace_mean(others)
        scenarios.append((f"single_{removed}_mean", out))
        # Copy one other (first remaining)
        out = {k: replace_copy(mods[k]) for k in MODALITY_ORDER}
        out[removed] = replace_copy(mods[remaining[0]])
        scenarios.append((f"single_{removed}_copy_{remaining[0]}", out))
        # Interpolation
        out = {k: replace_copy(mods[k]) for k in MODALITY_ORDER}
        out[removed] = replace_interpolation(removed, mods)
        scenarios.append((f"single_{removed}_interp", out))
    return scenarios


def build_double_removal_scenarios(mods: dict[str, np.ndarray]) -> list[tuple[str, dict[str, np.ndarray]]]:
    """سناریوهای حذف دو مدالیته (جفت‌های بحرانی)."""
    pairs = [
        ("flair", "t1ce"), ("flair", "t1"), ("flair", "t2"),
        ("t1ce", "t1"), ("t1ce", "t2"), ("t1", "t2"),
    ]
    scenarios = []
    for r1, r2 in pairs:
        remaining = [k for k in MODALITY_ORDER if k not in (r1, r2)]
        others = [mods[k] for k in remaining]
        mean_rest = replace_mean(others)
        # Black
        out = {k: replace_copy(mods[k]) for k in MODALITY_ORDER}
        out[r1] = replace_black(mods[r1])
        out[r2] = replace_black(mods[r2])
        scenarios.append((f"double_{r1}_{r2}_black", out))
        # Gaussian
        out = {k: replace_copy(mods[k]) for k in MODALITY_ORDER}
        out[r1] = replace_gaussian(mods[r1])
        out[r2] = replace_gaussian(mods[r2])
        scenarios.append((f"double_{r1}_{r2}_gaussian", out))
        # Mean of remaining
        out = {k: replace_copy(mods[k]) for k in MODALITY_ORDER}
        out[r1] = mean_rest
        out[r2] = replace_copy(mean_rest)
        scenarios.append((f"double_{r1}_{r2}_mean", out))
        # Copy first remaining
        out = {k: replace_copy(mods[k]) for k in MODALITY_ORDER}
        out[r1] = replace_copy(mods[remaining[0]])
        out[r2] = replace_copy(mods[remaining[0]])
        scenarios.append((f"double_{r1}_{r2}_copy_{remaining[0]}", out))
        # Copy second remaining
        out = {k: replace_copy(mods[k]) for k in MODALITY_ORDER}
        out[r1] = replace_copy(mods[remaining[1]])
        out[r2] = replace_copy(mods[remaining[1]])
        scenarios.append((f"double_{r1}_{r2}_copy_{remaining[1]}", out))
    return scenarios


def nii_to_gz(nii_path: str) -> str:
    gz_path = nii_path + ".gz"
    with open(nii_path, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return gz_path


def run_api_with_volumes(
    mods: dict[str, np.ndarray],
    affine: np.ndarray,
    tmp_dir: str,
) -> bytes | None:
    """Write 4 volumes to temp NIfTI.gz and POST to API; return response bytes or None."""
    files_to_send = {}
    try:
        for key in MODALITY_ORDER:
            vol = mods[key]
            nii_path = os.path.join(tmp_dir, f"{key}.nii")
            nib.save(nib.Nifti1Image(vol, affine), nii_path)
            gz_path = nii_to_gz(nii_path)
            files_to_send[key] = (
                key,
                open(gz_path, "rb"),
                "application/octet-stream",
            )
        files_for_api = {
            k: (os.path.basename(v[1].name), v[1], v[2])
            for k, v in files_to_send.items()
        }
        r = requests.post(PREDICT_URL, files=files_for_api, timeout=300)
        if r.status_code != 200:
            print(f"    API {r.status_code}: {r.text[:300]}")
            return None
        content = r.content
        if content[:2] == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=io.BytesIO(content), mode="rb") as gz_in:
                content = gz_in.read()
        return content
    finally:
        for v in files_to_send.values():
            v[1].close()


# --- متریک‌ها ---
def dice_per_label(pred: np.ndarray, gt: np.ndarray, labels: list[int]) -> dict[int, float]:
    pred = np.round(pred).astype(np.int32)
    gt = np.round(gt).astype(np.int32)
    out = {}
    for L in labels:
        a = (pred == L)
        b = (gt == L)
        inter = np.sum(a & b)
        s = np.sum(a) + np.sum(b)
        out[L] = (2.0 * inter / s) if s > 0 else 1.0
    return out


def sensitivity_per_label(pred: np.ndarray, gt: np.ndarray, labels: list[int]) -> dict[int, float]:
    pred = np.round(pred).astype(np.int32)
    gt = np.round(gt).astype(np.int32)
    out = {}
    for L in labels:
        tp = np.sum((pred == L) & (gt == L))
        fn = np.sum((pred != L) & (gt == L))
        den = tp + fn
        out[L] = (tp / den) if den > 0 else 1.0
    return out


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Coordinates of boundary voxels (any face in 6-neighborhood is background)."""
    from scipy import ndimage
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))
    border = mask & (~eroded)
    return np.argwhere(border)


def hausdorff_95(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """HD95 between pred and gt for given label (voxel units)."""
    from scipy.spatial.distance import cdist
    pred = np.round(pred).astype(np.int32)
    gt = np.round(gt).astype(np.int32)
    pa = (pred == label)
    pb = (gt == label)
    if not np.any(pa) or not np.any(pb):
        return 0.0
    sa = surface_voxels(pa)
    sb = surface_voxels(pb)
    if sa.size == 0 or sb.size == 0:
        return 0.0
    # Subsample if huge
    max_pts = 2000
    if len(sa) > max_pts:
        rng = np.random.default_rng(42)
        sa = sa[rng.choice(len(sa), max_pts, replace=False)]
    if len(sb) > max_pts:
        rng = np.random.default_rng(43)
        sb = sb[rng.choice(len(sb), max_pts, replace=False)]
    d = cdist(sa, sb, metric="euclidean")
    d_a_to_b = np.min(d, axis=1)
    d_b_to_a = np.min(d, axis=0)
    h95_a = np.percentile(d_a_to_b, 95)
    h95_b = np.percentile(d_b_to_a, 95)
    return float(max(h95_a, h95_b))


def compute_metrics(pred_nii_path: str, gt_nii_path: str) -> dict:
    """Dice, HD95, Sensitivity per label; return dict for CSV."""
    pred_img = nib.load(pred_nii_path)
    gt_img = nib.load(gt_nii_path)
    pred = np.asarray(pred_img.dataobj)
    gt = np.asarray(gt_img.dataobj)
    pred = np.round(pred).astype(np.int32)
    gt = np.round(gt).astype(np.int32)
    pred[~np.isin(pred, [0, 1, 2, 4])] = 0
    gt[~np.isin(gt, [0, 1, 2, 4])] = 0
    if pred.shape != gt.shape:
        return {"error": "shape_mismatch"}
    dice = dice_per_label(pred, gt, LABELS)
    sens = sensitivity_per_label(pred, gt, LABELS)
    hd95 = {L: hausdorff_95(pred, gt, L) for L in LABELS}
    return {
        "dice_1": dice[1], "dice_2": dice[2], "dice_4": dice[4],
        "sensitivity_1": sens[1], "sensitivity_2": sens[2], "sensitivity_4": sens[4],
        "hd95_1": hd95[1], "hd95_2": hd95[2], "hd95_4": hd95[4],
    }


def seg_to_png(nii_path: str, png_path: str, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = nib.load(nii_path)
    data = np.asarray(img.dataobj)
    data = np.round(data).astype(np.int32)
    data[~np.isin(data, [0, 1, 2, 4])] = 0
    depth = data.shape[2]
    indices = [depth // 4, depth // 2 - 1, depth // 2, 3 * depth // 4]
    indices = [max(0, min(i, depth - 1)) for i in indices]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, idx in zip(axes.flat, indices):
        sl = data[:, :, idx]
        rgb = np.zeros((*sl.shape, 3), dtype=np.float32)
        for label, c in LABEL_RGB.items():
            rgb[sl == label] = c
        ax.imshow(rgb)
        ax.set_title(f"Slice {idx}")
        ax.axis("off")
    plt.suptitle(title or os.path.basename(nii_path).replace(".nii", ""), fontsize=12)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_ablation_for_case(
    data_dir: str,
    case_id: str,
    out_dir: str,
    run_baseline: bool = True,
    run_single: bool = True,
    run_double: bool = True,
) -> list[dict]:
    """Run all scenarios for one case; save PNGs and return list of metric rows."""
    gt_path = os.path.join(data_dir, f"{case_id}_final_seg.nii")
    if not os.path.isfile(gt_path):
        print(f"  Skip {case_id}: no GT {gt_path}")
        return []

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    with tempfile.TemporaryDirectory(prefix="ablation_") as tmp:
        print(f"  Loading modalities...")
        mods, affine = load_case_modalities(data_dir, case_id, tmp)

        def run_scenario(name: str, mod_dict: dict[str, np.ndarray]) -> dict | None:
            content = run_api_with_volumes(mod_dict, affine, tmp)
            if content is None:
                return None
            pred_path = os.path.join(out_dir, f"{case_id}_{name}_pred.nii")
            with open(pred_path, "wb") as f:
                f.write(content)
            png_path = os.path.join(out_dir, f"{case_id}_{name}_pred.png")
            seg_to_png(pred_path, png_path, f"{case_id} {name}")
            met = compute_metrics(pred_path, gt_path)
            if "error" in met:
                return None
            return {"case_id": case_id, "scenario": name, **met}

        # Baseline
        if run_baseline:
            print(f"  Baseline...")
            row = run_scenario("baseline", {k: replace_copy(mods[k]) for k in MODALITY_ORDER})
            if row:
                rows.append(row)

        # Single removal
        if run_single:
            for name, mod_dict in build_single_removal_scenarios(mods):
                print(f"  {name}...")
                row = run_scenario(name, mod_dict)
                if row:
                    rows.append(row)

        # Double removal
        if run_double:
            for name, mod_dict in build_double_removal_scenarios(mods):
                print(f"  {name}...")
                row = run_scenario(name, mod_dict)
                if row:
                    rows.append(row)

    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Modality ablation: replace modalities, call API, compute metrics.")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: project data/ and .)")
    parser.add_argument("--cases", nargs="+", default=["00000042", "00000057"], help="Case IDs")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: ablation_out/<case_id>)")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-single", action="store_true")
    parser.add_argument("--no-double", action="store_true")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Only compute metrics from existing *_pred.nii and write CSV (no API).")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    if args.metrics_only:
        rows = metrics_from_existing_predictions(project_dir)
        if rows:
            _write_metrics_csv(project_dir, rows)
            print(f"Computed metrics for {len(rows)} predictions. Saved ablation_out/metrics.csv")
        else:
            print("No *_pred.nii found in ablation_out/<case_id>/ or GT missing.")
        return

    # دو نمونه: 00000042 در data/ و 00000057 در روت پروژه
    case_dirs = []
    for c in args.cases:
        if args.data_dir:
            case_dirs.append((args.data_dir, c))
        else:
            d42 = os.path.join(project_dir, "data")
            d57 = project_dir
            if c == "00000042" and os.path.isdir(d42):
                case_dirs.append((d42, c))
            elif c == "00000057":
                case_dirs.append((d57, c))
            else:
                case_dirs.append((project_dir, c))

    all_rows = []
    for data_dir, case_id in case_dirs:
        out_dir = args.out_dir or os.path.join(project_dir, "ablation_out", case_id)
        print(f"\n--- {case_id} (from {data_dir}) ---")
        rows = run_ablation_for_case(
            data_dir, case_id, out_dir,
            run_baseline=not args.no_baseline,
            run_single=not args.no_single,
            run_double=not args.no_double,
        )
        all_rows.extend(rows)
        print(f"  Saved PNGs and NIfTIs to {out_dir}")

    # Save metrics CSV (دقیق با ۶ رقم اعشار)
    if all_rows:
        _write_metrics_csv(project_dir, all_rows)
        print(f"\nMetrics saved: {os.path.join(project_dir, 'ablation_out', 'metrics.csv')}")


def _write_metrics_csv(project_dir: str, rows: list[dict]) -> None:
    """Write metrics to CSV with 6 decimal places."""
    csv_path = os.path.join(project_dir, "ablation_out", "metrics.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "case_id", "scenario",
        "dice_TC", "dice_WT", "dice_ET",
        "sensitivity_TC", "sensitivity_WT", "sensitivity_ET",
        "hd95_TC", "hd95_WT", "hd95_ET",
    ]
    def _scenario_order(s: str) -> tuple:
        if s == "baseline":
            return (0, s)
        if s.startswith("single_"):
            return (1, s)
        return (2, s)
    sorted_rows = sorted(rows, key=lambda r: (r.get("case_id", ""), _scenario_order(r.get("scenario", ""))))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted_rows:
            out = {"case_id": r.get("case_id"), "scenario": r.get("scenario")}
            out["dice_TC"] = round(r.get("dice_1", 0), 6)
            out["dice_WT"] = round(r.get("dice_2", 0), 6)
            out["dice_ET"] = round(r.get("dice_4", 0), 6)
            out["sensitivity_TC"] = round(r.get("sensitivity_1", 0), 6)
            out["sensitivity_WT"] = round(r.get("sensitivity_2", 0), 6)
            out["sensitivity_ET"] = round(r.get("sensitivity_4", 0), 6)
            out["hd95_TC"] = round(r.get("hd95_1", 0), 6)
            out["hd95_WT"] = round(r.get("hd95_2", 0), 6)
            out["hd95_ET"] = round(r.get("hd95_4", 0), 6)
            w.writerow(out)
    # جدول خلاصه با ستون‌های حذف‌مدالیته و نوع جایگزین (برای تحلیل حساسیت)
    summary_path = os.path.join(project_dir, "ablation_out", "metrics_summary.csv")
    summary_fields = ["case_id", "scenario", "removed_modality", "replacement_type",
                      "dice_TC", "dice_WT", "dice_ET",
                      "sensitivity_TC", "sensitivity_WT", "sensitivity_ET",
                      "hd95_TC", "hd95_WT", "hd95_ET"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in sorted_rows:
            s = r.get("scenario", "")
            if s == "baseline":
                rem, rep = "none", "baseline"
            elif s.startswith("single_"):
                rest = s[7:]  # after "single_"
                for mod in MODALITY_ORDER:
                    if rest.startswith(mod + "_"):
                        rem = mod
                        rep = rest[len(mod) + 1:]
                        break
                else:
                    rem, rep = "", s
            elif s.startswith("double_"):
                parts = s[7:].split("_")
                if len(parts) >= 3:
                    rem = "_".join(parts[:2])
                    rep = "_".join(parts[2:])
                else:
                    rem, rep = s[7:], "double"
            else:
                rem, rep = "", s
            out = {"case_id": r.get("case_id"), "scenario": s, "removed_modality": rem, "replacement_type": rep}
            for k, v in [("dice_TC", "dice_1"), ("dice_WT", "dice_2"), ("dice_ET", "dice_4"),
                         ("sensitivity_TC", "sensitivity_1"), ("sensitivity_WT", "sensitivity_2"), ("sensitivity_ET", "sensitivity_4"),
                         ("hd95_TC", "hd95_1"), ("hd95_WT", "hd95_2"), ("hd95_ET", "hd95_4")]:
                out[k] = round(r.get(v, 0), 6)
            w.writerow(out)


def metrics_from_existing_predictions(project_dir: str, ablation_out_dir: str | None = None) -> list[dict]:
    """از روی فایل‌های *_pred.nii موجود متریک محاسبه و لیست دیکت برگردان."""
    import glob
    if ablation_out_dir is None:
        ablation_out_dir = os.path.join(project_dir, "ablation_out")
    data_dir_42 = os.path.join(project_dir, "data")
    gt_paths = {
        "00000042": os.path.join(data_dir_42, "00000042_final_seg.nii"),
        "00000057": os.path.join(project_dir, "00000057_final_seg.nii"),
    }
    rows = []
    for case_id, gt_path in gt_paths.items():
        if not os.path.isfile(gt_path):
            continue
        case_dir = os.path.join(ablation_out_dir, case_id)
        if not os.path.isdir(case_dir):
            continue
        for pred_path in glob.glob(os.path.join(case_dir, f"{case_id}_*_pred.nii")):
            base = os.path.basename(pred_path)
            # 00000042_baseline_pred.nii -> scenario = baseline
            rest = base.replace(f"{case_id}_", "").replace("_pred.nii", "")
            scenario = rest
            met = compute_metrics(pred_path, gt_path)
            if "error" in met:
                continue
            rows.append({"case_id": case_id, "scenario": scenario, **met})
    return rows


if __name__ == "__main__":
    main()
