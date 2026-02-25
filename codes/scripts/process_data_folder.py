#!/usr/bin/env python3
"""
Process all samples in the data/ folder: call SwinUNETR API then export PNGs.
- Reads *_brain_{flair,t1,t1ce,t2}.nii.zip from the folder
- Saves *_api_seg.nii from API
- Converts *_final_seg.nii and *_api_seg.nii to PNG in the same folder
"""

import glob
import gzip
import io
import os
import re
import shutil
import tempfile
import zipfile

import requests

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

API_BASE = "http://216.126.237.218:8086"
PREDICT_URL = f"{API_BASE}/predict"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SUBFOLDER = "data"

LABEL_RGB = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 1, 0),
    4: (0, 0, 1),
}


def find_cases_in_folder(folder: str) -> list[str]:
    """Find case IDs that have all 4 modalities (flair, t1ce, t1, t2)."""
    pattern = os.path.join(folder, "*_brain_flair.nii.zip")
    zips = glob.glob(pattern)
    cases = []
    for z in zips:
        base = os.path.basename(z)
        m = re.match(r"^(.+)_brain_flair\.nii\.zip$", base)
        if m:
            case_id = m.group(1)
            required = [
                os.path.join(folder, f"{case_id}_brain_flair.nii.zip"),
                os.path.join(folder, f"{case_id}_brain_t1ce.nii.zip"),
                os.path.join(folder, f"{case_id}_brain_t1.nii.zip"),
                os.path.join(folder, f"{case_id}_brain_t2.nii.zip"),
            ]
            if all(os.path.isfile(p) for p in required):
                cases.append(case_id)
    return sorted(cases)


def unzip_nii(zip_path: str, out_dir: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.endswith(".nii") and not n.endswith(".nii.gz")]
        if not names:
            raise FileNotFoundError(f"No .nii in {zip_path}")
        z.extract(names[0], out_dir)
    return os.path.join(out_dir, names[0])


def nii_to_gz(nii_path: str) -> str:
    gz_path = nii_path + ".gz"
    with open(nii_path, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return gz_path


def run_api_for_case(data_dir: str, case_id: str) -> str | None:
    """Call API for one case. Returns path to saved api_seg.nii or None on failure."""
    zips = {
        "flair": f"{case_id}_brain_flair.nii.zip",
        "t1ce": f"{case_id}_brain_t1ce.nii.zip",
        "t1": f"{case_id}_brain_t1.nii.zip",
        "t2": f"{case_id}_brain_t2.nii.zip",
    }
    files_to_send = {}
    try:
        with tempfile.TemporaryDirectory(prefix="swinunetr_") as tmp:
            for key, zip_name in zips.items():
                zip_path = os.path.join(data_dir, zip_name)
                nii_path = unzip_nii(zip_path, tmp)
                gz_path = nii_to_gz(nii_path)
                files_to_send[key] = (
                    key,
                    open(gz_path, "rb"),
                    "application/octet-stream",
                )
            try:
                files_for_api = {
                    k: (os.path.basename(v[1].name), v[1], v[2])
                    for k, v in files_to_send.items()
                }
                r = requests.post(PREDICT_URL, files=files_for_api, timeout=300)
                if r.status_code != 200:
                    print(f"  API error {r.status_code}: {r.text[:500]}")
                    return None
                content = r.content
                if content[:2] == b"\x1f\x8b":
                    with gzip.GzipFile(fileobj=io.BytesIO(content), mode="rb") as gz_in:
                        content = gz_in.read()
            finally:
                for v in files_to_send.values():
                    v[1].close()

        out_path = os.path.join(data_dir, f"{case_id}_api_seg.nii")
        with open(out_path, "wb") as f:
            f.write(content)
        return out_path
    except Exception as e:
        print(f"  Error: {e}")
        return None


def label_slice_to_rgb(sl: np.ndarray) -> np.ndarray:
    out = np.zeros((*sl.shape, 3), dtype=np.float32)
    for label, rgb in LABEL_RGB.items():
        out[sl == label] = rgb
    return out


def seg_to_png_for_case(data_dir: str, case_id: str) -> None:
    """Generate PNGs for final_seg and api_seg in data_dir."""
    for suffix in ("final_seg", "api_seg"):
        nii_name = f"{case_id}_{suffix}.nii"
        nii_path = os.path.join(data_dir, nii_name)
        if not os.path.isfile(nii_path):
            print(f"  Skip PNG (not found): {nii_name}")
            continue
        img = nib.load(nii_path)
        data = np.asarray(img.dataobj)
        if data.ndim != 3:
            continue
        data = np.round(data).astype(np.int32)
        data[~np.isin(data, [0, 1, 2, 4])] = 0
        depth = data.shape[2]
        indices = [
            depth // 4,
            depth // 2 - 1,
            depth // 2,
            3 * depth // 4,
        ]
        indices = [max(0, min(i, depth - 1)) for i in indices]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        for ax, idx in zip(axes, indices):
            rgb = label_slice_to_rgb(data[:, :, idx])
            ax.imshow(rgb)
            ax.set_title(f"Slice {idx}")
            ax.axis("off")
        plt.suptitle(nii_name.replace(".nii", ""), fontsize=12)
        plt.tight_layout()
        png_path = os.path.join(data_dir, nii_name.replace(".nii", ".png"))
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {png_path}")


def main():
    data_dir = os.path.join(PROJECT_DIR, DATA_SUBFOLDER)
    if not os.path.isdir(data_dir):
        print(f"Data folder not found: {data_dir}")
        return

    cases = find_cases_in_folder(data_dir)
    if not cases:
        print(f"No complete cases found in {data_dir} (need *_brain_{{flair,t1,t1ce,t2}}.nii.zip)")
        return

    print(f"Found cases: {cases}")

    for case_id in cases:
        print(f"\n--- {case_id} ---")
        print("Calling API...")
        out = run_api_for_case(data_dir, case_id)
        if out:
            print(f"  Saved: {out}")
        print("Converting to PNG...")
        seg_to_png_for_case(data_dir, case_id)

    print("\nDone.")


if __name__ == "__main__":
    main()
