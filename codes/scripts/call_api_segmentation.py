#!/usr/bin/env python3
"""
Call SwinUNETR Brain Tumor Segmentation API with local NIfTI data
and save the result next to the existing segmentation (00000057_final_seg.nii).
"""

import zipfile
import gzip
import shutil
import os
import tempfile
import requests

API_BASE = "http://216.126.237.218:8086"
PREDICT_URL = f"{API_BASE}/predict"

# Your data (zipped .nii)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_ID = "00000057"
ZIPS = {
    "flair": f"{CASE_ID}_brain_flair.nii.zip",
    "t1ce": f"{CASE_ID}_brain_t1ce.nii.zip",
    "t1": f"{CASE_ID}_brain_t1.nii.zip",
    "t2": f"{CASE_ID}_brain_t2.nii.zip",
}
OUTPUT_NEXT_TO = f"{CASE_ID}_final_seg.nii"
OUTPUT_API = f"{CASE_ID}_api_seg.nii"


def unzip_nii(zip_path: str, out_dir: str) -> str:
    """Extract single .nii from zip, return path to .nii."""
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        nii_name = [n for n in names if n.endswith(".nii") and not n.endswith(".nii.gz")]
        if not nii_name:
            raise FileNotFoundError(f"No .nii found in {zip_path}")
        z.extract(nii_name[0], out_dir)
    return os.path.join(out_dir, nii_name[0])


def nii_to_gz(nii_path: str) -> str:
    """Compress .nii to .nii.gz and return path to .nii.gz."""
    gz_path = nii_path + ".gz"
    with open(nii_path, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return gz_path


def main():
    os.chdir(DATA_DIR)

    # Check existing output location
    if not os.path.isfile(OUTPUT_NEXT_TO):
        print(f"Note: reference file '{OUTPUT_NEXT_TO}' not found; API output will still be saved as '{OUTPUT_API}'.")

    with tempfile.TemporaryDirectory(prefix="swinunetr_") as tmp:
        files_to_send = {}

        for key, zip_name in ZIPS.items():
            zip_path = os.path.join(DATA_DIR, zip_name)
            if not os.path.isfile(zip_path):
                raise FileNotFoundError(f"Data file not found: {zip_path}")

            nii_path = unzip_nii(zip_path, tmp)
            # API expects .nii.gz
            gz_path = nii_to_gz(nii_path)
            files_to_send[key] = (
                key,
                open(gz_path, "rb"),
                "application/octet-stream",
            )

        try:
            print("Calling SwinUNETR API...")
            # Use tuple (filename, fileobj, content_type) so server sees .nii.gz
            files_for_api = {
                k: (os.path.basename(v[1].name), v[1], v[2])
                for k, v in files_to_send.items()
            }
            r = requests.post(
                PREDICT_URL,
                files=files_for_api,
                timeout=300,
            )
            if r.status_code != 200:
                print("Response status:", r.status_code)
                print("Response body:", r.text[:2000])
            r.raise_for_status()
        finally:
            for _key, fh, _ in files_to_send.values():
                fh.close()

        out_path = os.path.join(DATA_DIR, OUTPUT_API)
        content = r.content
        # API may return gzipped NIfTI
        if content[:2] == b"\x1f\x8b":
            import io
            with gzip.GzipFile(fileobj=io.BytesIO(content), mode="rb") as gz_in:
                content = gz_in.read()
        with open(out_path, "wb") as f:
            f.write(content)
        print(f"Saved API segmentation: {out_path} (next to {OUTPUT_NEXT_TO})")


if __name__ == "__main__":
    main()
