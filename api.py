"""
SwinUNETR Brain Tumor Segmentation API

Upload 4 NIfTI files (flair, t1ce, t1, t2) and get back
a segmentation NIfTI file with labels: 1=TC, 2=WT, 4=ET.

Usage:
    python api.py

API Endpoints:
    POST /predict  — Upload 4 modality files, returns segmentation .nii.gz
    GET  /health   — Health check
"""

import io
import os
import tempfile
from functools import partial

import nibabel as nib
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai import transforms

# ── Config ──────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/sajjad/iabi/model.pt")
DEVICE = os.environ.get("DEVICE", "cuda:1" if torch.cuda.is_available() else "cpu")
FEATURE_SIZE = 48
IN_CHANNELS = 4
OUT_CHANNELS = 3
ROI_SIZE = [128, 128, 128]
INFER_OVERLAP = 0.6
PORT = int(os.environ.get("PORT", "8086"))

# ── App ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SwinUNETR Brain Tumor Segmentation API", version="1.0.0")

# ── Load model at startup ──────────────────────────────────────────────────────
device = torch.device(DEVICE)

model = SwinUNETR(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    feature_size=FEATURE_SIZE,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
)
model_dict = torch.load(MODEL_PATH, weights_only=False, map_location=device)["state_dict"]
model.load_state_dict(model_dict)
model.eval()
model.to(device)
print(f"Model loaded on {device}")

model_inferer = partial(
    sliding_window_inference,
    roi_size=ROI_SIZE,
    sw_batch_size=1,
    predictor=model,
    overlap=INFER_OVERLAP,
)

# ── Preprocessing (same as test.py / data_utils.py) ────────────────────────────
preprocess = transforms.Compose([
    transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    transforms.ToTensord(keys="image"),
])


def load_nifti_from_upload(upload_file: UploadFile):
    """Read an uploaded file into a nibabel image via temp file."""
    suffix = ".nii.gz" if upload_file.filename and upload_file.filename.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload_file.file.read())
        tmp_path = tmp.name
    img = nib.load(tmp_path)
    # Eagerly load data and affine before deleting temp file
    data = img.get_fdata().copy()
    affine = img.affine.copy()
    os.unlink(tmp_path)
    return data, affine


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": os.path.basename(MODEL_PATH),
        "device": str(device),
    }


@app.post("/predict")
async def predict(
    flair: UploadFile = File(..., description="FLAIR modality NIfTI file (.nii.gz)"),
    t1ce: UploadFile = File(..., description="T1CE modality NIfTI file (.nii.gz)"),
    t1: UploadFile = File(..., description="T1 modality NIfTI file (.nii.gz)"),
    t2: UploadFile = File(..., description="T2 modality NIfTI file (.nii.gz)"),
):
    """
    Run brain tumor segmentation on uploaded MRI scans.

    Upload 4 NIfTI files (flair, t1ce, t1, t2).
    Returns a segmentation NIfTI file with labels: 1=TC, 2=WT, 4=ET.
    """
    try:
        # Load all 4 modalities
        arrays = []
        affine = None
        for name, f in [("flair", flair), ("t1ce", t1ce), ("t1", t1), ("t2", t2)]:
            try:
                data, aff = load_nifti_from_upload(f)
                arrays.append(data.astype(np.float32))
                if affine is None:
                    affine = aff
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load {name}: {str(e)}")

        # Stack modalities into (4, H, W, D) array
        stacked = np.stack(arrays, axis=0)

        # Preprocess
        data = {"image": stacked}
        data = preprocess(data)
        image = data["image"].unsqueeze(0).to(device)  # (1, 4, H, W, D)

        # Inference
        with torch.no_grad():
            prob = torch.sigmoid(model_inferer(image))

        # Post-process (same as test.py)
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2  # WT
        seg_out[seg[0] == 1] = 1  # TC
        seg_out[seg[2] == 1] = 4  # ET

        # Save to buffer
        result_img = nib.Nifti1Image(seg_out.astype(np.uint8), affine)
        buffer = io.BytesIO()
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_path = tmp.name
        nib.save(result_img, tmp_path)
        with open(tmp_path, "rb") as f:
            buffer.write(f.read())
        os.unlink(tmp_path)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/gzip",
            headers={"Content-Disposition": "attachment; filename=segmentation.nii.gz"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print(f"Starting API on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
