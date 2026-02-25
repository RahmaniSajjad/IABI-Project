#!/usr/bin/env python3
"""
Convert the two segmentation NIfTI outputs to PNG images.
Labels (BraTS): 1=TC (Tumor Core), 2=WT (Whole Tumor), 4=ET (Enhancing Tumor).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_ID = "00000057"
FILES = [
    f"{CASE_ID}_final_seg.nii",
    f"{CASE_ID}_api_seg.nii",
]
# Colormap: 0=black, 1=TC=red, 2=WT=green, 4=ET=blue
LABEL_RGB = {
    0: (0, 0, 0),
    1: (1, 0, 0),   # TC
    2: (0, 1, 0),   # WT
    4: (0, 0, 1),   # ET
}


def label_volume_to_rgb(vol: np.ndarray) -> np.ndarray:
    """Convert 2D label slice to RGB (0-1)."""
    out = np.zeros((*vol.shape, 3), dtype=np.float32)
    for label, rgb in LABEL_RGB.items():
        out[vol == label] = rgb
    return out


def main():
    os.chdir(DATA_DIR)

    for nii_name in FILES:
        nii_path = os.path.join(DATA_DIR, nii_name)
        if not os.path.isfile(nii_path):
            print(f"Skip (not found): {nii_path}")
            continue

        img = nib.load(nii_path)
        data = np.asarray(img.dataobj)
        if data.ndim != 3:
            print(f"Skip (not 3D): {nii_name}")
            continue

        # Normalize labels to 0,1,2,4 if needed (float volumes)
        data = np.round(data).astype(np.int32)
        data[~np.isin(data, [0, 1, 2, 4])] = 0

        # Axial slices: take 4 slices around the middle
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
            sl = data[:, :, idx]
            rgb = label_volume_to_rgb(sl)
            ax.imshow(rgb)
            ax.set_title(f"Slice {idx}")
            ax.axis("off")
        plt.suptitle(nii_name.replace(".nii", ""), fontsize=12)
        plt.tight_layout()
        png_name = nii_name.replace(".nii", ".png")
        png_path = os.path.join(DATA_DIR, png_name)
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {png_path}")

    # Legend: TC=red, WT=green, ET=blue
    print("Labels: 1=TC (red), 2=WT (green), 4=ET (blue)")


if __name__ == "__main__":
    main()
