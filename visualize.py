import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

output_dir = "./outputs/IABI Project"
data_dir = "/home/sajjad/iabi/data"
save_dir = "./outputs/visualizations"
os.makedirs(save_dir, exist_ok=True)

seg_files = sorted(glob.glob(os.path.join(output_dir, "*.nii.gz")))

for seg_path in seg_files:
    case_name = os.path.basename(seg_path).replace(".nii.gz", "")
    case_id = case_name  # e.g. BraTS2021_01266

    # Load segmentation
    seg = nib.load(seg_path).get_fdata()

    # Try to load original FLAIR image for overlay
    flair_path = os.path.join(data_dir, "TrainingData", case_id, f"{case_id}_flair.nii.gz")
    has_flair = os.path.exists(flair_path)
    if has_flair:
        flair = nib.load(flair_path).get_fdata()

    # Pick middle slices in each axis
    mid_ax = seg.shape[2] // 2
    mid_cor = seg.shape[1] // 2
    mid_sag = seg.shape[0] // 2

    fig, axes = plt.subplots(2 if has_flair else 1, 3, figsize=(15, 10 if has_flair else 5))
    if not has_flair:
        axes = [axes]

    fig.suptitle(f"{case_name} — Labels: 1=TC (blue), 2=WT (green), 4=ET (red)", fontsize=14)

    slices_seg = [seg[:, :, mid_ax], seg[:, mid_cor, :], seg[mid_sag, :, :]]
    titles = ["Axial", "Coronal", "Sagittal"]

    if has_flair:
        slices_flair = [flair[:, :, mid_ax], flair[:, mid_cor, :], flair[mid_sag, :, :]]

        # Row 1: FLAIR with segmentation overlay
        for j in range(3):
            ax = axes[0][j]
            ax.imshow(slices_flair[j].T, cmap="gray", origin="lower")
            masked = np.ma.masked_where(slices_seg[j] == 0, slices_seg[j])
            ax.imshow(masked.T, cmap="jet", alpha=0.5, origin="lower", vmin=0, vmax=4)
            ax.set_title(f"{titles[j]} — Overlay")
            ax.axis("off")

        # Row 2: Segmentation only
        for j in range(3):
            ax = axes[1][j]
            ax.imshow(slices_seg[j].T, cmap="jet", origin="lower", vmin=0, vmax=4)
            ax.set_title(f"{titles[j]} — Segmentation")
            ax.axis("off")
    else:
        # Segmentation only
        for j in range(3):
            ax = axes[0][j]
            ax.imshow(slices_seg[j].T, cmap="jet", origin="lower", vmin=0, vmax=4)
            ax.set_title(f"{titles[j]} — Segmentation")
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{case_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

print(f"\nDone! {len(seg_files)} visualizations saved to {save_dir}/")
