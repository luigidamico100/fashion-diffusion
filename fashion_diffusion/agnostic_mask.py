from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from decovton.thirdparty.cloth_masker import AutoMasker

# DensePose/SCHP adaptive pool: non-divisible sizes not implemented on MPS.
MASK_DEVICE = "cpu"


def load_automasker(catvton_repo: str):
    repo_path = Path(snapshot_download(repo_id=catvton_repo))
    return AutoMasker(
        densepose_ckpt=str(repo_path / "DensePose"),
        schp_ckpt=str(repo_path / "SCHP"),
        device=MASK_DEVICE,
    )


def generate_agnostic_mask(automasker, person_path: Path, mask_type: str) -> Image.Image:
    return automasker(str(person_path), mask_type)["mask"]


def run_mask_pipeline(automasker, person_path: Path, mask_type: str) -> dict:
    return automasker(str(person_path), mask_type)


def _show_figure(fig) -> None:
    import io

    import matplotlib.pyplot as plt
    from IPython.display import Image as IPyImage
    from IPython.display import display

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    display(IPyImage(data=buf.getvalue()))


def _to_display_rgb(image: Image.Image) -> np.ndarray:
    if image.mode == "P":
        return np.asarray(image.convert("RGB"))
    if image.mode == "L":
        return np.asarray(image)
    return np.asarray(image.convert("RGB"))


def show_mask_pipeline(person_path: Path, steps: dict, mask_type: str) -> None:
    import matplotlib.pyplot as plt

    person = Image.open(person_path).convert("RGB")
    densepose = steps["densepose"]
    schp_atr = steps["schp_atr"]
    schp_lip = steps["schp_lip"]
    mask = steps["mask"].convert("L")

    mask_arr = np.asarray(mask)
    overlay = np.asarray(person).copy()
    overlay[mask_arr > 127] = (overlay[mask_arr > 127] * 0.35 + np.array([255, 64, 64]) * 0.65).astype(np.uint8)

    panels = [
        (person, "1. person", None),
        (_to_display_rgb(densepose), "2. DensePose (body parts)", "nipy_spectral" if densepose.mode == "L" else None),
        (_to_display_rgb(schp_atr), "3. SCHP ATR (clothes/parts)", None),
        (_to_display_rgb(schp_lip), "4. SCHP LIP (clothes/parts)", None),
        (mask_arr, f"5. agnostic mask ({mask_type})", "gray"),
        (overlay, "6. mask on person", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    for ax, (img, title, cmap) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    _show_figure(fig)
