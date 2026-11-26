import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

DECOVTON_ROOT = Path(__file__).resolve().parents[1] / "DeCo-VTON"


def load_automasker(catvton_repo: str = "zhengchong/CatVTON", device: str = "cuda"):
    if str(DECOVTON_ROOT) not in sys.path:
        sys.path.insert(0, str(DECOVTON_ROOT))
    from thirdparty.cloth_masker import AutoMasker

    repo_path = Path(catvton_repo)
    if not repo_path.is_dir():
        repo_path = Path(snapshot_download(repo_id=catvton_repo))
    return AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device=device,
    )


def generate_agnostic_mask(automasker, person_path: Path, mask_type: str = "upper") -> Image.Image:
    return automasker(str(person_path), mask_type)["mask"]
