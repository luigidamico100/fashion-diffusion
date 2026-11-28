from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from fashion_diffusion.config import DEVICE


def pairing_label(person_path: Path, cloth_path: Path) -> str:
    if Path(person_path).stem == Path(cloth_path).stem:
        return "paired"
    return "unpaired"


def _pil_to_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    rgb = image.convert("RGB").resize(size, Image.LANCZOS)
    return torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)



def compute_pair_metrics(pred: Image.Image, gt: Image.Image) -> dict[str, float]:
    width, height = gt.convert("RGB").size
    pred_t = _pil_to_tensor(pred, (width, height)).to(DEVICE)
    gt_t = _pil_to_tensor(gt, (width, height)).to(DEVICE)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(DEVICE)

    ssim = ssim_metric(pred_t, gt_t).item()
    lpips = lpips_metric(pred_t, gt_t).item()
    l1 = F.l1_loss(pred_t, gt_t).item()
    return {"SSIM": ssim, "LPIPS": lpips, "L1": l1}


def format_pair_metrics_report(label: str, metrics: dict[str, float]) -> str:
    lines = [
        f"==== Pair metrics ({label}):",
        f"\tSSIM:  {metrics['SSIM']:.4f}",
        f"\tLPIPS: {metrics['LPIPS']:.4f}",
        f"\tL1:    {metrics['L1']:.4f}",
    ]
    return "\n".join(lines)


def print_pair_metrics(person_path: Path, cloth_path: Path, result: Image.Image) -> dict[str, float]:
    label = pairing_label(person_path, cloth_path)
    gt = Image.open(person_path).convert("RGB")
    metrics = compute_pair_metrics(result, gt)
    print(format_pair_metrics_report(label, metrics))
    return metrics
