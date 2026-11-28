import logging
from pathlib import Path

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from decovton import DeCoVTONPipeline

from fashion_diffusion.config import DEVICE

DIFF_THRESHOLD = 30


class _DropVaeFp32ToWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "should be kept in float32" not in record.getMessage()


# Child logger emits this; parent-logger filters are skipped on propagation.
logging.getLogger("diffusers.models.modeling_utils").addFilter(_DropVaeFp32ToWarning())


def load_pipeline(hf_repo: str, subfolder: str, dtype: torch.dtype) -> DeCoVTONPipeline:
    pipe = DeCoVTONPipeline.from_vton_checkpoint(
        hf_repo=hf_repo,
        subfolder=subfolder,
        torch_dtype=dtype,
    )
    return pipe.to(DEVICE)


def agnostic_mask_from_pair(person_pil: Image.Image, agnostic_pil: Image.Image, height: int, width: int) -> Image.Image:
    person = np.asarray(person_pil.convert("RGB").resize((width, height), Image.LANCZOS), dtype=np.int16)
    agnostic = np.asarray(agnostic_pil.convert("RGB").resize((width, height), Image.LANCZOS), dtype=np.int16)
    diff = np.abs(person - agnostic).sum(axis=-1)
    mask = (diff > DIFF_THRESHOLD).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def prepare_inputs(person_path: Path, cloth_path: Path, agnostic_path: Path, height: int = 512, width: int = 384) -> tuple[torch.Tensor, torch.Tensor]:
    person_pil = Image.open(person_path).convert("RGB")
    cloth_pil = Image.open(cloth_path).convert("RGB")
    agnostic_pil = Image.open(agnostic_path).convert("RGB")
    mask_pil = agnostic_mask_from_pair(person_pil, agnostic_pil, height, width)
    return prepare_inputs_from_mask(person_path, cloth_path, mask_pil, height, width)


def prepare_inputs_from_mask(person_path: Path, cloth_path: Path, mask_pil: Image.Image, height: int = 512, width: int = 384) -> tuple[torch.Tensor, torch.Tensor]:
    vae_proc = VaeImageProcessor(vae_scale_factor=8)
    mask_proc = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    person_pil = Image.open(person_path).convert("RGB")
    cloth_pil = Image.open(cloth_path).convert("RGB")
    mask_pil = mask_pil.convert("L")

    person = vae_proc.preprocess(person_pil, height, width)
    cloth = vae_proc.preprocess(cloth_pil, height, width)
    mask = mask_proc.preprocess(mask_pil, height, width)

    masked = person * (mask < 0.5)
    condition = torch.cat([masked, cloth], dim=2)
    return condition, mask


def run_tryon(
    pipe: DeCoVTONPipeline,
    condition: torch.Tensor,
    mask: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 2.5,
    seed: int = 42,
) -> Image.Image:
    device = pipe.device
    dtype = pipe.unet.dtype
    condition = condition.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(seed)
    results, _ = pipe(
        condition_image=condition,
        mask=mask,
        warped_garment=None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pil",
        scheduler_type="ddim",
        eta=1.0,
        concat_mode="latent",
    )
    return results[0]


def save_result(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def save_gt_vs_result(gt: Image.Image, result: Image.Image, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    axes[0].imshow(gt)
    axes[0].set_title("GT")
    axes[0].axis("off")
    axes[1].imshow(result)
    axes[1].set_title("generated")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_person_cloth_result(
    person: Image.Image,
    cloth: Image.Image,
    agnostic: Image.Image,
    mask: Image.Image,
    result: Image.Image,
    path: Path,
    agnostic_title: str,
    mask_title: str,
    figure_title: str,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 4, figure=fig)
    ax_person = fig.add_subplot(gs[0, 0])
    ax_cloth = fig.add_subplot(gs[0, 1])
    ax_agnostic = fig.add_subplot(gs[1, 0])
    ax_mask = fig.add_subplot(gs[1, 1])
    ax_result = fig.add_subplot(gs[:, 2:])
    ax_person.imshow(person)
    ax_person.set_title("Person")
    ax_person.axis("off")
    ax_cloth.imshow(cloth)
    ax_cloth.set_title("Cloth")
    ax_cloth.axis("off")
    ax_agnostic.imshow(agnostic)
    ax_agnostic.set_title(agnostic_title)
    ax_agnostic.axis("off")
    ax_mask.imshow(mask.convert("L"), cmap="gray")
    ax_mask.set_title(mask_title)
    ax_mask.axis("off")
    ax_result.imshow(result)
    ax_result.set_title("Generated")
    ax_result.axis("off")
    fig.suptitle(figure_title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _show_figure(fig) -> None:
    import io

    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue()))


def show_inputs(person_path: Path, cloth_path: Path, agnostic_path: Path) -> None:
    import matplotlib.pyplot as plt

    person = Image.open(person_path)
    cloth = Image.open(cloth_path)
    agnostic = Image.open(agnostic_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(person)
    axes[0].set_title(f"person {np.asarray(person).shape}")
    axes[0].axis("off")
    axes[1].imshow(cloth)
    axes[1].set_title(f"cloth {np.asarray(cloth).shape}")
    axes[1].axis("off")
    axes[2].imshow(agnostic)
    axes[2].set_title(f"agnostic {np.asarray(agnostic).shape}")
    axes[2].axis("off")
    _show_figure(fig)


def show_person_mask(person_path: Path, mask_pil: Image.Image) -> None:
    import matplotlib.pyplot as plt

    person = Image.open(person_path)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(person)
    axes[0].set_title(f"person {np.asarray(person).shape}")
    axes[0].axis("off")
    axes[1].imshow(mask_pil.convert("L"), cmap="gray")
    axes[1].set_title(f"mask {np.asarray(mask_pil.convert('L')).shape}")
    axes[1].axis("off")
    _show_figure(fig)


def show_person_cloth_mask(person_path: Path, cloth_path: Path, mask_pil: Image.Image) -> None:
    import matplotlib.pyplot as plt

    person = Image.open(person_path)
    cloth = Image.open(cloth_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(person)
    axes[0].set_title(f"person {np.asarray(person).shape}")
    axes[0].axis("off")
    axes[1].imshow(cloth)
    axes[1].set_title(f"cloth {np.asarray(cloth).shape}")
    axes[1].axis("off")
    axes[2].imshow(mask_pil.convert("L"), cmap="gray")
    axes[2].set_title(f"mask {np.asarray(mask_pil.convert('L')).shape}")
    axes[2].axis("off")
    _show_figure(fig)


def show_condition_and_mask(condition: torch.Tensor, mask: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    condition_img = ((condition[0].permute(1, 2, 0) + 1) / 2).clamp(0, 1).cpu().numpy()
    mask_img = mask[0, 0].cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].imshow(condition_img)
    axes[0].set_title(f"condition {tuple(condition.shape)}")
    axes[0].axis("off")
    axes[1].imshow(mask_img, cmap="gray")
    axes[1].set_title(f"mask {tuple(mask.shape)}")
    axes[1].axis("off")
    _show_figure(fig)


def show_result(person_path: Path, cloth_path: Path, result: Image.Image) -> None:
    import matplotlib.pyplot as plt

    person = Image.open(person_path)
    cloth = Image.open(cloth_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(person)
    axes[0].set_title("person")
    axes[0].axis("off")
    axes[1].imshow(cloth)
    axes[1].set_title("cloth")
    axes[1].axis("off")
    axes[2].imshow(result)
    axes[2].set_title("try-on")
    axes[2].axis("off")
    _show_figure(fig)
