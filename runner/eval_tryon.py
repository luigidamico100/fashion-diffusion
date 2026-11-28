import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from fashion_diffusion.agnostic_mask import generate_agnostic_mask, load_automasker
from fashion_diffusion.config import CATVTON_HF_REPO, DECOVTON_HF_REPO, DECOVTON_HF_SUBFOLDER, DEVICE
from fashion_diffusion.decovton import (
    agnostic_mask_from_pair,
    load_pipeline,
    prepare_inputs,
    prepare_inputs_from_mask,
    run_tryon,
    save_person_cloth_result,
)
from fashion_diffusion.metrics import compute_pair_metrics
from fashion_diffusion.paths import DATA_DIR, OUTPUTS_DIR

DTYPE = torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch DeCo-VTON eval on HR-VITON pairs (try-on + SSIM/LPIPS/L1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input (from FASHION_DIFFUSION_DATA_DIR / DATA_DIR):\n"
            f"  pair list:  {DATA_DIR}/{{split}}_pairs.txt\n"
            f"  person:     {DATA_DIR}/{{split}}/image/<person>.jpg\n"
            f"  cloth:      {DATA_DIR}/{{split}}/cloth/<cloth>.jpg\n"
            f"  agnostic:   {DATA_DIR}/{{split}}/agnostic-v3.2/<person>.jpg  (when --mask-path includes dataset)\n"
            "\n"
            "Output (under DATA_DIR/outputs):\n"
            f"  experiment: {OUTPUTS_DIR}/eval_tryon_<YYYYMMDD_HHMMSS>/\n"
            "              config.json, metrics.csv, mean_metrics.csv,\n"
            "              paired_comparison/*.png, unpaired_comparison/*.png\n"
            "              (+ paired_comparison_automask/, unpaired_comparison_automask/ when --mask-path=both)"
        ),
    )
    parser.add_argument("--n", type=int, default=3, help="Number of pairs to take from the split pair list (first N lines); -1 = all")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split; selects {split}_pairs.txt and DATA_DIR/{split}/...")
    parser.add_argument(
        "--pairing",
        type=str,
        default="both",
        choices=["paired", "unpaired", "both"],
        help="Which pairings to evaluate: unpaired from the list, paired (same-name cloth for first N people), or both",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for diffusion sampling")
    parser.add_argument("--num-inference-steps", type=int, default=5, help="Diffusion denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=2.5, help="Classifier-free guidance scale")
    parser.add_argument(
        "--mask-path",
        type=str,
        default="both",
        choices=["dataset", "automask", "both"],
        help="How to build the agnostic try-on mask: 'dataset' = HR-VITON agnostic-v3.2, 'automask' = AutoMasker (DensePose+SCHP), 'both' = run both",
    )
    parser.add_argument("--height", type=int, default=512, help="Try-on input height in pixels")
    parser.add_argument("--width", type=int, default=384, help="Try-on input width in pixels")
    return parser.parse_args()


def resolve_mask_modes(mask_path: str) -> list[str]:
    if mask_path == "both":
        return ["dataset", "automask"]
    return [mask_path]


def comparison_dir_name(pairing: str, mask_mode: str, mask_path_arg: str) -> str:
    if mask_path_arg == "both" and mask_mode == "automask":
        return f"{pairing}_comparison_automask"
    return f"{pairing}_comparison"


def load_pairs(split: str, n: int, pairing: str, require_agnostic: bool) -> list[tuple[Path, Path, Path | None, str]]:
    pairs_file = DATA_DIR / f"{split}_pairs.txt"
    lines = pairs_file.read_text().splitlines()
    if n != -1:
        lines = lines[:n]
    modes = ["paired", "unpaired"] if pairing == "both" else [pairing]
    pairs = []
    for mode in modes:
        for line in lines:
            person_name, listed_cloth_name = line.split()
            cloth_name = person_name if mode == "paired" else listed_cloth_name
            person_path = DATA_DIR / split / "image" / person_name
            cloth_path = DATA_DIR / split / "cloth" / cloth_name
            if not person_path.is_file():
                raise FileNotFoundError(person_path)
            if not cloth_path.is_file():
                raise FileNotFoundError(cloth_path)
            agnostic_path = None
            if require_agnostic:
                agnostic_path = DATA_DIR / split / "agnostic-v3.2" / person_name
                if not agnostic_path.is_file():
                    raise FileNotFoundError(agnostic_path)
            pairs.append((person_path, cloth_path, agnostic_path, mode))
    return pairs


def masked_person_from_mask(person: Image.Image, mask_pil: Image.Image, height: int, width: int) -> Image.Image:
    person_arr = np.asarray(person.convert("RGB").resize((width, height), Image.LANCZOS), dtype=np.uint8)
    mask_arr = np.asarray(mask_pil.convert("L").resize((width, height), Image.NEAREST))
    out = person_arr.copy()
    out[mask_arr > 127] = 128
    return Image.fromarray(out, mode="RGB")


def run_one(pipe, person_path: Path, cloth_path: Path, agnostic_path: Path | None, comparison_path: Path, pairing: str, mask_mode: str, automasker, args: argparse.Namespace) -> dict:
    person = Image.open(person_path).convert("RGB")
    cloth = Image.open(cloth_path).convert("RGB")
    if mask_mode == "dataset":
        condition, mask = prepare_inputs(person_path, cloth_path, agnostic_path, height=args.height, width=args.width)
        agnostic = Image.open(agnostic_path).convert("RGB")
        mask_pil = agnostic_mask_from_pair(person, agnostic, args.height, args.width)
        agnostic_title = "Agnostic (dataset)"
        mask_title = "Mask (from dataset agnostic)"
    elif mask_mode == "automask":
        mask_pil = generate_agnostic_mask(automasker, person_path, mask_type="upper")
        condition, mask = prepare_inputs_from_mask(person_path, cloth_path, mask_pil, height=args.height, width=args.width)
        agnostic = masked_person_from_mask(person, mask_pil, args.height, args.width)
        agnostic_title = "Agnostic (automask)"
        mask_title = "Mask (automask)"
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")
    result = run_tryon(
        pipe,
        condition,
        mask,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    metrics = compute_pair_metrics(result, person)
    header = f"DeCo-VTON · {pairing.capitalize()} try-on · {mask_mode}"
    sample = f"Person: {person_path.name}  |  Cloth: {cloth_path.name}"
    if pairing == "paired":
        scores = f"SSIM={metrics['SSIM']:.3f}  LPIPS={metrics['LPIPS']:.3f}  L1={metrics['L1']:.3f}"
    else:
        scores = f"LPIPS={metrics['LPIPS']:.3f}  L1={metrics['L1']:.3f}"
    figure_title = f"{header}\n{sample}\n{scores}"
    save_person_cloth_result(person, cloth, agnostic, mask_pil, result, comparison_path, agnostic_title, mask_title, figure_title)
    return metrics


def main() -> None:
    args = parse_args()
    print(f"==== Device: {DEVICE}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = OUTPUTS_DIR / f"eval_tryon_{stamp}"
    experiment_dir.mkdir(parents=True)

    mask_modes = resolve_mask_modes(args.mask_path)
    pairing_modes = ["paired", "unpaired"] if args.pairing == "both" else [args.pairing]
    comparison_dirs = {}
    for pairing in pairing_modes:
        for mask_mode in mask_modes:
            dir_name = comparison_dir_name(pairing, mask_mode, args.mask_path)
            comparison_dirs[(pairing, mask_mode)] = experiment_dir / dir_name
            comparison_dirs[(pairing, mask_mode)].mkdir()
    metrics_path = experiment_dir / "metrics.csv"

    config = {
        "n": args.n,
        "split": args.split,
        "pairing": args.pairing,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "mask_path": args.mask_path,
        "height": args.height,
        "width": args.width,
        "hf_repo": DECOVTON_HF_REPO,
        "hf_subfolder": DECOVTON_HF_SUBFOLDER,
        "dtype": str(DTYPE).replace("torch.", ""),
        "device": DEVICE,
    }
    (experiment_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"==== Experiment dir: {experiment_dir}")
    require_agnostic = "dataset" in mask_modes
    pairs = load_pairs(args.split, args.n, args.pairing, require_agnostic=require_agnostic)
    print(f"==== Pairs: {len(pairs)} (pairing={args.pairing}, mask_path={args.mask_path})")

    automasker = None
    if "automask" in mask_modes:
        print(f"==== Loading AutoMasker from {CATVTON_HF_REPO}")
        automasker = load_automasker(CATVTON_HF_REPO)

    pipe = load_pipeline(DECOVTON_HF_REPO, DECOVTON_HF_SUBFOLDER, DTYPE)
    pipe.set_progress_bar_config(disable=True)

    jobs = []
    for pair_i, (person_path, cloth_path, agnostic_path, pairing) in enumerate(pairs):
        for mask_mode in mask_modes:
            jobs.append((pair_i, person_path, cloth_path, agnostic_path, pairing, mask_mode))

    fieldnames = ["index", "person", "cloth", "pairing", "mask_path", "SSIM", "LPIPS", "L1", "comparison"]
    rows = []
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (pair_i, person_path, cloth_path, agnostic_path, pairing, mask_mode) in enumerate(tqdm(jobs, desc="try-on")):
            result_name = f"{pair_i:05d}_{person_path.stem}__{cloth_path.stem}.png"
            dir_name = comparison_dir_name(pairing, mask_mode, args.mask_path)
            comparison_rel = f"{dir_name}/{result_name}"
            comparison_path = comparison_dirs[(pairing, mask_mode)] / result_name
            metrics = run_one(pipe, person_path, cloth_path, agnostic_path, comparison_path, pairing, mask_mode, automasker, args)
            # Paper reports SSIM only for paired evaluation.
            ssim = metrics["SSIM"] if pairing == "paired" else ""
            row = {
                "index": i,
                "person": person_path.name,
                "cloth": cloth_path.name,
                "pairing": pairing,
                "mask_path": mask_mode,
                "SSIM": ssim,
                "LPIPS": metrics["LPIPS"],
                "L1": metrics["L1"],
                "comparison": comparison_rel,
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)
            ssim_msg = f"  SSIM={ssim:.4f}" if pairing == "paired" else ""
            tqdm.write(f"\t[{i + 1}/{len(jobs)}] {pairing}/{mask_mode} {person_path.name} + {cloth_path.name}{ssim_msg}  LPIPS={row['LPIPS']:.4f}  L1={row['L1']:.4f}")

    mean_path = experiment_dir / "mean_metrics.csv"
    mean_fieldnames = ["pairing", "mask_path", "n", "SSIM", "LPIPS", "L1"]
    with mean_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mean_fieldnames)
        writer.writeheader()
        for pairing in pairing_modes:
            for mask_mode in mask_modes:
                mode_rows = [r for r in rows if r["pairing"] == pairing and r["mask_path"] == mask_mode]
                if not mode_rows:
                    continue
                mean_lpips = sum(r["LPIPS"] for r in mode_rows) / len(mode_rows)
                mean_l1 = sum(r["L1"] for r in mode_rows) / len(mode_rows)
                mean_ssim = ""
                if pairing == "paired":
                    mean_ssim = sum(r["SSIM"] for r in mode_rows) / len(mode_rows)
                writer.writerow({"pairing": pairing, "mask_path": mask_mode, "n": len(mode_rows), "SSIM": mean_ssim, "LPIPS": mean_lpips, "L1": mean_l1})
                print(f"==== Mean metrics ({pairing}, {mask_mode}, n={len(mode_rows)}):")
                if pairing == "paired":
                    print(f"\tSSIM:  {mean_ssim:.4f}")
                print(f"\tLPIPS: {mean_lpips:.4f}")
                print(f"\tL1:    {mean_l1:.4f}")
    print(f"==== Experiment dir: {experiment_dir}")


if __name__ == "__main__":
    main()
