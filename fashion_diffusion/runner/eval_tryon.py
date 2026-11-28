import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from fashion_diffusion.config import DECOVTON_HF_REPO, DECOVTON_HF_SUBFOLDER
from fashion_diffusion.decovton import load_pipeline, prepare_inputs, run_tryon, save_result
from fashion_diffusion.metrics import compute_pair_metrics, pairing_label
from fashion_diffusion.paths import DATA_DIR, OUTPUTS_DIR

DTYPE = torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch DeCo-VTON eval on HR-VITON pairs")
    parser.add_argument("--n", type=int, default=5, help="Number of pairs from the split pair list")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--mask-path", type=str, default="dataset", choices=["dataset"])
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=384)
    return parser.parse_args()


def load_pairs(split: str, n: int) -> list[tuple[Path, Path, Path]]:
    pairs_file = DATA_DIR / f"{split}_pairs.txt"
    lines = pairs_file.read_text().splitlines()[:n]
    pairs = []
    for line in lines:
        person_name, cloth_name = line.split()
        person_path = DATA_DIR / split / "image" / person_name
        cloth_path = DATA_DIR / split / "cloth" / cloth_name
        agnostic_path = DATA_DIR / split / "agnostic-v3.2" / person_name
        if not person_path.is_file():
            raise FileNotFoundError(person_path)
        if not cloth_path.is_file():
            raise FileNotFoundError(cloth_path)
        if not agnostic_path.is_file():
            raise FileNotFoundError(agnostic_path)
        pairs.append((person_path, cloth_path, agnostic_path))
    return pairs


def run_one(pipe, person_path: Path, cloth_path: Path, agnostic_path: Path, result_path: Path, args: argparse.Namespace) -> dict:
    condition, mask = prepare_inputs(person_path, cloth_path, agnostic_path, height=args.height, width=args.width)
    result = run_tryon(
        pipe,
        condition,
        mask,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    save_result(result, result_path)
    gt = Image.open(person_path).convert("RGB")
    metrics = compute_pair_metrics(result, gt)
    return metrics


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = OUTPUTS_DIR / f"eval_tryon_{stamp}"
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True)
    metrics_path = experiment_dir / "metrics.jsonl"

    config = {
        "n": args.n,
        "split": args.split,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "mask_path": args.mask_path,
        "height": args.height,
        "width": args.width,
        "hf_repo": DECOVTON_HF_REPO,
        "hf_subfolder": DECOVTON_HF_SUBFOLDER,
        "dtype": str(DTYPE).replace("torch.", ""),
    }
    (experiment_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"==== Experiment dir: {experiment_dir}")
    pairs = load_pairs(args.split, args.n)
    pipe = load_pipeline(DECOVTON_HF_REPO, DECOVTON_HF_SUBFOLDER, DTYPE)

    rows = []
    with metrics_path.open("w") as f:
        for i, (person_path, cloth_path, agnostic_path) in enumerate(pairs):
            result_name = f"{i:05d}_{person_path.stem}__{cloth_path.stem}.png"
            result_path = results_dir / result_name
            print(f"==== [{i + 1}/{args.n}] {person_path.name} + {cloth_path.name}")
            metrics = run_one(pipe, person_path, cloth_path, agnostic_path, result_path, args)
            row = {
                "index": i,
                "person": person_path.name,
                "cloth": cloth_path.name,
                "pairing": pairing_label(person_path, cloth_path),
                "result": f"results/{result_name}",
                "SSIM": metrics["SSIM"],
                "LPIPS": metrics["LPIPS"],
                "L1": metrics["L1"],
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            print(f"\tSSIM={row['SSIM']:.4f}  LPIPS={row['LPIPS']:.4f}  L1={row['L1']:.4f}")

    mean_ssim = sum(r["SSIM"] for r in rows) / len(rows)
    mean_lpips = sum(r["LPIPS"] for r in rows) / len(rows)
    mean_l1 = sum(r["L1"] for r in rows) / len(rows)
    print("==== Mean metrics:")
    print(f"\tSSIM:  {mean_ssim:.4f}")
    print(f"\tLPIPS: {mean_lpips:.4f}")
    print(f"\tL1:    {mean_l1:.4f}")


if __name__ == "__main__":
    main()
