from pathlib import Path
import json

MODALITIES = [
    ("image", "{stem}.jpg", "foto della persona"),
    ("cloth", "{stem}.jpg", "immagine del capo"),
    ("cloth-mask", "{stem}.jpg", "maschera del capo"),
    ("agnostic-v3.2", "{stem}.jpg", "persona agnostica"),
    ("image-parse-v3", "{stem}.png", "parsing semantico"),
    ("image-parse-agnostic-v3.2", "{stem}.png", "parsing agnostico"),
    ("image-densepose", "{stem}.jpg", "mappa DensePose"),
    ("openpose_img", "{stem}_rendered.png", "pose renderizzata"),
    ("openpose_json", "{stem}_keypoints.json", "keypoints OpenPose"),
]


def sample_stem(filename: str) -> str:
    return Path(filename).stem.replace("_rendered", "").replace("_keypoints", "")


def find_modalities(data_dir: Path, filename: str) -> list[tuple[str, str, Path]]:
    stem = sample_stem(filename)
    found = []
    for split in ("train", "test"):
        for folder, name_tmpl, desc in MODALITIES:
            path = data_dir / split / folder / name_tmpl.format(stem=stem)
            if path.exists():
                found.append((folder, desc, path))
    return found


def show_modalities(modalities: list[tuple[str, str, Path]], title: str | None = None):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    image_items = [(folder, desc, path) for folder, desc, path in modalities if path.suffix.lower() != ".json"]
    n = len(image_items)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for ax, (folder, desc, path) in zip(axes, image_items):
        arr = np.asarray(Image.open(path))
        ax.imshow(arr)
        ax.set_title(f"{folder}\n{desc}", fontsize=10)
        ax.axis("off")

        def format_coord(x, y, arr=arr):
            xi, yi = int(x + 0.5), int(y + 0.5)
            if yi < 0 or yi >= arr.shape[0] or xi < 0 or xi >= arr.shape[1]:
                return ""
            pixel = arr[yi, xi]
            if arr.ndim == 2:
                return f"x={xi}, y={yi}, gray={int(pixel)}"
            r, g, b = (int(v) for v in pixel[:3])
            return f"x={xi}, y={yi}, RGB=({r}, {g}, {b})"

        ax.format_coord = format_coord
    for ax in axes[n:]:
        ax.axis("off")
    if title is not None:
        fig.suptitle(title, fontsize=14)
    status = fig.text(0.5, 0.01, "move over an image to see RGB", ha="center", fontsize=11)

    def on_move(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if not event.inaxes.images:
            return
        status.set_text(event.inaxes.format_coord(event.xdata, event.ydata))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.show()


def print_json_modalities(modalities: list[tuple[str, str, Path]], max_chars: int = 2000):
    for folder, desc, path in modalities:
        if path.suffix.lower() != ".json":
            continue
        text = json.dumps(json.loads(path.read_text()), indent=2)
        print(f"=== {folder} — {desc} ===")
        print(f"path: {path}")
        print(text[:max_chars])
        if len(text) > max_chars:
            print("... (troncato)")
