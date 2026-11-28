import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# DeCo-VTON try-on model on Hugging Face (UNet weights live under DECOVTON_HF_SUBFOLDER)
# https://huggingface.co/levinna/DeCo-VTON
DECOVTON_HF_REPO = "levinna/DeCo-VTON"
DECOVTON_HF_SUBFOLDER = "VITON-HD-512/unet"

# Preprocess checkpoints only (not the CatVTON try-on model) for AutoMasker agnostic-mask generation.
# DensePose/: body-part map (torso, arms, legs, hands, …) — where the body surface is.
# SCHP/: human/fashion parsing (clothes, hair, face, accessories, …) — what is worn on top.
#   Two SCHP heads, same architecture, different label taxonomies (combined with bitwise OR for the mask):
#   - ATR (Active Template Regression, ~18 classes): e.g. Belt, Bag; no Coat/Jumpsuits/Socks/Glove.
#   - LIP (Look Into Person, ~20 classes): e.g. Coat, Glove, Socks, Jumpsuits; no Belt/Bag.
# Together they build the cloth-agnostic mask. https://huggingface.co/zhengchong/CatVTON
CATVTON_HF_REPO = "zhengchong/CatVTON"
