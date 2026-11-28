from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
DATA_DIR = Path(os.environ["FASHION_DIFFUSION_DATA_DIR"])
OUTPUTS_DIR = DATA_DIR / "outputs"
