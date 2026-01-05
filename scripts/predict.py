"""Prediction script for Luogu captchas fetched from the live endpoint.

Fetches a captcha image from https://www.luogu.com.cn/lg4/captcha,
runs the trained CRNN model, and displays the image and prediction.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from torchvision import transforms

# Allow importing src modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import BLANK_INDEX, CHARSET  # noqa: E402
from src.model import CRNN, ctc_greedy_decode  # noqa: E402


def load_model(ckpt_path: Path, device: torch.device) -> CRNN:
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config_dict = ckpt.get("config", {})
    img_channels = config_dict.get("img_channels", 3)

    model = CRNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def fetch_captcha_image(timeout: int = 10) -> Image.Image:
    """Fetch a captcha image from the Luogu endpoint."""
    url = "https://www.luogu.com.cn/lg4/captcha"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.luogu.com.cn/",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    img = Image.open(io.BytesIO(response.content))
    return img


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor (1, 3, 35, 90)."""
    img = img.convert("RGB")
    img = img.resize((90, 35), Image.BILINEAR)

    transform = transforms.ToTensor()
    tensor = transform(img)
    return tensor.unsqueeze(0)  # Add batch dimension


def predict(model: CRNN, image_tensor: torch.Tensor, device: torch.device) -> str:
    """Run inference and return the decoded text."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        log_probs = model(image_tensor)  # (T, B, C)

    decoded = ctc_greedy_decode(log_probs, charset=CHARSET)
    return decoded[0] if decoded else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Luogu captchas from live endpoint with visualization"
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "checkpoints" / "crnn.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of captchas to fetch and predict",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    print(f"Loading model from {args.ckpt}...")
    model = load_model(args.ckpt, device)
    print("Model loaded.\n")

    # Fetch and predict with visualization
    for i in range(1, args.count + 1):
        try:
            print(f"[{i}/{args.count}] Fetching captcha...")
            img = fetch_captcha_image()

            tensor = preprocess_image(img)
            prediction = predict(model, tensor, device)

            # Display with matplotlib
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.imshow(img)
            ax.set_title(f"Predicted: {prediction}", fontsize=14, fontweight="bold")
            ax.axis("off")
            plt.tight_layout()
            plt.show()

            print(f"         Prediction: {prediction}\n")

        except requests.RequestException as e:
            print(f"         Error fetching captcha: {e}\n")


if __name__ == "__main__":
    main()
