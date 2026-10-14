"""Analyze one or more images with DeepSeek Flash vision.

Examples:
  uv run python 01_vision_deepseek.py photo.jpg
  uv run python 01_vision_deepseek.py https://example.com/photo.jpg --prompt "Describe it"
  uv run python 01_vision_deepseek.py before.png after.png --prompt "Compare them"
"""

import argparse
import base64
import mimetypes
from pathlib import Path

from deepseek_client import create_client

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def image_url(source: str) -> str:
    """Return a remote URL or encode a local image as a data URL."""
    if source.startswith(("https://", "http://")):
        return source
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(source)
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported image type: {media_type}. Use JPEG, PNG, GIF, or WebP")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def analyze(sources: list[str], prompt: str) -> str:
    content = []
    for index, source in enumerate(sources, 1):
        content.append({"type": "text", "text": f"Image {index}:"})
        content.append({"type": "image_url", "image_url": {"url": image_url(source)}})
    content.append({"type": "text", "text": prompt})
    response = create_client().chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": content}],
        max_tokens=2048,
        extra_body={"thinking": {"type": "disabled"}},
    )
    result = response.choices[0].message.content
    if not result:
        raise RuntimeError("DeepSeek returned no analysis")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Local image paths or public image URLs")
    parser.add_argument("--prompt", default="Describe the image(s) in detail.")
    args = parser.parse_args()
    print(analyze(args.images, args.prompt))


if __name__ == "__main__":
    main()
