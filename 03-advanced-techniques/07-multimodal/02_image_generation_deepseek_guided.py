"""DeepSeek writes an image prompt; Gemini generates the image.

DeepSeek's hosted API does not currently return image output. This example
keeps the image generation step explicit instead of presenting it as native.
"""

import argparse
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

from deepseek_client import create_client, generate_text

GEMINI_MODEL = "gemini-2.0-flash-exp-image-generation"
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def generate(idea: str, output_dir: Path) -> tuple[str, Path]:
    image_prompt = generate_text(
        create_client(),
        "Turn the user's idea into one clear, detailed image generation prompt. "
        "Keep the intended subject and style. Return only the prompt.",
        idea,
    )
    response = genai.Client().models.generate_content(
        model=GEMINI_MODEL,
        contents=image_prompt,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )
    if not response.candidates or not response.candidates[0].content:
        raise RuntimeError("Gemini returned no content")
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            mime_type = part.inline_data.mime_type
            if mime_type not in EXTENSIONS:
                raise RuntimeError(f"Unexpected generated image type: {mime_type}")
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"generated_{datetime.now():%Y%m%d_%H%M%S_%f}{EXTENSIONS[mime_type]}"
            path.write_bytes(part.inline_data.data)
            return image_prompt, path
    raise RuntimeError("Gemini returned text but no image")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("idea", help="Image idea for DeepSeek to turn into a prompt")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    prompt, path = generate(args.idea, args.output_dir)
    print(f"DeepSeek prompt: {prompt}\nGemini image: {path}")


if __name__ == "__main__":
    main()
