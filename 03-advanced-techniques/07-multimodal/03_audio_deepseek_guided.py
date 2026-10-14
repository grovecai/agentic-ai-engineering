"""DeepSeek handles text; OpenAI handles audio input and output.

Examples:
  uv run python 03_audio_deepseek_guided.py speak "A short welcome for visitors"
  uv run python 03_audio_deepseek_guided.py transcribe recording.mp3
  uv run python 03_audio_deepseek_guided.py roundtrip "Explain what this demo does"
"""

import argparse
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from deepseek_client import create_client, generate_text


def speech_script(idea: str) -> str:
    return generate_text(
        create_client(),
        "Write a short, natural spoken script from the user's idea. "
        "Preserve facts and language. Return only words to be spoken.",
        idea,
    )


def speak(text: str, voice: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"tts_{voice}_{datetime.now():%Y%m%d_%H%M%S_%f}.mp3"
    response = OpenAI().audio.speech.create(model="tts-1", voice=voice, input=text)
    response.write_to_file(str(path))
    return path


def transcribe(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as audio_file:
        result = OpenAI().audio.transcriptions.create(model="whisper-1", file=audio_file)
    return result.text


def summarize_transcript(transcript: str) -> str:
    return generate_text(
        create_client(),
        "Summarize the transcript accurately in its original language. "
        "Do not add facts. Return a concise summary.",
        transcript,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["speak", "transcribe", "roundtrip"])
    parser.add_argument("input", help="Speech idea, or path to an audio file for transcribe")
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    if args.mode == "transcribe":
        transcript = transcribe(Path(args.input))
        print(f"Whisper transcript: {transcript}")
        print(f"DeepSeek summary: {summarize_transcript(transcript)}")
        return

    script = speech_script(args.input)
    audio_path = speak(script, args.voice, args.output_dir)
    print(f"DeepSeek script: {script}\nOpenAI audio: {audio_path}")
    if args.mode == "roundtrip":
        transcript = transcribe(audio_path)
        print(f"Whisper transcript: {transcript}")
        print(f"Exact match (ignoring case and outer whitespace): "
              f"{script.casefold().strip() == transcript.casefold().strip()}")


if __name__ == "__main__":
    main()
