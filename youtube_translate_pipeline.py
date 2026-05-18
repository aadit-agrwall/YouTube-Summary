#!/usr/bin/env python3
"""
YouTube video translator pipeline using only open-source libraries.

Pipeline:
1. Download audio-only from YouTube as WAV with yt-dlp.
2. Transcribe with faster-whisper.
3. Translate each segment with Meta NLLB-200.
4. Write source and translated SRT files.

Example:
    python youtube_translate_pipeline.py \
        --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
        --target-lang spa_Latn \
        --output-dir outputs
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import yt_dlp
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


LOGGER = logging.getLogger("youtube_translate_pipeline")
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_NLLB_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_SOURCE_LANG = "eng_Latn"
DEFAULT_TARGET_LANG = "spa_Latn"

# A minimal mapping for common Whisper language codes to NLLB language tags.
# If a language is not in the map, the script falls back to the user-provided
# source language.
WHISPER_TO_NLLB = {
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "zh": "zho_Hans",
}


class PipelineError(Exception):
    """Raised when a pipeline step fails."""


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, transcribe, and translate a YouTube video."
    )
    parser.add_argument("--url", required=True, help="YouTube video URL.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated files. Default: outputs",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        choices=["base", "small"],
        help="faster-whisper model size. Default: small",
    )
    parser.add_argument(
        "--nllb-model",
        default=DEFAULT_NLLB_MODEL,
        help=f"NLLB model id. Default: {DEFAULT_NLLB_MODEL}",
    )
    parser.add_argument(
        "--source-lang",
        default=DEFAULT_SOURCE_LANG,
        help=f"NLLB source language tag fallback. Default: {DEFAULT_SOURCE_LANG}",
    )
    parser.add_argument(
        "--target-lang",
        default=DEFAULT_TARGET_LANG,
        help=f"NLLB target language tag. Default: {DEFAULT_TARGET_LANG}",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for Whisper transcription. Default: 5",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable VAD filtering during transcription.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the downloaded WAV file instead of deleting it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def detect_device() -> tuple[str, str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", "float16", torch.float16
    return "cpu", "int8", torch.float32


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_stem(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return safe.strip("_") or "video"


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def write_srt(segments: Iterable[Segment], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(f"{segment.index}\n")
            handle.write(
                f"{format_srt_timestamp(segment.start)} --> "
                f"{format_srt_timestamp(segment.end)}\n"
            )
            handle.write(f"{segment.text.strip()}\n\n")


def download_audio(url: str, working_dir: Path) -> tuple[Path, str]:
    output_template = str(working_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise PipelineError(f"Audio download failed: {exc}") from exc

    video_id = info.get("id", "video")
    wav_path = working_dir / f"{video_id}.wav"
    if not wav_path.exists():
        raise PipelineError("yt-dlp completed, but WAV file was not created.")

    return wav_path, sanitize_stem(video_id)


def transcribe_audio(
    audio_path: Path,
    model_size: str,
    beam_size: int,
    vad_filter: bool,
    device: str,
    compute_type: str,
) -> tuple[list[Segment], Optional[str]]:
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments_iter, info = model.transcribe(
            str(audio_path),
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
    except Exception as exc:
        raise PipelineError(f"Transcription failed: {exc}") from exc

    segments: list[Segment] = []
    for idx, piece in enumerate(segments_iter, start=1):
        text = piece.text.strip()
        if not text:
            continue
        segments.append(
            Segment(
                index=idx,
                start=float(piece.start),
                end=float(piece.end),
                text=text,
            )
        )

    if not segments:
        raise PipelineError("Transcription finished, but no speech segments were found.")

    language = getattr(info, "language", None)
    return segments, language


class NLLBTranslator:
    def __init__(
        self,
        model_name: str,
        source_lang: str,
        target_lang: str,
        device: str,
        torch_dtype: torch.dtype,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.device = torch.device(device if device == "cuda" else "cpu")
        self.model.to(self.device)
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate_text(self, text: str) -> str:
        self.tokenizer.src_lang = self.source_lang
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def resolve_source_lang(detected_whisper_lang: Optional[str], fallback: str) -> str:
    if not detected_whisper_lang:
        return fallback
    return WHISPER_TO_NLLB.get(detected_whisper_lang, fallback)


def translate_segments(
    segments: list[Segment],
    model_name: str,
    source_lang: str,
    target_lang: str,
    device: str,
    torch_dtype: torch.dtype,
) -> list[Segment]:
    try:
        translator = NLLBTranslator(
            model_name=model_name,
            source_lang=source_lang,
            target_lang=target_lang,
            device=device,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:
        raise PipelineError(f"Failed to load translation model: {exc}") from exc

    translated: list[Segment] = []
    for segment in segments:
        try:
            text = translator.translate_text(segment.text)
        except Exception as exc:
            raise PipelineError(
                f"Translation failed at segment {segment.index}: {exc}"
            ) from exc
        translated.append(
            Segment(
                index=segment.index,
                start=segment.start,
                end=segment.end,
                text=text,
            )
        )
    return translated


def run_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir).resolve()
    ensure_directory(output_dir)

    device, whisper_compute_type, torch_dtype = detect_device()
    LOGGER.info("Device: %s", device)
    LOGGER.info("Whisper compute type: %s", whisper_compute_type)

    temp_dir = Path(tempfile.mkdtemp(prefix="yt_translate_"))
    audio_path: Optional[Path] = None

    try:
        LOGGER.info("Downloading audio...")
        audio_path, file_stem = download_audio(args.url, temp_dir)
        LOGGER.info("Audio downloaded: %s", audio_path)

        LOGGER.info("Transcribing audio with faster-whisper (%s)...", args.whisper_model)
        source_segments, detected_language = transcribe_audio(
            audio_path=audio_path,
            model_size=args.whisper_model,
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
            device=device,
            compute_type=whisper_compute_type,
        )
        resolved_source_lang = resolve_source_lang(detected_language, args.source_lang)
        LOGGER.info(
            "Detected Whisper language: %s | NLLB source language: %s",
            detected_language,
            resolved_source_lang,
        )

        LOGGER.info("Translating segments with NLLB (%s)...", args.nllb_model)
        translated_segments = translate_segments(
            segments=source_segments,
            model_name=args.nllb_model,
            source_lang=resolved_source_lang,
            target_lang=args.target_lang,
            device=device,
            torch_dtype=torch_dtype,
        )

        source_srt = output_dir / f"{file_stem}.source.srt"
        translated_srt = output_dir / f"{file_stem}.{args.target_lang}.srt"

        write_srt(source_segments, source_srt)
        write_srt(translated_segments, translated_srt)

        if args.keep_audio and audio_path.exists():
            kept_audio = output_dir / f"{file_stem}.wav"
            shutil.copy2(audio_path, kept_audio)
        LOGGER.info("Done.")

        return {
            "source_srt": source_srt,
            "translated_srt": translated_srt,
        }
    finally:
        if audio_path and audio_path.exists() and not args.keep_audio:
            try:
                audio_path.unlink()
            except OSError:
                LOGGER.warning("Could not remove temporary audio file: %s", audio_path)
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    try:
        outputs = run_pipeline(args)
    except PipelineError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130

    print(f"Source SRT: {outputs['source_srt']}")
    print(f"Translated SRT: {outputs['translated_srt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
