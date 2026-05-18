import argparse
from pathlib import Path

import streamlit as st

from youtube_translate_pipeline import run_pipeline


DEFAULT_OUTPUT_DIR = "translator_outputs"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_args(
    url: str,
    output_dir: str,
    whisper_model: str,
    source_lang: str,
    target_lang: str,
    beam_size: int,
    vad_filter: bool,
    keep_audio: bool,
    verbose: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        url=url,
        output_dir=output_dir,
        whisper_model=whisper_model,
        nllb_model="facebook/nllb-200-distilled-600M",
        source_lang=source_lang,
        target_lang=target_lang,
        beam_size=beam_size,
        vad_filter=vad_filter,
        keep_audio=keep_audio,
        verbose=verbose,
    )


def main() -> None:
    st.set_page_config(
        page_title="YouTube Translator",
        page_icon="🌐",
        layout="centered",
    )

    st.title("YouTube Video Translator")
    st.caption(
        "Download audio, transcribe with faster-whisper, translate with NLLB-200, and export SRT files locally."
    )

    with st.sidebar:
        st.header("Settings")
        whisper_model = st.selectbox("Whisper model", ["base", "small"], index=1)
        target_lang = st.text_input("Target language", value="spa_Latn")
        source_lang = st.text_input("Source language fallback", value="eng_Latn")
        beam_size = st.slider("Beam size", min_value=1, max_value=8, value=5)
        output_dir = st.text_input("Output directory", value=DEFAULT_OUTPUT_DIR)
        vad_filter = st.checkbox("Enable VAD filter", value=False)
        keep_audio = st.checkbox("Keep WAV audio file", value=False)
        verbose = st.checkbox("Verbose logs", value=False)

        st.markdown("Common NLLB targets:")
        st.code(
            "spa_Latn  # Spanish\n"
            "fra_Latn  # French\n"
            "deu_Latn  # German\n"
            "hin_Deva  # Hindi\n"
            "jpn_Jpan  # Japanese",
            language="text",
        )

    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    if st.button("Translate Video", type="primary"):
        if not url.strip():
            st.warning("Please paste a YouTube URL.")
            return

        args = build_args(
            url=url.strip(),
            output_dir=output_dir.strip() or DEFAULT_OUTPUT_DIR,
            whisper_model=whisper_model,
            source_lang=source_lang.strip() or "eng_Latn",
            target_lang=target_lang.strip() or "spa_Latn",
            beam_size=beam_size,
            vad_filter=vad_filter,
            keep_audio=keep_audio,
            verbose=verbose,
        )

        with st.spinner("Downloading, transcribing, and translating..."):
            try:
                outputs = run_pipeline(args)
            except Exception as exc:
                st.error(str(exc))
                return

        source_srt = Path(outputs["source_srt"])
        translated_srt = Path(outputs["translated_srt"])

        st.success("Translation completed.")
        st.write(f"Source SRT: `{source_srt}`")
        st.write(f"Translated SRT: `{translated_srt}`")

        with st.expander("Source subtitles", expanded=False):
            st.text(read_text(source_srt))

        with st.expander("Translated subtitles", expanded=True):
            st.text(read_text(translated_srt))

        st.download_button(
            "Download Source SRT",
            data=read_text(source_srt),
            file_name=source_srt.name,
            mime="application/x-subrip",
        )
        st.download_button(
            "Download Translated SRT",
            data=read_text(translated_srt),
            file_name=translated_srt.name,
            mime="application/x-subrip",
        )


if __name__ == "__main__":
    main()
