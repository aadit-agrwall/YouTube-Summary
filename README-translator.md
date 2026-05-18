# YouTube Translator App

This local app:

- downloads YouTube audio as WAV using `yt-dlp`
- transcribes speech with `faster-whisper`
- translates each subtitle segment with Meta NLLB-200
- exports source and translated `.srt` files
- runs on CPU by default and uses CUDA automatically if available

## Install

Make sure `ffmpeg` is installed first.

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Create or activate a virtualenv, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-translator.txt
```

## Run Web App

```bash
streamlit run translator_app.py
```

## Run CLI

```bash
python youtube_translate_pipeline.py \
  --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --target-lang spa_Latn \
  --output-dir translator_outputs
```

## Output

- `translator_outputs/<video_id>.source.srt`
- `translator_outputs/<video_id>.<target_lang>.srt`
