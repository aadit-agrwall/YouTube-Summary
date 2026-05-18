# YouTube Simple Summarizer

A small Streamlit app that:

- accepts up to 4 YouTube URLs in one run
- fetches and caches video transcripts
- uses a fast single-pass summary path for most videos
- uses a compact map-reduce path only for longer transcripts
- sends summaries to Groq through LangChain
- returns a simple but detailed explanation that grows with video length

## Architecture

1. User inputs YouTube URL
2. Transcript extractor gets the transcript
3. Transcript is cached for repeat use
4. App chooses a summary mode based on video length
5. Short and medium videos use one Groq summary call
6. Very long videos use a small map-reduce flow
7. Streamlit shows the result and keeps a small in-session history

## Setup

Create a virtual environment and install packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set a Groq API key:

```bash
export GROQ_API_KEY="your-key-here"
```

The website does not show the API key field. It reads the key from your local `.env` file or shell environment.

Optional:

```bash
export GROQ_MODEL="llama-3.1-8b-instant"
```

Run the app:

```bash
streamlit run app.py
```

## Notes

- Some YouTube videos do not expose transcripts, so those cannot be summarized with this approach.
- The current setup uses Groq.
- Some videos may still fail if YouTube blocks transcript access or the video has no transcript.
- Longer videos produce longer summaries with more sections and more explanation.
- Repeated requests for the same video are faster because the transcript and summary are cached.
- You can paste up to 4 links in one submission.
