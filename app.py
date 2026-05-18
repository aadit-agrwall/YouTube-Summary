import os
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


DEFAULT_MODEL = "llama-3.1-8b-instant"
MAX_URLS_PER_RUN = 4
DEFAULT_SUMMARY_STYLE = "Long"

load_dotenv()


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        @keyframes textReveal {
            0% { opacity: 0; transform: translateY(15px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .stApp {
            background: #000000;
            color: #ffffff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 900px;
        }
        /* Apple-like Glass Effect for cards and forms */
        .hero-card, div[data-testid="stForm"], div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 24px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
            backdrop-filter: blur(24px) !important;
            -webkit-backdrop-filter: blur(24px) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        .hero-card {
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        /* Reveal animation on expander content */
        div[data-testid="stExpanderDetails"] p, 
        div[data-testid="stExpanderDetails"] li, 
        div[data-testid="stExpanderDetails"] h1, 
        div[data-testid="stExpanderDetails"] h2, 
        div[data-testid="stExpanderDetails"] h3 {
            animation: textReveal 0.6s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
        }
        .mini-note {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 0.8rem 1rem;
            margin-bottom: 1rem;
            color: #a0a0a5;
            font-size: 0.9rem;
        }
        div[data-testid="stSidebar"] {
            background: #000000;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        label, p, div, span {
            color: #ffffff !important;
        }
        /* Input fields wrapper clear */
        div[data-baseweb="textarea"], div[data-baseweb="input"],
        div[data-baseweb="textarea"] > div, div[data-baseweb="input"] > div {
            background: transparent !important;
            border: none !important;
        }
        
        /* Input fields frozen glass */
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(25px) saturate(200%) !important;
            -webkit-backdrop-filter: blur(25px) saturate(200%) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-top: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-left: 1px solid rgba(255, 255, 255, 0.2) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), inset 0 0 10px rgba(0, 0, 0, 0.2) !important;
            border-radius: 16px !important;
            color: #ffffff !important;
            caret-color: #ffffff !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        }
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stTextInput"] input:focus {
            border-color: rgba(255, 255, 255, 0.4) !important;
            background: rgba(255, 255, 255, 0.08) !important;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4), inset 0 0 10px rgba(0, 0, 0, 0.2), 0 0 15px rgba(255, 255, 255, 0.1) !important;
        }
        /* Stronger Frozen Glass Buttons */
        button[data-testid="baseButton-primary"], 
        button[data-testid="baseButton-secondary"],
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0)) !important;
            backdrop-filter: blur(25px) saturate(200%) !important;
            -webkit-backdrop-filter: blur(25px) saturate(200%) !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            border-top: 1px solid rgba(255, 255, 255, 0.35) !important;
            border-left: 1px solid rgba(255, 255, 255, 0.25) !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
            color: #ffffff !important;
            border-radius: 999px !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
        }

        button[data-testid="baseButton-primary"]:hover, 
        button[data-testid="baseButton-secondary"]:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-2px) scale(1.02) !important;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05)) !important;
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(255,255,255,0.1) !important;
            border-top: 1px solid rgba(255, 255, 255, 0.45) !important;
            color: #ffffff !important;
        }

        button[data-testid="baseButton-primary"]:active, 
        button[data-testid="baseButton-secondary"]:active,
        div[data-testid="stFormSubmitButton"] button:active {
            transform: translateY(1px) scale(0.98) !important;
            box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.2) !important;
        }
        .stCaption {
            color: #88888e !important;
        }
        /* Mobile Specific */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1.5rem;
                padding-left: 1rem;
                padding-right: 1rem;
                padding-bottom: 2rem;
            }
            .hero-card, div[data-testid="stForm"], div[data-testid="stExpander"] {
                border-radius: 20px !important;
            }
            .hero-card h1 {
                font-size: 1.8rem !important;
                line-height: 1.15;
            }
            .mini-note {
                font-size: 0.85rem;
            }
            /* Target columns to make buttons full width on very small devices */
            div[data-testid="column"] {
                min-width: 100% !important;
                margin-bottom: 0.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()

    if "youtu.be" in host:
        return parsed.path.lstrip("/") or None

    if "youtube.com" in host:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]

        match = re.match(r"^/(shorts|embed)/([^/?]+)", parsed.path)
        if match:
            return match.group(2)

    return None


def parse_urls(raw_text: str) -> list[str]:
    found_urls = []
    seen = set()

    for line in raw_text.splitlines():
        match = re.search(r"https?://\S+", line.strip())
        if not match:
            continue

        url = match.group(0).strip()
        if url not in seen:
            found_urls.append(url)
            seen.add(url)

    return found_urls[:MAX_URLS_PER_RUN]


@st.cache_data(show_spinner=False, ttl=3600, max_entries=128)
def get_transcript_data(video_id: str) -> dict[str, object]:
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-US"])
    parts = [item.text.strip() for item in transcript if getattr(item, "text", "").strip()]
    text = " ".join(parts)

    if not text:
        raise ValueError("Transcript was empty for this video.")

    last_timestamp = 0.0
    for item in transcript:
        start = float(getattr(item, "start", 0.0) or 0.0)
        duration = float(getattr(item, "duration", 0.0) or 0.0)
        last_timestamp = max(last_timestamp, start + duration)

    duration_minutes = round(last_timestamp / 60, 1) if last_timestamp else 0.0

    return {
        "text": text,
        "duration_minutes": duration_minutes,
        "segment_count": len(parts),
    }


@st.cache_resource(show_spinner=False)
def build_llm(api_key: str, max_tokens: int) -> ChatGroq:
    if not api_key:
        raise ValueError("Missing Groq API key. Add it to your local .env file.")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        temperature=0.2,
        max_tokens=max_tokens,
        timeout=45,
    )


def get_summary_plan(
    duration_minutes: float,
    transcript_length: int,
    summary_style: str,
) -> dict[str, object]:
    if duration_minutes >= 20 or transcript_length >= 18000:
        plan = {
            "label": "deep",
            "section_count": 6,
            "takeaway_count": 5,
            "max_tokens": 1100,
            "chunk_size": 2400,
            "sample_count": 5,
            "snippet_chars": 700,
        }
    elif duration_minutes >= 8 or transcript_length >= 7000:
        plan = {
            "label": "detailed",
            "section_count": 5,
            "takeaway_count": 5,
            "max_tokens": 900,
            "chunk_size": 2200,
            "sample_count": 4,
            "snippet_chars": 650,
        }
    else:
        plan = {
            "label": "standard",
            "section_count": 4,
            "takeaway_count": 4,
            "max_tokens": 700,
            "chunk_size": 2000,
            "sample_count": 3,
            "snippet_chars": 550,
        }

    if summary_style == "Short":
        plan["label"] = f"{plan['label']}-short"
        plan["section_count"] = max(3, int(plan["section_count"]) - 2)
        plan["takeaway_count"] = max(3, int(plan["takeaway_count"]) - 1)
        plan["max_tokens"] = max(500, int(plan["max_tokens"]) - 250)
        plan["sample_count"] = max(2, int(plan["sample_count"]) - 1)
        plan["snippet_chars"] = max(380, int(plan["snippet_chars"]) - 160)
    else:
        plan["label"] = f"{plan['label']}-long"

    return plan


def run_prompt(prompt: ChatPromptTemplate, payload: dict[str, object], api_key: str, max_tokens: int) -> str:
    last_error = None

    for token_limit in (max_tokens, max(900, max_tokens - 500)):
        try:
            llm = build_llm(api_key, token_limit)
            chain = prompt | llm | StrOutputParser()
            return chain.invoke(payload)
        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error

    raise RuntimeError("Summary generation failed.")


def build_representative_context(
    transcript_text: str,
    chunk_size: int,
    sample_count: int,
    snippet_chars: int,
) -> tuple[str, int]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
    )
    docs = splitter.create_documents([transcript_text])

    if not docs:
        return transcript_text[:snippet_chars], 1

    if len(docs) <= sample_count:
        selected_indices = list(range(len(docs)))
    else:
        selected_indices = []
        for i in range(sample_count):
            index = round(i * (len(docs) - 1) / max(1, sample_count - 1))
            if index not in selected_indices:
                selected_indices.append(index)

    parts = []
    for order, doc_index in enumerate(selected_indices, start=1):
        snippet = docs[doc_index].page_content[:snippet_chars].strip()
        parts.append(f"Transcript Snippet {order}:\n{snippet}")

    return "\n\n".join(parts), len(docs)


def is_rate_limit_error(exc: Exception) -> bool:
    return "rate limit" in str(exc).lower() or "429" in str(exc)


def summarize_single_pass(
    context_text: str,
    duration_minutes: float,
    api_key: str,
    plan: dict[str, object],
    docs_count: int,
    summary_style: str,
) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
        You are a patient teacher.
        Explain this YouTube video in very simple English for a beginner.

        Video length: {duration_minutes} minutes
        Depth mode: {depth_label}
        Summary style: {summary_style}
        Transcript parts: {docs_count}
        Write {section_count} numbered sections and {takeaway_count} takeaways.

        Rules:
        - Use simple English.
        - Be clear, reliable, and not too short.
        - Cover the beginning, middle, and end of the video.
        - Include the main ideas, examples, lessons, and warnings if they exist.
        - If summary style is Long, explain with more detail.
        - If summary style is Short, keep it compact but still useful.
        - Make the answer longer when the video is longer.
        - The transcript below is a representative set of snippets from the full video.
        - Build the answer from those snippets without inventing facts.
        - Stay grounded in the transcript only.
        - Return the answer in exactly this format:

        Video Overview:
        <3 to 5 sentences>

        Detailed Explanation:
        1. <section title>: <2 to 4 sentences>
        2. <section title>: <2 to 4 sentences>
        3. <section title>: <2 to 4 sentences>
        4. <section title>: <2 to 4 sentences>
        5. <section title>: <2 to 4 sentences if needed>
        6. <section title>: <2 to 4 sentences if needed>

        Important Takeaways:
        - <takeaway 1>
        - <takeaway 2>
        - <takeaway 3>
        - <takeaway 4>
        - <takeaway 5 if needed>
        - <takeaway 6 if needed>

        Simple Example:
        <one easy example that helps a beginner understand the video>

        Who This Video Helps:
        <1 to 2 sentences>

        Transcript Snippets:
        {context_text}
        """
    )

    return run_prompt(
        prompt,
        {
            "duration_minutes": duration_minutes,
            "depth_label": plan["label"],
            "summary_style": summary_style,
            "docs_count": docs_count,
            "section_count": plan["section_count"],
            "takeaway_count": plan["takeaway_count"],
            "context_text": context_text,
        },
        api_key,
        int(plan["max_tokens"]),
    )


@st.cache_data(show_spinner=False, ttl=3600, max_entries=128)
def summarize_video(
    video_id: str,
    transcript_text: str,
    duration_minutes: float,
    _api_key: str,
    summary_style: str,
) -> dict[str, object]:
    plan = get_summary_plan(duration_minutes, len(transcript_text), summary_style)
    context_text, docs_count = build_representative_context(
        transcript_text,
        int(plan["chunk_size"]),
        int(plan["sample_count"]),
        int(plan["snippet_chars"]),
    )

    try:
        summary = summarize_single_pass(
            context_text,
            duration_minutes,
            _api_key,
            plan,
            docs_count,
            summary_style,
        )
    except Exception as exc:
        if not is_rate_limit_error(exc):
            raise

        fallback_plan = dict(plan)
        fallback_plan["max_tokens"] = max(600, int(plan["max_tokens"]) - 250)
        fallback_plan["section_count"] = max(4, int(plan["section_count"]) - 1)
        fallback_plan["takeaway_count"] = max(4, int(plan["takeaway_count"]) - 1)
        smaller_context, docs_count = build_representative_context(
            transcript_text,
            int(plan["chunk_size"]),
            max(3, int(plan["sample_count"]) - 1),
            max(450, int(plan["snippet_chars"]) - 150),
        )
        summary = summarize_single_pass(
            smaller_context,
            duration_minutes,
            _api_key,
            fallback_plan,
            docs_count,
            summary_style,
        )

    return {
        "video_id": video_id,
        "summary": summary,
        "duration_minutes": duration_minutes,
        "chunk_count": docs_count,
        "depth_label": str(plan["label"]),
        "summary_style": summary_style,
        "strategy": "fast-context",
    }


def render_history() -> None:
    history = st.session_state.get("summary_history", [])
    if not history:
        return

    st.subheader("Recent Summaries")
    for item in reversed(history[-6:]):
        with st.expander(item["title"]):
            st.caption(item["meta"])
            st.write(item["summary"])


def summarize_urls(urls: list[str], api_key: str, summary_style: str) -> list[dict[str, object]]:
    results = []
    progress = st.progress(0.0, text="Starting...")
    status_box = st.empty()

    for index, url in enumerate(urls, start=1):
        status_box.info(f"Processing video {index} of {len(urls)}")
        video_id = extract_video_id(url)

        if not video_id:
            results.append(
                {
                    "url": url,
                    "error": "Could not read this YouTube URL.",
                }
            )
            progress.progress(index / len(urls), text=f"Finished {index} of {len(urls)}")
            continue

        try:
            transcript_data = get_transcript_data(video_id)
            result = summarize_video(
                video_id,
                str(transcript_data["text"]),
                float(transcript_data["duration_minutes"]),
                api_key,
                summary_style,
            )
            results.append(
                {
                    "url": url,
                    "result": result,
                }
            )
        except Exception as exc:
            error_name = exc.__class__.__name__

            if error_name in {
                "NoTranscriptFound",
                "TranscriptsDisabled",
                "RequestBlocked",
                "IpBlocked",
            }:
                message = "Transcript is missing, disabled, or blocked for this video."
            elif error_name == "VideoUnavailable":
                message = "This video is unavailable."
            else:
                message = str(exc)

            results.append(
                {
                    "url": url,
                    "error": message,
                }
            )

        progress.progress(index / len(urls), text=f"Finished {index} of {len(urls)}")

    status_box.success("All videos processed.")
    return results


def main() -> None:
    st.set_page_config(
        page_title="YouTube Simple Summarizer",
        page_icon="▶️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    apply_app_styles()

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin:0 0 0.35rem 0;">YouTube Transcript Summarizer</h1>
            <p style="margin:0; font-size:1rem;">
                Fast summaries for videos that already have transcripts.
                Paste up to 4 links. Longer videos get longer explanations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            st.success("App ready")
        else:
            st.error("App is missing local server config")

        st.subheader("How It Works")
        st.markdown(
            "- Works only for videos with transcripts\n"
            "- Same video is cached for 1 hour\n"
            "- Duplicate links are skipped automatically\n"
            "- Up to 4 links can run in one batch\n"
            "- Output length grows with video length"
        )

    if "summary_history" not in st.session_state:
        st.session_state["summary_history"] = []

    with st.form("youtube_summary_form"):
        st.markdown(
            """
            <div class="mini-note">
                Paste one YouTube link per line. The app skips videos with missing or blocked transcripts.
            </div>
            """,
            unsafe_allow_html=True,
        )
        raw_urls = st.text_area(
            "YouTube link(s)",
            height=120,
            placeholder="Paste one YouTube URL per line",
            label_visibility="collapsed"
        )
        col1, col2 = st.columns(2)
        with col1:
            long_clicked = st.form_submit_button("Summarize (Long - Default)", type="primary", use_container_width=True)
        with col2:
            short_clicked = st.form_submit_button("Summarize (Short)", use_container_width=True)
            
        submitted = long_clicked or short_clicked
        summary_style = "Short" if short_clicked else "Long"

    if submitted:
        urls = parse_urls(raw_urls)

        if not api_key:
            st.error("Groq API key is missing in your local .env file.")
            return

        if not raw_urls.strip():
            st.warning("Please paste at least one YouTube link.")
            return

        if not urls:
            st.error("Could not find a valid YouTube URL.")
            return

        with st.spinner("Fetching transcripts and generating summaries..."):
            results = summarize_urls(urls, api_key, summary_style)

        st.subheader("Results")
        for item in results:
            if "error" in item:
                st.error(f"{item['url']}\n\n{item['error']}")
                continue

            result = item["result"]
            meta = (
                f"Video length: {result['duration_minutes']} minutes | "
                f"Chunks: {result['chunk_count']} | "
                f"Mode: {result['summary_style']} | "
                f"Depth: {result['depth_label']} | "
                f"Strategy: {result['strategy']}"
            )

            with st.expander(item["url"], expanded=True):
                st.caption(meta)
                st.write(result["summary"])

            st.session_state["summary_history"].append(
                {
                    "title": item["url"],
                    "meta": meta,
                    "summary": result["summary"],
                }
            )

    render_history()


if __name__ == "__main__":
    main()
