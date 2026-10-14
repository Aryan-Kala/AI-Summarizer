import base64
import os
import re
import tempfile
import urllib.request
from datetime import datetime
import traceback
from urllib.parse import urlparse, parse_qs
from yt_dlp.networking.impersonate import ImpersonateTarget

import streamlit as st
import streamlit.components.v1 as components
import yt_dlp
from google import genai
from google.genai import types
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from fpdf import FPDF
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv("pass.env")

GEMINI_MODEL = "gemini-2.5-flash"  # gemini-1.5-* models are retired

# Edit these defaults to change how every summary is written
DEFAULT_INSTRUCTIONS = """You are an expert summarizer who turns video transcripts into study notes.

Rules:
- Make the summary broad and detailed: cover every major topic, concept, example, formula, step and conclusion. Do not leave out important sections.
- Write everything as bullet points (pointers). Start each bullet with "- ". Use short section headings in plain text, with bullets beneath each heading.
- Never mention the name of the educator, teacher, instructor, speaker, host, channel, academy or institute. If someone must be referred to, say "the instructor" or "the speaker".
- Do not include greetings, intros, subscribe requests, promotions or other filler.
- Do not use markdown symbols like ** or #. Plain text and "- " bullets only.
- Write the summary in English, even if the transcript is in another language.
"""

THEMES = {
    "Navy & Gold":     {"primary": (18, 32, 64),  "accent": (191, 155, 48)},
    "Emerald & Cream": {"primary": (14, 74, 62),  "accent": (201, 164, 92)},
    "Charcoal & Rose": {"primary": (38, 38, 43),  "accent": (190, 80, 100)},
}
BODY = (45, 45, 52)
MUTED = (125, 125, 135)

# --- UI Configuration ---
st.set_page_config(page_title="YouTube to PDF Summarizer", page_icon="🎥")
st.title("🎥 YouTube to PDF Summarizer (Hybrid)")
st.write("Converts any video to a PDF. Automatically falls back to AI transcription if subtitles are disabled.")

# --- API Key Management ---
env_api_key = os.getenv("GEMINI_API_KEY")

st.sidebar.header("Configuration")
if env_api_key:
    api_key = env_api_key
    st.sidebar.success("🔑 API Key loaded securely from pass.env")
else:
    api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")
    st.sidebar.markdown("Get your free API key from [Google AI Studio](https://aistudio.google.com/).")


st.sidebar.header("Summary Instructions")
custom_instructions = st.sidebar.text_area(
    "Instructions for the AI summarizer:",
    value=DEFAULT_INSTRUCTIONS,
    height=300,
)


st.sidebar.header("PDF Style")
theme_name = st.sidebar.selectbox("Colour theme:", list(THEMES.keys()))


# --- Caching AI Model ---
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")


# --- Helper Functions ---
def extract_video_id(url):
    """Handles watch?v=, youtu.be, shorts, embed, live and extra query params."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().replace("www.", "")
    vid = None

    if host == "youtu.be":
        vid = parsed.path.lstrip("/")
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [None])[0]
        else:
            m = re.match(r"^/(shorts|embed|live|v)/([^/?]+)", parsed.path)
            if m:
                vid = m.group(2)

    if vid:
        vid = vid[:11]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
            return vid
    return None
def fetch_transcript(video_id):
    """youtube-transcript-api >= 1.0: get_transcript() was removed and
    snippets are objects (.text), not dicts (['text'])."""
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    try:
        transcript = transcript_list.find_transcript(["en"])
    except Exception:
        transcript = next(iter(transcript_list))  # any available language
    fetched = transcript.fetch()
    return " ".join(snippet.text for snippet in fetched)

def download_audio(youtube_url, temp_dir):
    """Downloads audio and returns the final .mp3 path.
    Uses IPv4 and TV clients to bypass cloud IP blocks."""
    base_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(temp_dir, "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "retries": 3,
        "source_address": "0.0.0.0",
        "impersonate": ImpersonateTarget(client='chrome'),
        "extractor_args": {"youtube": ["player_client=tv"]},
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    if any(os.path.exists(f) for f in ("ffmpeg.exe", "ffmpeg")):
        base_opts["ffmpeg_location"] = os.path.abspath(".")

    with yt_dlp.YoutubeDL(base_opts) as ydl:
        ydl.extract_info(youtube_url, download=True)
        
    return os.path.join(temp_dir, "audio.mp3")

# --- Premium PDF fonts (Lato, auto-downloaded once into ./fonts) ---
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/"
FONT_FILES = ["Lato-Regular.ttf", "Lato-Bold.ttf", "Lato-Italic.ttf", "Lato-Light.ttf", "Lato-Black.ttf"]


@st.cache_resource
def ensure_fonts():
    """Returns True if all font files are available (downloads them if missing)."""
    os.makedirs(FONT_DIR, exist_ok=True)
    try:
        for name in FONT_FILES:
            path = os.path.join(FONT_DIR, name)
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                urllib.request.urlretrieve(FONT_URL + name, path)
        return True
    except Exception:
        return False



def clean_markdown(text, unicode_ok):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = text.replace("`", "")
    if unicode_ok:
        return text.replace("\u00a0", " ")
    repl = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2022": "-", "\u2026": "...", "\u00a0": " "}
    for a, b in repl.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


class PremiumPDF(FPDF):
    def __init__(self, theme, fonts_ok, footer_text="Video Summary"):
        super().__init__(format="A4")
        self.theme, self.fonts_ok, self.footer_text = theme, fonts_ok, footer_text
        self.set_margins(20, 18, 20)
        self.set_auto_page_break(True, margin=22)
        if fonts_ok:
            p = lambda f: os.path.join(FONT_DIR, f)
            self.add_font("Lato", "", p("Lato-Regular.ttf"))
            self.add_font("Lato", "B", p("Lato-Bold.ttf"))
            self.add_font("Lato", "I", p("Lato-Italic.ttf"))
            self.add_font("LatoLight", "", p("Lato-Light.ttf"))
            self.add_font("LatoBlack", "", p("Lato-Black.ttf"))
            self.f_body, self.f_light, self.f_black = "Lato", "LatoLight", "LatoBlack"
        else:
            self.f_body = self.f_light = self.f_black = "helvetica"

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*self.theme["accent"])
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_y(-11)
        self.set_font(self.f_light, size=8.5)
        self.set_text_color(*MUTED)
        self.cell(0, 6, clean_markdown(self.footer_text, self.fonts_ok), align="L")
        self.set_x(self.l_margin)
        self.cell(0, 6, f"{self.page_no()}", align="R")


def generate_pdf(summary_text, filepath, title="Video Summary", footer_text="Video Summary", theme_name="Navy & Gold", fonts_ok=True):
    theme = THEMES.get(theme_name, THEMES["Navy & Gold"])
    pdf = PremiumPDF(theme, fonts_ok, footer_text=footer_text)
    pdf.add_page()

    # Cover banner
    pdf.set_fill_color(*theme["primary"])
    pdf.rect(0, 0, pdf.w, 52, style="F")
    pdf.set_fill_color(*theme["accent"])
    pdf.rect(0, 52, pdf.w, 1.6, style="F")

    pdf.set_xy(pdf.l_margin, 17)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(pdf.f_black, size=26 if len(title) <= 36 else 19)
    pdf.multi_cell(0, 11, clean_markdown(title, fonts_ok), new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin)
    pdf.set_text_color(*theme["accent"])
    pdf.set_font(pdf.f_light, size=10.5)
    pdf.cell(0, 8, datetime.now().strftime("%d %B %Y").upper() + "   |   KEY NOTES", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(66)

    for raw in clean_markdown(summary_text, fonts_ok).splitlines():
        if not raw.strip():
            continue
        m = re.match(r"^(\s*)[-*\u2022]\s+(.*)$", raw)
        if m:  # bullet
            level = 1 if len(m.group(1).replace("\t", "    ")) >= 2 else 0
            x0 = pdf.l_margin + 3 + level * 8
            glyph = ("\u2022" if level == 0 else "\u2013") if fonts_ok else "-"
            pdf.set_x(x0)
            pdf.set_text_color(*theme["accent"])
            pdf.set_font(pdf.f_body, "B", 11)
            pdf.cell(5, 6.4, glyph, new_x="RIGHT", new_y="TOP")
            pdf.set_text_color(*BODY)
            pdf.set_font(pdf.f_body, "", 10.8)
            pdf.multi_cell(0, 6.4, m.group(2).strip(), new_x="LMARGIN", new_y="NEXT", align="L")
            pdf.ln(1.6)
        else:
            line = raw.strip()
            if len(line) > 90:  # paragraph
                pdf.set_text_color(*BODY)
                pdf.set_font(pdf.f_body, "", 10.8)
                pdf.multi_cell(0, 6.4, line, new_x="LMARGIN", new_y="NEXT", align="L")
                pdf.ln(2)
                continue
            if pdf.get_y() > pdf.h - 50:
                pdf.add_page()
            pdf.ln(4)
            pdf.set_text_color(*theme["primary"])
            pdf.set_font(pdf.f_body, "B", 14)
            pdf.cell(0, 8, line.rstrip(":"), new_x="LMARGIN", new_y="NEXT")
            y = pdf.get_y()
            pdf.set_draw_color(*theme["accent"])
            pdf.set_line_width(0.7)
            pdf.line(pdf.l_margin, y, pdf.l_margin + 18, y)
            pdf.ln(3.5)

    pdf.output(filepath)


def render_pdf_preview(pdf_bytes, height=700):
    """Renders an interactive, multi-page PDF preview using PDF.js on HTML5 canvases.
    Bypasses browser restrictions on data:application/pdf iframes."""
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                padding: 16px;
                background: #1e1e24;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                color: #e0e0e0;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            #toolbar {{
                width: 100%;
                max-width: 820px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: #2b2b36;
                padding: 8px 14px;
                border-radius: 6px;
                margin-bottom: 14px;
                font-size: 13px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
            }}
            #toolbar button {{
                background: #3a3a48;
                color: #fff;
                border: 1px solid #505062;
                border-radius: 4px;
                padding: 4px 10px;
                cursor: pointer;
                font-size: 12px;
            }}
            #toolbar button:hover {{
                background: #4b4b5e;
            }}
            #pdf-container {{
                width: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 16px;
            }}
            .pdf-page {{
                background: white;
                box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
                border-radius: 4px;
                max-width: 100%;
                height: auto !important;
            }}
            #status {{
                padding: 24px;
                text-align: center;
                color: #b0b0c0;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    </head>
    <body>
        <div id="toolbar">
            <span id="page-info">Loading PDF preview...</span>
            <button id="open-tab-btn" type="button">Open in New Tab ↗</button>
        </div>
        <div id="status">Rendering pages...</div>
        <div id="pdf-container"></div>

        <script>
            const b64Data = "{b64_pdf}";
            const raw = atob(b64Data);
            const uint8Array = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; i++) {{
                uint8Array[i] = raw.charCodeAt(i);
            }}

            document.getElementById("open-tab-btn").addEventListener("click", () => {{
                const blob = new Blob([uint8Array], {{ type: "application/pdf" }});
                const url = URL.createObjectURL(blob);
                window.open(url, "_blank");
            }});

            const pdfjsLib = window["pdfjs-dist/build/pdf"];
            if (!pdfjsLib) {{
                document.getElementById("status").textContent =
                    "Could not load PDF renderer (check internet connection). Use 'Open in New Tab' above.";
            }} else {{
                pdfjsLib.GlobalWorkerOptions.workerSrc =
                    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

                pdfjsLib.getDocument({{ data: uint8Array }}).promise.then(async (pdf) => {{
                    document.getElementById("status").style.display = "none";
                    document.getElementById("page-info").textContent =
                        `Preview (${{pdf.numPages}} ${{pdf.numPages === 1 ? "page" : "pages"}})`;
                    const container = document.getElementById("pdf-container");
                    const dpr = window.devicePixelRatio || 1;
                    const scale = 1.35 * dpr;

                    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {{
                        const page = await pdf.getPage(pageNum);
                        const viewport = page.getViewport({{ scale }});
                        const canvas = document.createElement("canvas");
                        canvas.className = "pdf-page";
                        canvas.width = viewport.width;
                        canvas.height = viewport.height;
                        canvas.style.width = `${{viewport.width / dpr}}px`;
                        container.appendChild(canvas);

                        const ctx = canvas.getContext("2d");
                        await page.render({{ canvasContext: ctx, viewport }}).promise;
                    }}
                }}).catch((err) => {{
                    document.getElementById("status").textContent =
                        "Failed to render PDF preview: " + err.message;
                }});
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=True)


# --- Main App Logic ---
youtube_url = st.text_input("Paste YouTube URL here:")

if st.button("Generate Summary & PDF"):
    st.session_state.pop("result", None)

    if not api_key:
        st.error("Please provide a Gemini API Key in your pass.env file or sidebar.")
    elif not youtube_url:
        st.warning("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(youtube_url)

        if not video_id:
            st.error("Could not extract a valid YouTube Video ID. Please check the URL format.")
        else:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    transcript = ""

                    # --- APPROACH 1: Fast API Fetch ---
                    try:
                        with st.spinner("Attempting to fetch transcript directly from YouTube servers..."):
                            transcript = fetch_transcript(video_id)
                            st.toast("⚡ Fast transcript fetched successfully!")

                    # --- APPROACH 2: Audio Download + Faster Whisper Fallback ---
                    except Exception as e:
                        st.warning(f"Subtitles unavailable ({type(e).__name__}). Falling back to AI audio transcription...")

                        with st.spinner("Downloading audio..."):
                            final_audio_path = download_audio(youtube_url, temp_dir)

                        with st.spinner("Transcribing audio locally with AI (this may take a minute)..."):
                            model = load_whisper_model()
                            segments, info = model.transcribe(final_audio_path, beam_size=5)
                            transcript = " ".join(segment.text for segment in segments)

                    if not transcript.strip():
                        raise ValueError("Transcript is empty - nothing to summarize.")

                    # --- Step 3: Summarize ---
                    with st.spinner("Summarizing transcript with Gemini..."):
                        client = genai.Client(api_key=api_key)
                        prompt = f"Summarize the following video transcript.\n\nTranscript:\n{transcript}"
                        
                        response = client.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=custom_instructions,
                            )
                        )
                        summary = response.text

                    # Seed the editable text box with the new summary
                    st.session_state["result"] = {"summary": summary}
                    st.session_state["edited_summary"] = summary
                    st.session_state["pdf_title"] = "Video Summary"
                    st.session_state["pdf_footer"] = "Video Summary"

            except Exception:
                st.error("A background component crashed. See the exact details below:")
                st.code(traceback.format_exc())

# --- Edit, Preview & Download ---
if "result" in st.session_state:
    st.success("Summary generated! Edit it below, then download your PDF.")

    if "pdf_footer" not in st.session_state:
        st.session_state["pdf_footer"] = "Video Summary"

    pdf_title = st.text_input("PDF title:", key="pdf_title")
    pdf_footer = st.text_input("PDF footer:", key="pdf_footer")
    st.caption('Tip: start a line with "- " for a bullet. A short line without "- " becomes a section heading.')
    edited = st.text_area("Edit your summary:", key="edited_summary", height=450)

    fonts_ok = ensure_fonts()
    if not fonts_ok:
        st.warning("Could not download the premium font (offline?). Using the basic font instead.")

    if edited.strip():
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = os.path.join(tmp, "summary.pdf")
                generate_pdf(edited, pdf_path, title=pdf_title or "Video Summary",
                             footer_text=pdf_footer, theme_name=theme_name, fonts_ok=fonts_ok)
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

            st.download_button(
                label="📄 Download PDF Summary",
                data=pdf_bytes,
                file_name="video_summary.pdf",
                mime="application/pdf",
            )

            with st.expander("👁 Preview PDF", expanded=True):
                render_pdf_preview(pdf_bytes)

        except Exception:
            st.error("Could not build the PDF. Details below:")
            st.code(traceback.format_exc())
    else:
        st.info("The summary is empty - add some text to create a PDF.")