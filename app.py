import base64
import json
import os
import re
import tempfile
import urllib.request
from datetime import datetime
import traceback
from urllib.parse import urlparse, parse_qs, quote

import streamlit as st
import streamlit.components.v1 as components
import yt_dlp
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from fpdf import FPDF
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig

# Load environment variables
load_dotenv("pass.env")

# Tried in order; the lighter model takes over when the first is overloaded (503)
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]  # gemini-1.5-* models are retired
GEMINI_BUSY_CODES = (429, 500, 503, 504)

# Optional: a residential proxy (e.g. http://user:pass@host:port) for the
# transcript API and yt-dlp. YouTube blocks most cloud/datacenter IPs.
PROXY_URL = os.getenv("PROXY_URL")

# Edit these defaults to change how every summary is written
DEFAULT_INSTRUCTIONS = """You are an expert summarizer who turns video transcripts into study notes.

Rules:
- Make the summary broad and detailed: cover every major topic, concept, example, formula, step and conclusion. Do not leave out important sections.
- Write everything as bullet points (pointers). Start each bullet with "- ". Use short section headings in plain text, with bullets beneath each heading.
- Never mention the name of the educator, teacher, instructor, speaker, host, channel, academy or institute. If someone must be referred to, say "the instructor" or "the speaker".
- Do not include greetings, intros, subscribe requests, promotions or other filler.
- Do not use markdown symbols like ** or #. Plain text and "- " bullets only.
"""

# The selected language is appended to the instructions at request time
LANGUAGES = [
    "English", "Hindi", "Spanish", "French", "German", "Portuguese", "Italian",
    "Dutch", "Russian", "Turkish", "Indonesian", "Vietnamese", "Arabic", "Urdu",
    "Bengali", "Marathi", "Gujarati", "Punjabi", "Tamil", "Telugu", "Kannada",
    "Malayalam", "Thai", "Chinese (Simplified)", "Japanese", "Korean",
]

THEMES = {
    "Navy & Gold":     {"primary": (18, 32, 64),  "accent": (191, 155, 48)},
    "Emerald & Cream": {"primary": (14, 74, 62),  "accent": (201, 164, 92)},
    "Charcoal & Rose": {"primary": (38, 38, 43),  "accent": (190, 80, 100)},
}
DEFAULT_FOOTER = "Aryan Kala [ IIT Madras]"
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
summary_language = st.sidebar.selectbox("Summary language:", LANGUAGES)


st.sidebar.header("PDF Style")
theme_name = st.sidebar.selectbox("Colour theme:", list(THEMES.keys()))
custom_title = st.sidebar.text_input("PDF title:", placeholder="Leave blank to use the video title")
default_footer = st.sidebar.text_input("PDF footer:", value=DEFAULT_FOOTER)


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
    proxy = GenericProxyConfig(http_url=PROXY_URL, https_url=PROXY_URL) if PROXY_URL else None
    api = YouTubeTranscriptApi(proxy_config=proxy)
    transcript_list = api.list(video_id)
    try:
        transcript = transcript_list.find_transcript(["en"])
    except Exception:
        transcript = next(iter(transcript_list))  # any available language
    fetched = transcript.fetch()
    return " ".join(snippet.text for snippet in fetched)


def fetch_video_title(video_id):
    """oEmbed is a lightweight public endpoint that isn't behind YouTube's
    bot checks, so it usually works even from cloud IPs."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with urllib.request.urlopen(f"https://www.youtube.com/oembed?format=json&url={quote(video_url, safe='')}", timeout=10) as r:
            return json.load(r).get("title")
    except Exception:
        return None


def safe_filename(title):
    """Strips characters that Windows/macOS don't allow in file names."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", title)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return (name[:150] or "video_summary") + ".pdf"

def download_audio(youtube_url, temp_dir):
    """Downloads the audio track with yt-dlp. faster-whisper decodes any
    audio format itself, so no ffmpeg conversion step is needed."""
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(temp_dir, "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "retries": 3,
        "source_address": "0.0.0.0",  # force IPv4
        # yt-dlp only enables deno by default; packages.txt installs node,
        # which is needed to solve YouTube's JS signature challenges.
        "js_runtimes": {"deno": {}, "node": {}},
        # Values must be lists of strings (a plain list is silently ignored)
        "extractor_args": {"youtube": {"player_client": ["default", "tv", "web_safari"]}},
    }
    if PROXY_URL:
        opts["proxy"] = PROXY_URL

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

    return info["requested_downloads"][0]["filepath"], info.get("title")


class GeminiBusyError(RuntimeError):
    pass


def make_client():
    # The SDK retries 503/429 itself (5 attempts by default); fewer attempts
    # per model means we switch to the fallback model sooner.
    retry = types.HttpRetryOptions(attempts=3, initial_delay=2)
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=retry))


def generate(client, contents, config):
    last_error = None
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except genai_errors.APIError as e:
            if e.code not in GEMINI_BUSY_CODES:
                raise
            last_error = e
            continue
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response (it may have been blocked by safety filters).")
        return text
    if last_error.code == 429:
        raise GeminiBusyError("Gemini rate limit or daily quota reached. Wait a minute and click Generate again.")
    raise GeminiBusyError("Gemini is overloaded right now (high demand on Google's side). Wait a minute and click Generate again.")


def system_instruction():
    return (
        f"{custom_instructions.strip()}\n"
        f"- Write the entire summary, including headings, in {summary_language}, even if the video is in another language. "
        "Keep formulas, code and technical terms that have no common translation as they are."
    )


def summarize_transcript(client, transcript):
    return generate(
        client,
        contents=f"Summarize the following video transcript.\n\nTranscript:\n{transcript}",
        config=types.GenerateContentConfig(system_instruction=system_instruction()),
    )


def summarize_youtube_url(client, youtube_url):
    """Gemini fetches public YouTube videos itself, so this never hits
    YouTube's block on our server's IP. Low resolution keeps long videos
    within the context window (the summary relies mostly on the audio)."""
    return generate(
        client,
        contents=types.Content(parts=[
            types.Part(file_data=types.FileData(file_uri=youtube_url)),
            types.Part(text="Summarize this video based on everything that is said and shown in it."),
        ]),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction(),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        ),
    )


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


# Lato only covers Latin. Text in other scripts falls back to the matching
# Noto font, downloaded only when the PDF actually contains that script.
NOTO_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
NOTO_LATIN = "notosans/NotoSans[wdth,wght].ttf"
NOTO_SC = "notosanssc/NotoSansSC[wght].ttf"
SCRIPT_FONTS = [  # (first codepoint, last codepoint, google/fonts file)
    (0x0370, 0x052F, NOTO_LATIN),                                    # Greek, Cyrillic
    (0x1E00, 0x1EFF, NOTO_LATIN),                                    # Vietnamese
    (0x0600, 0x06FF, "notonaskharabic/NotoNaskhArabic[wght].ttf"),   # Arabic, Urdu
    (0x0750, 0x077F, "notonaskharabic/NotoNaskhArabic[wght].ttf"),
    (0xFB50, 0xFEFF, "notonaskharabic/NotoNaskhArabic[wght].ttf"),
    (0x0900, 0x097F, "notosansdevanagari/NotoSansDevanagari[wdth,wght].ttf"),
    (0x0980, 0x09FF, "notosansbengali/NotoSansBengali[wdth,wght].ttf"),
    (0x0A00, 0x0A7F, "notosansgurmukhi/NotoSansGurmukhi[wdth,wght].ttf"),
    (0x0A80, 0x0AFF, "notosansgujarati/NotoSansGujarati[wdth,wght].ttf"),
    (0x0B80, 0x0BFF, "notosanstamil/NotoSansTamil[wdth,wght].ttf"),
    (0x0C00, 0x0C7F, "notosanstelugu/NotoSansTelugu[wdth,wght].ttf"),
    (0x0C80, 0x0CFF, "notosanskannada/NotoSansKannada[wdth,wght].ttf"),
    (0x0D00, 0x0D7F, "notosansmalayalam/NotoSansMalayalam[wdth,wght].ttf"),
    (0x0E00, 0x0E7F, "notosansthai/NotoSansThai[wdth,wght].ttf"),
    (0x3040, 0x30FF, "notosansjp/NotoSansJP[wght].ttf"),            # Japanese kana
    (0x1100, 0x11FF, "notosanskr/NotoSansKR[wght].ttf"),            # Korean hangul
    (0x3130, 0x318F, "notosanskr/NotoSansKR[wght].ttf"),
    (0xAC00, 0xD7AF, "notosanskr/NotoSansKR[wght].ttf"),
    (0x3000, 0x303F, NOTO_SC),                                       # CJK punctuation
    (0x3400, 0x9FFF, NOTO_SC),                                       # Chinese characters
    (0xFF00, 0xFFEF, NOTO_SC),
]
CJK_FONTS = ("notosansjp/NotoSansJP[wght].ttf", "notosanskr/NotoSansKR[wght].ttf")


def needed_script_fonts(text):
    files = []
    for ch in set(text):
        cp = ord(ch)
        for lo, hi, font in SCRIPT_FONTS:
            if lo <= cp <= hi and font not in files:
                files.append(font)
    # The Japanese/Korean fonts also contain Chinese characters (kanji/hanja)
    if any(f in files for f in CJK_FONTS) and NOTO_SC in files:
        files.remove(NOTO_SC)
    return files


@st.cache_resource
def ensure_script_font(font):
    """Returns (regular_path, bold_path) for a Noto font, or None if it can't be fetched.
    Google ships variable fonts; static weights are cut once and kept on disk because
    fpdf2 would otherwise redo that on every PDF build (~10s for the CJK fonts)."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    variable = os.path.join(FONT_DIR, os.path.basename(font))
    stem = os.path.splitext(os.path.basename(font))[0].split("[")[0]
    paths = tuple(os.path.join(FONT_DIR, f"{stem}-{w}.ttf") for w in (400, 700))
    try:
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths):
            return paths
        if not os.path.exists(variable) or os.path.getsize(variable) == 0:
            urllib.request.urlretrieve(NOTO_URL + quote(font), variable)
        vf = TTFont(variable)
        defaults = {axis.axisTag: axis.defaultValue for axis in vf["fvar"].axes}
        for weight, path in zip((400, 700), paths):
            instantiateVariableFont(vf, {**defaults, "wght": weight}, static=True).save(path)
        os.remove(variable)
        return paths
    except Exception:
        return None


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
    def __init__(self, theme, fonts_ok, footer_text=DEFAULT_FOOTER, script_fonts=()):
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

            fallbacks = []
            for i, (regular, bold) in enumerate(script_fonts):
                family = f"Script{i}"
                self.add_font(family, "", regular)
                self.add_font(family, "B", bold)
                fallbacks.append(family)
            if fallbacks:
                self.set_fallback_fonts(fallbacks)
                # Joins letters correctly for Indic/Arabic scripts and lays out right-to-left text
                self.set_text_shaping(True)
        else:
            self.f_body = self.f_light = self.f_black = "helvetica"

    def _render_styled_text_line(self, *args, **kwargs):
        # fpdf2 2.8.9 bug: after a fallback-font fragment the PDF keeps that font
        # active, but fpdf2 assumes the main font is, so a following line in the
        # main font (e.g. English words in a Hindi summary) renders invisibly.
        # Forcing the font to be re-selected for every line avoids it.
        self.current_font_is_set_on_page = False
        return super()._render_styled_text_line(*args, **kwargs)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*self.theme["accent"])
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_y(-11)
        self.set_font(self.f_body, "B", size=10)
        self.set_text_color(*MUTED)
        self.cell(0, 6, clean_markdown(self.footer_text, self.fonts_ok), align="L")
        self.set_x(self.l_margin)
        self.cell(0, 6, f"{self.page_no()}", align="R")


def generate_pdf(summary_text, filepath, title="Video Summary", footer_text=DEFAULT_FOOTER, theme_name="Navy & Gold", fonts_ok=True, script_fonts=()):
    theme = THEMES.get(theme_name, THEMES["Navy & Gold"])
    pdf = PremiumPDF(theme, fonts_ok, footer_text=footer_text, script_fonts=script_fonts)
    pdf.add_page()

    # Cover banner grows with the title (YouTube titles are often 2-3 lines)
    title = clean_markdown(title, fonts_ok)
    size, line_h = (26, 11) if len(title) <= 36 else (19, 9) if len(title) <= 80 else (16, 7.5)
    pdf.set_font(pdf.f_black, size=size)
    title_lines = len(pdf.multi_cell(0, line_h, title, dry_run=True, output="LINES"))
    banner_h = max(52, 17 + title_lines * line_h + 8 + 9)

    pdf.set_fill_color(*theme["primary"])
    pdf.rect(0, 0, pdf.w, banner_h, style="F")
    pdf.set_fill_color(*theme["accent"])
    pdf.rect(0, banner_h, pdf.w, 1.6, style="F")

    pdf.set_xy(pdf.l_margin, 17)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(0, line_h, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin)
    pdf.set_text_color(*theme["accent"])
    pdf.set_font(pdf.f_light, size=10.5)
    pdf.cell(0, 8, datetime.now().strftime("%d %B %Y").upper() + "   |   KEY NOTES", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(banner_h + 14)

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
                client = make_client()
                # Canonical URL: Gemini and yt-dlp both accept it, regardless of the pasted format
                canonical_url = f"https://www.youtube.com/watch?v={video_id}"
                summary = ""
                video_title = fetch_video_title(video_id)

                # Reuse a transcript from an earlier attempt on this video (e.g. after Gemini was busy)
                cached = st.session_state.get("transcript_cache")
                transcript = cached[1] if cached and cached[0] == video_id else ""

                # --- APPROACH 1: Fast transcript fetch ---
                if not transcript:
                    try:
                        with st.spinner("Attempting to fetch transcript directly from YouTube servers..."):
                            transcript = fetch_transcript(video_id)
                        st.session_state["transcript_cache"] = (video_id, transcript)
                        st.toast("⚡ Fast transcript fetched successfully!")
                    except Exception as e:
                        st.warning(f"Direct transcript unavailable ({type(e).__name__}). Trying Gemini video analysis...")

                if transcript.strip():
                    with st.spinner("Summarizing transcript with Gemini..."):
                        summary = summarize_transcript(client, transcript)

                # --- APPROACH 2: Gemini watches the video itself (not affected by YouTube IP blocks) ---
                if not summary:
                    try:
                        with st.spinner("Gemini is watching the video (this may take a minute)..."):
                            summary = summarize_youtube_url(client, canonical_url)
                        st.toast("🎬 Summary generated directly from the video!")
                    except GeminiBusyError:
                        raise  # downloading the audio won't help: its summary needs Gemini too
                    except Exception as e:
                        st.warning(f"Gemini video analysis failed ({type(e).__name__}: {e}). Falling back to audio download...")

                # --- APPROACH 3: Audio download + local Whisper transcription ---
                if not summary:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        with st.spinner("Downloading audio..."):
                            audio_path, ytdlp_title = download_audio(canonical_url, temp_dir)
                            video_title = video_title or ytdlp_title

                        with st.spinner("Transcribing audio locally with AI (this may take a minute)..."):
                            model = load_whisper_model()
                            segments, _ = model.transcribe(audio_path, beam_size=5)
                            transcript = " ".join(segment.text for segment in segments)
                    st.session_state["transcript_cache"] = (video_id, transcript)

                    with st.spinner("Summarizing transcript with Gemini..."):
                        summary = summarize_transcript(client, transcript)

                # Seed the editable text box with the new summary
                st.session_state["result"] = {"summary": summary}
                st.session_state["edited_summary"] = summary
                st.session_state["pdf_title"] = custom_title.strip() or video_title or "Video Summary"
                st.session_state["pdf_footer"] = default_footer

            except Exception:
                st.error("A background component crashed. See the exact details below:")
                st.code(traceback.format_exc())

# --- Edit, Preview & Download ---
if "result" in st.session_state:
    st.success("Summary generated! Edit it below, then download your PDF.")

    if "pdf_footer" not in st.session_state:
        st.session_state["pdf_footer"] = default_footer

    pdf_title = st.text_input("PDF title (also used as the file name):", key="pdf_title")
    pdf_footer = st.text_input("PDF footer:", key="pdf_footer")
    st.caption('Tip: start a line with "- " for a bullet. A short line without "- " becomes a section heading.')
    edited = st.text_area("Edit your summary:", key="edited_summary", height=450)

    fonts_ok = ensure_fonts()
    if not fonts_ok:
        st.warning("Could not download the premium font (offline?). Using the basic font instead.")

    if edited.strip():
        try:
            pdf_title = pdf_title or "Video Summary"
            script_fonts = []
            if fonts_ok:
                with st.spinner("Loading fonts for this language..."):
                    for font in needed_script_fonts(edited + pdf_title + pdf_footer):
                        path = ensure_script_font(font)
                        if path:
                            script_fonts.append(path)
                        else:
                            st.warning(f"Could not download the font for some characters ({font.split('/')[0]}). They may be missing from the PDF.")

            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = os.path.join(tmp, "summary.pdf")
                generate_pdf(edited, pdf_path, title=pdf_title, footer_text=pdf_footer,
                             theme_name=theme_name, fonts_ok=fonts_ok, script_fonts=script_fonts)
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

            st.download_button(
                label="📄 Download PDF Summary",
                data=pdf_bytes,
                file_name=safe_filename(pdf_title),
                mime="application/pdf",
            )

            with st.expander("👁 Preview PDF", expanded=True):
                render_pdf_preview(pdf_bytes)

        except Exception:
            st.error("Could not build the PDF. Details below:")
            st.code(traceback.format_exc())
    else:
        st.info("The summary is empty - add some text to create a PDF.")
