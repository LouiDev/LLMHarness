"""
LlmHarness - a local chat harness for Ollama models.

Features: model selection, per-chat system prompts, thinking mode, loop guard,
sampling controls (max tokens, temperature, ...), saved chats, optional web search.

Run:  python server.py   (or: uvicorn server:app --reload)
Open: http://127.0.0.1:8766
"""
from __future__ import annotations

import asyncio
import collections
import html
import io
import zipfile
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("uvicorn.error")   # shows up in the server console next to uvicorn's own lines

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
PORT = int(os.environ.get("LLMHARNESS_PORT", "8766"))
ROOT = Path(__file__).resolve().parent
CHATS_DIR = Path(os.environ.get("LLMHARNESS_CHATS", ROOT / "chats"))
CHATS_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = ROOT / "settings.json"

LOOP_MIN_UNIT_CHARS = 20          # shorter sentences/paragraphs are ignored by the repeat counter
LOOP_CHECK_EVERY_CHARS = 40       # how often the tail-repeat check runs
SEARCH_PAGE_CHARS = 2500          # extracted text kept per fetched page
SEARCH_SNIPPET_ONLY_CHARS = 600
TITLE_MAX_CHARS = 60
MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # per file
MAX_EXTRACT_CHARS = 400_000           # extracted text kept per file (the context budget trims further)
ATTACHMENT_RESERVE_TOKENS = 1500      # context kept free for system prompt, chat text and the reply
CHARS_PER_TOKEN = 3.2                 # conservative estimate for mixed prose/code

DEFAULT_SETTINGS: dict[str, Any] = {
    "search_provider": "ddgs",        # "ddgs" (DuckDuckGo, no key) or "searxng"
    "searxng_url": "",                # e.g. http://localhost:8080
    "search_region": "wt-wt",
    "keep_alive": "5m",
    "helper_model": "",               # optional small model for search planning and titles ("" = the chat model)
    "chat_defaults": {},              # settings new chats start with ("Use as defaults" in the controls panel)
    "system_prompt_presets": [
        {"name": "Helpful assistant",
         "prompt": "You are a helpful, knowledgeable assistant. Answer directly and accurately. "
                   "Use Markdown formatting when it aids readability."},
        {"name": "Concise",
         "prompt": "You are a precise assistant. Answer in as few words as the question allows. "
                   "No preamble, no summaries, no filler."},
        {"name": "Programmer",
         "prompt": "You are an expert software engineer. Give correct, idiomatic code with brief explanations. "
                   "Always put code in fenced blocks with the language tag. Point out pitfalls and edge cases."},
        {"name": "Writer",
         "prompt": "You are a skilled writer and editor. Write vivid, natural prose. Match the tone the user "
                   "asks for and avoid clichés."},
    ],
}


# ----------------------------------------------------------------------------- helpers

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_settings() -> dict[str, Any]:
    data = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            data.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return data


def save_settings(data: dict[str, Any]) -> None:
    SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def chat_path(chat_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", chat_id):
        raise HTTPException(400, "invalid chat id")
    return CHATS_DIR / f"{chat_id}.json"


def load_chat(chat_id: str) -> dict[str, Any]:
    p = chat_path(chat_id)
    if not p.exists():
        raise HTTPException(404, "chat not found")
    return json.loads(p.read_text(encoding="utf-8"))


def save_chat(chat: dict[str, Any]) -> None:
    chat["updated_at"] = now_iso()
    p = chat_path(chat["id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(chat, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def new_chat(title: str = "", settings: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "settings": settings or {},
        "messages": [],
    }


def chat_summary(chat: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": chat["id"],
        "title": chat.get("title") or "",
        "created_at": chat.get("created_at"),
        "updated_at": chat.get("updated_at"),
        "model": (chat.get("settings") or {}).get("model", ""),
        "message_count": len(chat.get("messages", [])),
        "pinned": bool(chat.get("pinned")),
    }


def sse(event: dict[str, Any]) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


# ----------------------------------------------------------------------------- loop guard

class LoopGuard:
    """Detects a model that is stuck repeating itself.

    Two signals: (1) a sentence/paragraph of at least LOOP_MIN_UNIT_CHARS that has been
    produced `threshold` times in this reply, (2) the tail of the text being the same
    block repeated 4+ times back-to-back (catches short loops like "and and and ...").
    """

    def __init__(self, threshold: int):
        self.threshold = max(2, int(threshold))
        self.text = ""
        self.counts: collections.Counter[str] = collections.Counter()
        self.consumed = 0
        self.last_tail_check = 0

    def feed(self, delta: str) -> str | None:
        self.text += delta
        # Count completed units (paragraphs and sentences).
        while True:
            m = re.search(r"(\n\s*\n|[.!?]\s+\n?)", self.text[self.consumed:])
            if not m:
                break
            end = self.consumed + m.end()
            unit = re.sub(r"\s+", " ", self.text[self.consumed:end]).strip().lower()
            self.consumed = end
            if len(unit) >= LOOP_MIN_UNIT_CHARS:
                self.counts[unit] += 1
                if self.counts[unit] >= self.threshold:
                    return f"repeated the same passage {self.counts[unit]} times"
        # Tail repetition check.
        if len(self.text) - self.last_tail_check >= LOOP_CHECK_EVERY_CHARS:
            self.last_tail_check = len(self.text)
            tail = self.text[-600:]
            for period in range(4, 150):
                if period * 4 > len(tail):
                    break
                block = tail[-period:]
                if block.strip() and tail.endswith(block * 4):
                    return "repeating the same fragment"
        return None


# ----------------------------------------------------------------------------- <think> tag splitter

class ThinkSplitter:
    """Splits a token stream that embeds <think>...</think> into (kind, text) parts.

    Only a <think> tag at the very start of the response opens a thinking block. Once
    real content has been emitted, a later "<think>" is just prose (e.g. a model describing
    thinking tags) and is passed through untouched. With ``enabled=False`` everything is
    content; use that for models whose thinking Ollama already delivers separately.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, enabled: bool = True) -> None:
        self.buf = ""
        self.in_think = False
        self.enabled = enabled
        self.started = False      # True once any content has been emitted (tags no longer honoured)

    def feed(self, delta: str) -> list[tuple[str, str]]:
        if not self.enabled:
            return [("content", delta)] if delta else []
        self.buf += delta
        out: list[tuple[str, str]] = []
        while self.buf:
            if self.started and not self.in_think:
                out.append(("content", self.buf))
                self.buf = ""
                break
            tag = self.CLOSE if self.in_think else self.OPEN
            if self.in_think:
                idx = self.buf.find(tag)
            else:
                # Opening tag only counts if it is the first non-whitespace thing in the response.
                stripped = self.buf.lstrip()
                idx = len(self.buf) - len(stripped) if stripped.startswith(tag) else -1
                if idx < 0:
                    if not stripped or tag.startswith(stripped):
                        break                   # only whitespace / a partial tag so far: wait for more
                    # Response begins with something other than a <think> tag: plain content from here on.
                    self.started = True
                    continue
            if idx >= 0:
                if idx and self.in_think:      # whitespace before an opening tag is dropped
                    out.append(("thinking", self.buf[:idx]))
                self.buf = self.buf[idx + len(tag):]
                self.in_think = not self.in_think
                if not self.in_think:
                    self.started = True
                continue
            # Keep a possible partial tag at the end of the buffer.
            keep = 0
            for k in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                if tag.startswith(self.buf[-k:]):
                    keep = k
                    break
            emit, self.buf = (self.buf[:-keep], self.buf[-keep:]) if keep else (self.buf, "")
            if emit:
                out.append(("thinking" if self.in_think else "content", emit))
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        out = [("thinking" if self.in_think else "content", self.buf)] if self.buf else []
        self.buf = ""
        return out


# ----------------------------------------------------------------------------- ollama

_model_caps: dict[str, list[str]] = {}


async def show_caps(client: httpx.AsyncClient, model: str) -> list[str]:
    """/api/tags under-reports capabilities for some imported models; /api/show is authoritative."""
    try:
        r = await client.post(f"{OLLAMA_HOST}/api/show", json={"model": model}, timeout=15)
        r.raise_for_status()
        caps = r.json().get("capabilities") or []
    except httpx.HTTPError:
        caps = _model_caps.get(model, [])
    _model_caps[model] = caps
    return caps


async def ollama_models(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
    r.raise_for_status()
    raw = r.json().get("models", [])
    all_caps = await asyncio.gather(*(show_caps(client, m["name"]) for m in raw))
    models = []
    for m, caps in zip(raw, all_caps):
        d = m.get("details") or {}
        models.append({
            "name": m["name"],
            "size": m.get("size", 0),
            "parameter_size": d.get("parameter_size", ""),
            "quantization": d.get("quantization_level", ""),
            "family": d.get("family", ""),
            "context_length": d.get("context_length"),
            "capabilities": caps,
            "modified_at": m.get("modified_at"),
        })
    models.sort(key=lambda x: x["name"].lower())
    return models


async def model_caps(client: httpx.AsyncClient, model: str) -> list[str]:
    if model not in _model_caps:
        await show_caps(client, model)
    return _model_caps.get(model, [])


def clean_options(opts: dict[str, Any] | None) -> dict[str, Any]:
    """Drop empty values so Ollama falls back to the model's own defaults."""
    out: dict[str, Any] = {}
    for k, v in (opts or {}).items():
        if v is None or v == "":
            continue
        if k in ("num_predict", "num_ctx", "top_k", "repeat_last_n", "seed"):
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                continue
        elif k in ("temperature", "top_p", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty"):
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
        elif k == "stop" and isinstance(v, list) and v:
            out[k] = [str(s) for s in v if s]
    return out


def think_param(mode: str, caps: list[str]) -> bool | str | None:
    """Translate the UI think mode into Ollama's `think` field (None = don't send)."""
    if "thinking" not in caps:
        return None
    mode = (mode or "default").lower()
    if mode in ("default", ""):
        return None
    if mode == "off":
        return False
    if mode == "on":
        return True
    if mode in ("low", "medium", "high"):
        return mode
    return None


async def ollama_chat_once(client: httpx.AsyncClient, payload: dict[str, Any], timeout: float = 60) -> str:
    payload = {**payload, "stream": False}
    r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "")


# ----------------------------------------------------------------------------- web search

_TAG_RE = re.compile(r"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", re.S | re.I)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n\s*\n+")


def html_to_text(raw: str) -> str:
    raw = _TAG_RE.sub(" ", raw)
    raw = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</h\d>|</tr>", "\n", raw, flags=re.I)
    text = html.unescape(_HTML_RE.sub(" ", raw))
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


async def search_ddgs(query: str, max_results: int, region: str) -> list[dict[str, str]]:
    def run() -> list[dict[str, str]]:
        from ddgs import DDGS  # imported lazily so the app still starts without it
        with DDGS() as ddgs:
            rows = ddgs.text(query, region=region or "wt-wt", safesearch="moderate", max_results=max_results)
        return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in rows or [] if r.get("href")]
    return await asyncio.to_thread(run)


async def search_searxng(client: httpx.AsyncClient, base: str, query: str, max_results: int) -> list[dict[str, str]]:
    r = await client.get(f"{base.rstrip('/')}/search", params={"q": query, "format": "json"}, timeout=15)
    r.raise_for_status()
    rows = r.json().get("results", [])[:max_results]
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")} for x in rows]


async def fetch_page_text(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, timeout=8, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0 (LlmHarness local research)"})
        ctype = r.headers.get("content-type", "")
        if r.status_code != 200 or not ("html" in ctype or "text/plain" in ctype):
            return ""
        return html_to_text(r.text[:600_000])[:SEARCH_PAGE_CHARS]
    except (httpx.HTTPError, UnicodeDecodeError):
        return ""


async def web_search(client: httpx.AsyncClient, settings: dict[str, Any], query: str,
                     max_results: int, fetch_pages: bool) -> list[dict[str, str]]:
    provider = settings.get("search_provider", "ddgs")
    if provider == "searxng" and settings.get("searxng_url"):
        results = await search_searxng(client, settings["searxng_url"], query, max_results)
    else:
        results = await search_ddgs(query, max_results, settings.get("search_region", "wt-wt"))
    results = results[:max_results]
    if fetch_pages and results:
        texts = await asyncio.gather(*(fetch_page_text(client, r["url"]) for r in results))
        for r, t in zip(results, texts):
            r["text"] = t
    return results


def build_search_context(query: str, results: list[dict[str, str]]) -> str:
    lines = [f'Web search results for "{query}" (retrieved {datetime.now().strftime("%Y-%m-%d")}):', ""]
    for i, r in enumerate(results, 1):
        body = r.get("text") or r.get("snippet", "")
        body = body[:SEARCH_PAGE_CHARS] if r.get("text") else body[:SEARCH_SNIPPET_ONLY_CHARS]
        lines.append(f"[{i}] {r['title']}\nURL: {r['url']}\n{body}\n")
    lines.append("Use these results where relevant and cite them inline as [1], [2], ... "
                 "If the results do not answer the question, say so instead of guessing.")
    return "\n".join(lines)


SEARCH_PLANNER_PROMPT = (
    "You are a search planner for an assistant. Given a conversation, decide whether looking "
    "up current information on the web would materially help answer the LATEST user message. "
    "Search for facts, news, prices, versions, documentation, people, places, events, or anything "
    "you might not know reliably. Do not search for greetings, opinions, creative writing, math, "
    "or questions fully answerable from the conversation. "
    'Reply with JSON only: {"search": true or false, "query": "a concise web search query"}'
)


EXPLICIT_SEARCH_RE = re.compile(
    r"\b(search|look(?: it| this| that)? up|google|browse|check|find|research)\b[^.?!\n]{0,40}?"
    r"\b(web|internet|online|news|net)\b"
    r"|\b(web search|search the web|search online|latest news|current news|news today|breaking news)\b"
    r"|\bgoogle\b",
    re.I,
)
SEARCH_VERB_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?(?:search|look up|google|browse|check|find|research)\b"
    r"[^,:.]{0,30}?\b(?:web|internet|online|news)\b[\s,:]*(?:and|then|to|for)?\s*", re.I)


def explicit_search_request(text: str) -> bool:
    return bool(EXPLICIT_SEARCH_RE.search(text or ""))


def fallback_query(text: str) -> str:
    """Turn 'Search the web and list latest BBC news' into 'list latest BBC news'."""
    q = SEARCH_VERB_RE.sub("", text or "", count=1).strip()
    return (q or text or "").strip()[:200]


def extract_json(raw: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a reply that may contain fences, <think> blocks or chatter."""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    for m in re.finditer(r"\{.*?\}", raw, flags=re.S):
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


async def helper_model_for(client: httpx.AsyncClient, settings: dict[str, Any], chat_model: str,
                           chat_caps: list[str]) -> tuple[str, list[str]]:
    helper = (settings.get("helper_model") or "").strip()
    if helper and helper != chat_model:
        return helper, await model_caps(client, helper)
    return chat_model, chat_caps


def helper_extras(helper: str, chat_model: str, options: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Payload fields that keep a helper call from reloading the chat model.

    Ollama reloads a model whenever num_ctx differs from the loaded runner, so when the helper
    *is* the chat model we send the same num_ctx (and keep_alive) the chat uses.
    """
    extras: dict[str, Any] = {"keep_alive": settings.get("keep_alive", "5m")}
    if helper == chat_model and options.get("num_ctx"):
        extras["options"] = {"num_ctx": options["num_ctx"]}
    return extras


async def plan_search(client: httpx.AsyncClient, model: str, caps: list[str],
                      history: list[dict[str, Any]], force: bool,
                      extras: dict[str, Any] | None = None) -> tuple[bool, str, str]:
    """Returns (should_search, query, note). Never raises."""
    latest = history[-1]["content"] if history else ""
    if not force and explicit_search_request(latest):
        force = True
    recent = history[-6:-1]
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in recent)
    instruction = SEARCH_PLANNER_PROMPT
    if force:
        instruction += " A search IS required for this message; just produce the best query."
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Conversation so far:\n{convo}\n\nLatest user message:\n{latest}"},
        ],
        "format": "json",
        "options": {"temperature": 0, "num_predict": 160, **(extras or {}).get("options", {})},
        **{k: v for k, v in (extras or {}).items() if k != "options"},
    }
    tp = think_param("off", caps)
    if tp is not None:
        payload["think"] = tp
    try:
        raw = await ollama_chat_once(client, payload, timeout=120)
    except httpx.HTTPError as e:
        log.warning("search planner call to %s failed: %s: %s", model, e.__class__.__name__, e)
        return force, fallback_query(latest), f"planner call failed ({e.__class__.__name__})"
    data = extract_json(raw)
    if data is None:
        return force, fallback_query(latest), "planner returned no usable JSON"
    query = str(data.get("query") or "").strip() or fallback_query(latest)
    return bool(data.get("search")) or force, query, ""


# ----------------------------------------------------------------------------- file attachments

TEXT_EXTENSIONS = {
    "txt", "md", "markdown", "rst", "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml", "html", "htm",
    "log", "ini", "toml", "cfg", "conf", "env", "tex", "srt", "vtt",
    "py", "js", "mjs", "ts", "tsx", "jsx", "java", "kt", "cs", "cpp", "cc", "c", "h", "hpp", "go", "rs", "rb",
    "php", "swift", "sh", "bash", "zsh", "ps1", "psm1", "bat", "cmd", "sql", "r", "lua", "pl", "dart", "scala",
    "gradle", "cmake", "make", "dockerfile", "gitignore", "editorconfig", "vue", "svelte", "css", "scss", "less",
}


def _decode_text(data: bytes) -> str | None:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = data.decode(enc)
            if "\x00" in text:
                return None
            return text
        except UnicodeDecodeError:
            continue
    try:
        text = data.decode("cp1252")
        return None if "\x00" in text else text
    except UnicodeDecodeError:
        return None


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader  # lazy import keeps startup fast
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"--- Page {i} ---\n{text}")
        if sum(len(x) for x in pages) > MAX_EXTRACT_CHARS:
            break
    return "\n\n".join(pages)


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab/>", "\t", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml)).strip()


def extract_text(filename: str, mime: str, data: bytes) -> dict[str, Any]:
    """Turn an uploaded file into text the model can read. Never raises."""
    name = filename or "file"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else name.lower()
    mime = (mime or "").lower()
    info: dict[str, Any] = {"name": name, "type": mime or ext, "size": len(data), "kind": "text", "text": "",
                            "chars": 0, "truncated": False}
    if len(data) > MAX_UPLOAD_BYTES:
        return {**info, "kind": "unsupported", "error": f"larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}
    if mime.startswith("image/"):
        return {**info, "kind": "image"}
    try:
        if ext == "pdf" or mime == "application/pdf":
            text = _pdf_text(data)
            if not text.strip():
                return {**info, "kind": "unsupported", "error": "no selectable text in this PDF (scanned pages need a vision model)"}
        elif ext == "docx" or mime.endswith("wordprocessingml.document"):
            text = _docx_text(data)
        elif ext in TEXT_EXTENSIONS or mime.startswith("text/") or mime in ("application/json", "application/xml", "application/javascript"):
            decoded = _decode_text(data)
            if decoded is None:
                return {**info, "kind": "unsupported", "error": "not a text file"}
            text = decoded
        else:
            decoded = _decode_text(data)
            if decoded is None:
                return {**info, "kind": "unsupported", "error": f".{ext} files are not supported; use text, code, CSV, JSON, PDF or DOCX"}
            text = decoded
    except Exception as e:  # corrupt archives, encrypted PDFs, ...
        return {**info, "kind": "unsupported", "error": f"could not read the file ({e.__class__.__name__})"}
    text = text.replace("\r\n", "\n")
    truncated = len(text) > MAX_EXTRACT_CHARS
    return {**info, "text": text[:MAX_EXTRACT_CHARS], "chars": len(text), "truncated": truncated}


def attachment_budget_chars(options: dict[str, Any]) -> int:
    num_ctx = int(options.get("num_ctx") or 4096)
    return max(1500, int((num_ctx - ATTACHMENT_RESERVE_TOKENS) * CHARS_PER_TOKEN))


def format_attachment(att: dict[str, Any], limit: int) -> str:
    name = att.get("name") or "file"
    text = att.get("text") or ""
    total = att.get("chars") or len(text)
    if limit <= 0 or not text:
        return f"[Attached file: {name}, {total} characters, omitted because the context window is full]"
    shown = text[:limit]
    note = f", showing the first {len(shown)} of {total} characters" if len(shown) < total else f", {total} characters"
    return f"[Attached file: {name}{note}]\n{shown}\n[End of {name}]"


def expand_attachments(outgoing: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
    """Inline attached-file text into the user messages, newest first, within the context budget."""
    remaining = attachment_budget_chars(options)
    used = 0
    truncated = False
    for m in reversed(outgoing):
        atts = m.pop("attachments", None) or []
        if not atts:
            continue
        blocks = []
        for a in atts:
            text_len = len(a.get("text") or "")
            take = min(text_len, remaining)
            if take < text_len:
                truncated = True
            blocks.append(format_attachment(a, take))
            remaining -= take
            used += take
        m["content"] = "\n\n".join(blocks + [m.get("content") or ""]).strip()
    return {"chars": used, "tokens": int(used / CHARS_PER_TOKEN), "truncated": truncated}


# ----------------------------------------------------------------------------- generation

_cancel_events: dict[str, asyncio.Event] = {}


async def generate_stream(req: dict[str, Any], request: Request) -> AsyncIterator[str]:
    gen_id = str(req.get("gen_id") or uuid.uuid4().hex)
    cancel = _cancel_events.setdefault(gen_id, asyncio.Event())
    settings = load_settings()

    model = req.get("model") or ""
    system_prompt = (req.get("system_prompt") or "").strip()
    history: list[dict[str, Any]] = [
        {k: v for k, v in m.items() if k in ("role", "content", "images", "attachments")}
        for m in req.get("messages", [])
        if m.get("role") in ("user", "assistant") and (m.get("content") or m.get("images") or m.get("attachments"))
    ]
    options = clean_options(req.get("options"))
    think_mode = req.get("think", "default")
    loop_cfg = req.get("loop_guard") or {}
    search_cfg = req.get("web_search") or {}
    timeout_s = float(req.get("timeout_s") or 0)
    chat_id = req.get("chat_id")
    save = bool(req.get("save", True)) and bool(chat_id)

    if not model:
        yield sse({"type": "error", "message": "Pick a model first."})
        return
    if not history or history[-1]["role"] != "user":
        yield sse({"type": "error", "message": "The last message must come from the user."})
        return

    assistant: dict[str, Any] = {
        "role": "assistant", "content": "", "thinking": "", "model": model,
        "created_at": now_iso(),
    }
    aborted: str | None = None
    stats: dict[str, Any] = {}
    started = time.monotonic()
    first_token_at: float | None = None
    token_count = 0

    async with httpx.AsyncClient() as client:
        caps = await model_caps(client, model)
        if "vision" not in caps and any(m.get("images") for m in history):
            yield sse({"type": "error", "message": f"{model} cannot see images. Remove them or switch to a vision model."})
            return
        outgoing = [dict(m) for m in history]
        att_info = expand_attachments(outgoing, options)
        if att_info["chars"]:
            detail = f"Attached files use about {att_info['tokens']:,} tokens"
            if att_info["truncated"]:
                detail += "; some content was cut to fit the context length"
            yield sse({"type": "status", "phase": "attachments", "detail": detail})

        # --- optional web search ------------------------------------------------------
        mode = (search_cfg.get("mode") or "off").lower()
        if mode in ("auto", "always"):
            yield sse({"type": "status", "phase": "planning", "detail": "Deciding whether to search"})
            helper, helper_caps = await helper_model_for(client, settings, model, caps)
            want, query, note = await plan_search(client, helper, helper_caps, history, force=(mode == "always"),
                                                  extras=helper_extras(helper, model, options, settings))
            if note:
                yield sse({"type": "status", "phase": "planning", "detail": note})
            if cancel.is_set():
                yield sse({"type": "done", "aborted": "stopped", "stats": {}})
                return
            if want and query:
                yield sse({"type": "status", "phase": "searching", "detail": f"Searching: {query}"})
                try:
                    results = await web_search(client, settings, query,
                                               int(search_cfg.get("max_results") or 5),
                                               bool(search_cfg.get("fetch_pages", True)))
                except Exception as e:  # network / provider failures should not kill the reply
                    results = []
                    yield sse({"type": "status", "phase": "searching", "detail": f"Search failed: {e}"})
                if results:
                    sources = [{"n": i, "title": r["title"], "url": r["url"], "snippet": r.get("snippet", "")}
                               for i, r in enumerate(results, 1)]
                    assistant["sources"] = sources
                    assistant["search_query"] = query
                    yield sse({"type": "sources", "query": query, "items": sources})
                    outgoing[-1] = {**outgoing[-1],
                                    "content": build_search_context(query, results) + "\n\nUser's message:\n" + outgoing[-1]["content"]}
            else:
                yield sse({"type": "status", "phase": "planning", "detail": "No search needed"})

        # --- main streamed completion ----------------------------------------------------
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + outgoing
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True,
                                   "options": options, "keep_alive": settings.get("keep_alive", "5m")}
        tp = think_param(think_mode, caps)
        if tp is not None:
            payload["think"] = tp

        guard_on = loop_cfg.get("enabled", True)
        guard = LoopGuard(loop_cfg.get("threshold", 3)) if guard_on else None
        think_guard = LoopGuard(loop_cfg.get("threshold", 3)) if guard_on else None
        think_budget = int(req.get("think_budget") or 0)
        think_tokens = 0
        # Models with native thinking stream it in msg["thinking"]; a "<think>" in their content is prose.
        splitter = ThinkSplitter(enabled="thinking" not in caps)
        yield sse({"type": "status", "phase": "generating", "detail": "Waiting for the model"})

        def take(kind: str, text: str) -> tuple[str | None, list[str]]:
            """Append a piece to the assistant message; returns (abort_reason, events)."""
            events: list[str] = []
            if not text:
                return None, events
            if kind == "thinking":
                assistant["thinking"] += text
                events.append(sse({"type": "thinking", "delta": text}))
                if think_guard:
                    reason = think_guard.feed(text)
                    if reason:
                        return f"loop guard (thinking): {reason}", events
                if think_budget and think_tokens > think_budget:
                    return f"thinking exceeded the {think_budget}-token budget", events
                return None, events
            assistant["content"] += text
            events.append(sse({"type": "content", "delta": text}))
            if guard:
                reason = guard.feed(text)
                if reason:
                    return reason, events
            return None, events

        try:
            async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload, timeout=httpx.Timeout(None, connect=15)) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    try:
                        body = json.loads(body).get("error", body)
                    except json.JSONDecodeError:
                        pass
                    yield sse({"type": "error", "message": f"Ollama returned {resp.status_code}: {body}"})
                    return
                async for line in resp.aiter_lines():
                    if cancel.is_set():
                        aborted = "stopped"
                        break
                    if timeout_s and time.monotonic() - started > timeout_s:
                        aborted = f"timed out after {int(timeout_s)} s"
                        break
                    if await request.is_disconnected():
                        aborted = "client disconnected"
                        break
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        yield sse({"type": "error", "message": chunk["error"]})
                        aborted = "error"
                        break
                    msg = chunk.get("message") or {}
                    if first_token_at is None and (msg.get("thinking") or msg.get("content")):
                        first_token_at = time.monotonic()
                    token_count += 1
                    if msg.get("thinking"):
                        think_tokens += 1
                    reason = None
                    for kind, text in ([("thinking", msg.get("thinking") or "")] +
                                       [p for p in splitter.feed(msg.get("content") or "")]):
                        r, events = take(kind, text)
                        for e in events:
                            yield e
                        reason = reason or r
                    if reason:
                        aborted = reason if reason.startswith(("loop guard", "thinking exceeded")) else f"loop guard: {reason}"
                        break
                    if chunk.get("done"):
                        for kind, text in splitter.flush():
                            _, events = take(kind, text)
                            for e in events:
                                yield e
                        ev = chunk.get("eval_duration") or 0
                        stats = {
                            "prompt_tokens": chunk.get("prompt_eval_count"),
                            "tokens": chunk.get("eval_count"),
                            "eval_seconds": round(ev / 1e9, 2),
                            "load_seconds": round((chunk.get("load_duration") or 0) / 1e9, 2),
                            "total_seconds": round((chunk.get("total_duration") or 0) / 1e9, 2),
                            "tokens_per_second": round(chunk.get("eval_count", 0) / (ev / 1e9), 1) if ev else None,
                            "done_reason": chunk.get("done_reason"),
                        }
                        break
        except httpx.HTTPError as e:
            yield sse({"type": "error", "message": f"Could not reach Ollama at {OLLAMA_HOST}: {e}"})
            aborted = "error"
        finally:
            _cancel_events.pop(gen_id, None)

        if aborted:
            for kind, text in splitter.flush():
                _, events = take(kind, text)
                for e in events:
                    yield e
            elapsed = time.monotonic() - (first_token_at or started)
            stats = {
                "tokens": token_count,
                "total_seconds": round(time.monotonic() - started, 2),
                "tokens_per_second": round(token_count / elapsed, 1) if elapsed > 0 and token_count else None,
                "done_reason": aborted,
            }
            assistant["aborted"] = aborted
        assistant["stats"] = stats
        assistant["think_mode"] = think_mode
        yield sse({"type": "done", "stats": stats, "aborted": aborted})

        # --- persist -------------------------------------------------------------------
        chat: dict[str, Any] | None = None
        if save and (assistant["content"] or assistant["thinking"]):
            try:
                chat = load_chat(chat_id)
            except HTTPException:
                chat = new_chat()
                chat["id"] = chat_id
            chat["settings"] = req.get("chat_settings") or chat.get("settings") or {}
            chat["messages"] = [
                {**m, "created_at": m.get("created_at") or now_iso()}
                for m in req.get("messages", [])
            ] + [assistant]
            save_chat(chat)
            yield sse({"type": "saved", "chat_id": chat["id"], "message": assistant})

        # --- auto title ----------------------------------------------------------------
        if chat is not None and not chat.get("title") and assistant["content"] and aborted not in ("error",):
            # Runs as its own task: if the browser drops the SSE connection (which cancels this
            # generator), the title is still generated and saved; the UI picks it up on refresh.
            task = asyncio.create_task(auto_title(chat["id"], settings, model, caps, options,
                                                  history[-1]["content"], assistant["content"]))
            try:
                title = await asyncio.shield(task)
            except asyncio.CancelledError:
                log.info("chat %s: client went away before the title was ready; finishing it in the background",
                         chat["id"])
                raise
            if title:
                yield sse({"type": "title", "chat_id": chat["id"], "title": title})


async def auto_title(chat_id: str, settings: dict[str, Any], model: str, caps: list[str],
                     options: dict[str, Any], user: str, answer: str) -> str:
    """Generate and persist a title for a chat that has none. Never raises; returns the saved title or ""."""
    try:
        async with httpx.AsyncClient() as client:
            helper, helper_caps = await helper_model_for(client, settings, model, caps)
            title = await make_title(client, helper, helper_caps, user, answer,
                                     extras=helper_extras(helper, model, options, settings))
        if not title:
            log.warning("chat %s: no title produced", chat_id)
            return ""
        chat = load_chat(chat_id)
        if chat.get("title"):            # the user typed one meanwhile
            return ""
        chat["title"] = title
        save_chat(chat)
        return title
    except Exception:
        log.exception("chat %s: auto-title failed", chat_id)
        return ""


async def make_title(client: httpx.AsyncClient, model: str, caps: list[str], user: str, answer: str,
                     extras: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Write a short title (3 to 6 words) for the conversation. "
                                          "Reply with the title only: no quotes, no trailing period."},
            {"role": "user", "content": f"User: {user[:800]}\n\nAssistant: {answer[:800]}"},
        ],
        "options": {"temperature": 0.2, "num_predict": 24, **(extras or {}).get("options", {})},
        **{k: v for k, v in (extras or {}).items() if k != "options"},
    }
    tp = think_param("off", caps)
    if tp is not None:
        payload["think"] = tp
    try:
        raw = await ollama_chat_once(client, payload, timeout=60)
    except httpx.HTTPError as e:
        log.warning("title request to %s failed: %s: %s", model, e.__class__.__name__, e)
        return ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    title = raw.strip().splitlines()[0].strip().strip('"\'“”').rstrip(".") if raw.strip() else ""
    title = re.sub(r"^(title:)\s*", "", title, flags=re.I)
    return title[:TITLE_MAX_CHARS] or user.strip()[:TITLE_MAX_CHARS]


# ----------------------------------------------------------------------------- app

app = FastAPI(title="LlmHarness")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
async def health() -> JSONResponse:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{OLLAMA_HOST}/api/version", timeout=5)
            r.raise_for_status()
            return JSONResponse({"ok": True, "ollama": r.json().get("version"), "host": OLLAMA_HOST})
        except httpx.HTTPError as e:
            return JSONResponse({"ok": False, "error": str(e), "host": OLLAMA_HOST})


@app.get("/api/models")
async def models() -> JSONResponse:
    async with httpx.AsyncClient() as client:
        try:
            return JSONResponse(await ollama_models(client))
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Could not reach Ollama at {OLLAMA_HOST}: {e}")


@app.get("/api/models/running")
async def running_models() -> JSONResponse:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{OLLAMA_HOST}/api/ps", timeout=5)
            r.raise_for_status()
            return JSONResponse(r.json().get("models", []))
        except httpx.HTTPError as e:
            raise HTTPException(502, str(e))


@app.post("/api/models/unload")
async def unload_model(body: dict[str, Any]) -> JSONResponse:
    model = body.get("model")
    if not model:
        raise HTTPException(400, "model required")
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{OLLAMA_HOST}/api/chat", json={"model": model, "messages": [], "keep_alive": 0}, timeout=30)
        return JSONResponse({"ok": r.status_code == 200})


@app.get("/api/settings")
async def get_settings() -> JSONResponse:
    return JSONResponse(load_settings())


@app.put("/api/settings")
async def put_settings(body: dict[str, Any]) -> JSONResponse:
    data = load_settings()
    data.update({k: v for k, v in body.items() if k in DEFAULT_SETTINGS})
    save_settings(data)
    return JSONResponse(data)


@app.get("/api/chats")
async def list_chats() -> JSONResponse:
    out = []
    for p in CHATS_DIR.glob("*.json"):
        try:
            out.append(chat_summary(json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    out.sort(key=lambda c: c["updated_at"] or "", reverse=True)
    out.sort(key=lambda c: not c["pinned"])  # stable: pinned first, newest first within each group
    return JSONResponse(out)


@app.post("/api/chats")
async def create_chat(body: dict[str, Any] | None = None) -> JSONResponse:
    body = body or {}
    chat = new_chat(title=body.get("title", ""), settings=body.get("settings"))
    save_chat(chat)
    return JSONResponse(chat)


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str) -> JSONResponse:
    return JSONResponse(load_chat(chat_id))


@app.put("/api/chats/{chat_id}")
async def put_chat(chat_id: str, body: dict[str, Any]) -> JSONResponse:
    try:
        chat = load_chat(chat_id)
    except HTTPException:
        chat = new_chat()
        chat["id"] = chat_id
    for key in ("title", "messages", "settings", "pinned"):
        if key in body:
            chat[key] = body[key]
    save_chat(chat)
    return JSONResponse(chat)


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str) -> JSONResponse:
    p = chat_path(chat_id)
    if p.exists():
        p.unlink()
    return JSONResponse({"ok": True})


@app.post("/api/generate")
async def generate(request: Request) -> StreamingResponse:
    req = await request.json()
    return StreamingResponse(generate_stream(req, request), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/generate/{gen_id}/stop")
async def stop_generation(gen_id: str) -> JSONResponse:
    ev = _cancel_events.get(gen_id)
    if ev:
        ev.set()
    return JSONResponse({"ok": bool(ev)})


@app.post("/api/extract")
async def extract_files(files: list[UploadFile] = File(...)) -> JSONResponse:
    """Turn uploaded documents into text for the chat. Images are reported as kind=image and left to the client."""
    out = []
    for f in files:
        data = await f.read()
        out.append(await asyncio.to_thread(extract_text, f.filename or "file", f.content_type or "", data))
    return JSONResponse(out)


@app.post("/api/search")
async def search_endpoint(body: dict[str, Any]) -> JSONResponse:
    """Manual search, handy for testing the provider configuration."""
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query required")
    async with httpx.AsyncClient() as client:
        try:
            results = await web_search(client, load_settings(), query, int(body.get("max_results") or 5),
                                       bool(body.get("fetch_pages", False)))
        except Exception as e:
            raise HTTPException(502, f"search failed: {e}")
    return JSONResponse(results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=PORT, reload=False)
