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
import sys
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
SETTINGS_FILE = Path(os.environ.get("LLMHARNESS_SETTINGS", ROOT / "settings.json"))

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
    "workspace_dir": "",              # folder the agent's file tools may touch ("" = ./workspace)
    "allow_outside_workspace": False, # let file tools use absolute paths anywhere on this computer
    "tool_policies": {},              # per tool: "ask" (Allow / Deny in the chat) or "auto"; missing = the tool's default
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


def build_search_context(query: str, results: list[dict[str, str]], start: int = 1) -> str:
    lines = [f'Web search results for "{query}" (retrieved {datetime.now().strftime("%Y-%m-%d")}):', ""]
    for i, r in enumerate(results, start):
        body = r.get("text") or r.get("snippet", "")
        body = body[:SEARCH_PAGE_CHARS] if r.get("text") else body[:SEARCH_SNIPPET_ONLY_CHARS]
        lines.append(f"[{i}] {r['title']}\nURL: {r['url']}\n{body}\n")
    lines.append(f"Use these results where relevant and cite them inline as [{start}], [{start + 1}], ... "
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


# ----------------------------------------------------------------------------- agent tools

TOOL_RESULT_CHARS = 12_000            # tool output kept per call (sent to the model and stored with the chat)
TOOL_STEPS_DEFAULT = 8                # model turns per reply before a final answer is forced
TOOL_PYTHON_TIMEOUT_S = 30
TOOL_LIST_MAX = 400


def workspace_dir(settings: dict[str, Any]) -> Path:
    raw = (settings.get("workspace_dir") or os.environ.get("LLMHARNESS_WORKSPACE") or "").strip()
    return Path(raw).expanduser() if raw else ROOT / "workspace"


def workspace_path(base: Path, rel: str, unrestricted: bool = False) -> Path:
    """Resolve a model-supplied path. Confined to the workspace unless `unrestricted` (a server setting)."""
    base = base.resolve()
    rel = (rel or ".").strip().replace("\\", "/")
    absolute = rel.startswith("/") or bool(re.match(r"^[a-zA-Z]:", rel))
    if unrestricted:
        return (Path(rel) if absolute else base / rel).expanduser().resolve()
    if absolute:
        raise ValueError(f"'{rel}' is absolute; use a path relative to the workspace")
    p = (base / rel).resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"'{rel}' is outside the workspace")
    return p


class ToolContext:
    """What tool handlers get: the HTTP client, server settings, and per-reply search state."""

    def __init__(self, client: httpx.AsyncClient, settings: dict[str, Any], search_cfg: dict[str, Any],
                 attachments: list[dict[str, Any]] | None = None, chat_id: str | None = None):
        self.client = client
        self.settings = settings
        self.search_cfg = search_cfg
        self.attachments = attachments or []      # files attached anywhere in this chat
        self.chat_id = chat_id
        self.workspace = workspace_dir(settings)
        self.unrestricted = bool(settings.get("allow_outside_workspace"))
        self.sources: list[dict[str, Any]] = []   # numbered across every search in the reply
        self.queries: list[str] = []


async def tool_web_search(ctx: ToolContext, args: dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    n = max(1, min(int(args.get("max_results") or ctx.search_cfg.get("max_results") or 5), 10))
    results = await web_search(ctx.client, ctx.settings, query, n, bool(ctx.search_cfg.get("fetch_pages", True)))
    if not results:
        return f'No results for "{query}".'
    start = len(ctx.sources) + 1
    ctx.sources += [{"n": start + i, "title": r["title"], "url": r["url"], "snippet": r.get("snippet", "")}
                    for i, r in enumerate(results)]
    ctx.queries.append(query)
    return build_search_context(query, results, start=start)


async def tool_fetch_page(ctx: ToolContext, args: dict[str, Any]) -> str:
    url = str(args.get("url") or "").strip()
    if not re.match(r"^https?://", url):
        raise ValueError("url must start with http:// or https://")
    r = await ctx.client.get(url, timeout=15, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0 (LlmHarness local research)"})
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code}")
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
        text = await asyncio.to_thread(_pdf_text, r.content[:40_000_000])
    elif "html" in ctype:
        text = html_to_text(r.text[:1_500_000])
    else:
        text = r.text
    return text[:TOOL_RESULT_CHARS] or "(the page has no readable text)"


def _rel(base: Path, p: Path) -> str:
    try:
        return p.relative_to(base.resolve()).as_posix() or "."
    except ValueError:
        return str(p)


async def tool_list_files(ctx: ToolContext, args: dict[str, Any]) -> str:
    rel = str(args.get("path") or ".")
    p = workspace_path(ctx.workspace, rel, ctx.unrestricted)
    if not p.exists():
        if p == ctx.workspace.resolve():
            return "The workspace is empty (the folder does not exist yet; writing a file creates it)."
        raise ValueError(f"'{rel}' does not exist")
    if p.is_file():
        return f"{_rel(ctx.workspace, p)}  ({p.stat().st_size:,} bytes)"
    rows = []
    for child in sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
        if child.is_dir():
            rows.append(f"{_rel(ctx.workspace, child)}/")
        else:
            rows.append(f"{_rel(ctx.workspace, child)}  ({child.stat().st_size:,} bytes)")
        if len(rows) >= TOOL_LIST_MAX:
            rows.append(f"... more entries not shown (limit {TOOL_LIST_MAX})")
            break
    return "\n".join(rows) or f"'{_rel(ctx.workspace, p)}' is empty."


async def tool_read_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    rel = str(args.get("path") or "")
    if not rel:
        raise ValueError("path is required")
    p = workspace_path(ctx.workspace, rel, ctx.unrestricted)
    if not p.is_file():
        raise ValueError(f"'{rel}' is not a file in the workspace")
    text = _decode_text(p.read_bytes()[:2_000_000])
    if text is None:
        raise ValueError(f"'{rel}' is not a text file")
    text = text.replace("\r\n", "\n")     # the model works with \n; edit_file maps it back to the file's style
    if len(text) > TOOL_RESULT_CHARS:
        return text[:TOOL_RESULT_CHARS] + f"\n\n[... cut after {TOOL_RESULT_CHARS:,} of {len(text):,} characters]"
    return text or "(empty file)"


async def tool_write_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    rel = str(args.get("path") or "")
    if not rel:
        raise ValueError("path is required")
    content = args.get("content")
    if content is None:
        raise ValueError("content is required")
    if not isinstance(content, str):
        content = json.dumps(content, indent=2, ensure_ascii=False)
    p = workspace_path(ctx.workspace, rel, ctx.unrestricted)
    if p.is_dir():
        raise ValueError(f"'{rel}' is a folder")
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content.encode("utf-8"))       # bytes, so Windows does not turn \n into \r\n (or \r\n into \r\r\n)
    return f"{'Overwrote' if existed else 'Created'} {_rel(ctx.workspace, p)} ({len(content):,} characters)."


async def tool_run_python(ctx: ToolContext, args: dict[str, Any]) -> str:
    code = str(args.get("code") or "")
    if not code.strip():
        raise ValueError("code is required")
    ctx.workspace.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code, cwd=str(ctx.workspace),
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), TOOL_PYTHON_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Timed out after {TOOL_PYTHON_TIMEOUT_S} s and was killed."
    parts = [f"exit code {proc.returncode}"]
    if out:
        parts.append("stdout:\n" + out.decode("utf-8", "replace"))
    if err:
        parts.append("stderr:\n" + err.decode("utf-8", "replace"))
    return "\n".join(parts)[:TOOL_RESULT_CHARS]



async def tool_edit_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    """Exact-text replacement; refuses ambiguous matches so the model cannot edit the wrong spot."""
    rel = str(args.get("path") or "")
    old = args.get("old_text")
    new = args.get("new_text")
    if not rel:
        raise ValueError("path is required")
    if not isinstance(old, str) or not old:
        raise ValueError("old_text is required and must be a non-empty string")
    if not isinstance(new, str):
        raise ValueError("new_text is required (use an empty string to delete old_text)")
    p = workspace_path(ctx.workspace, rel, ctx.unrestricted)
    if not p.is_file():
        raise ValueError(f"'{rel}' is not a file in the workspace")
    raw = _decode_text(p.read_bytes()[:2_000_000])
    if raw is None:
        raise ValueError(f"'{rel}' is not a text file")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
    count = text.count(old)
    if count == 0:
        raise ValueError("old_text was not found in the file. Read the file again and copy the text exactly, "
                         "including indentation.")
    replace_all = bool(args.get("replace_all"))
    if count > 1 and not replace_all:
        raise ValueError(f"old_text occurs {count} times; include more surrounding lines to make it unique, "
                         "or set replace_all to true.")
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    p.write_bytes((updated.replace("\n", "\r\n") if crlf else updated).encode("utf-8"))
    before, after = text.count("\n") + 1, updated.count("\n") + 1
    return (f"Replaced {count if replace_all else 1} occurrence{'s' if replace_all and count != 1 else ''} in "
            f"{_rel(ctx.workspace, p)}; the file now has {after:,} lines (was {before:,}).")


SEARCH_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode", "dist", "build"}
SEARCH_MAX_FILE_BYTES = 2_000_000
SEARCH_MAX_FILES = 5000


async def tool_search_files(ctx: ToolContext, args: dict[str, Any]) -> str:
    """Grep-style search: lines matching a substring (default) or regex, with the file path and line number."""
    pattern = str(args.get("pattern") or "")
    if not pattern:
        raise ValueError("pattern is required")
    root = workspace_path(ctx.workspace, str(args.get("path") or "."), ctx.unrestricted)
    if not root.exists():
        raise ValueError(f"'{args.get('path') or '.'}' does not exist")
    glob = str(args.get("glob") or "").strip() or "*"
    limit = max(1, min(int(args.get("max_results") or 50), 200))
    flags = 0 if args.get("case_sensitive") else re.IGNORECASE
    try:
        rx = re.compile(pattern if args.get("regex") else re.escape(pattern), flags)
    except re.error as e:
        raise ValueError(f"invalid regular expression: {e}")

    def run() -> str:
        hits: list[str] = []
        scanned = 0
        files = [root] if root.is_file() else sorted(
            (f for f in root.rglob(glob)
             if f.is_file() and not (SEARCH_SKIP_DIRS & set(f.relative_to(root).parts[:-1]))),
            key=lambda f: str(f).lower())
        for f in files:
            if scanned >= SEARCH_MAX_FILES or len(hits) >= limit:
                break
            scanned += 1
            try:
                if f.stat().st_size > SEARCH_MAX_FILE_BYTES:
                    continue
                text = _decode_text(f.read_bytes())
            except OSError:
                continue
            if text is None:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{_rel(ctx.workspace, f)}:{i}: {line.strip()[:240]}")
                    if len(hits) >= limit:
                        break
        if not hits:
            return f"No matches for {pattern!r} in {scanned} file{'s' if scanned != 1 else ''}."
        head = f"{len(hits)} match{'es' if len(hits) != 1 else ''} in {scanned} file{'s' if scanned != 1 else ''} scanned"
        if len(hits) >= limit:
            head += f" (stopped at the limit of {limit}; narrow the pattern or path for more)"
        return head + ":\n" + "\n".join(hits)

    return await asyncio.to_thread(run)


async def tool_get_datetime(ctx: ToolContext, args: dict[str, Any]) -> str:
    now = datetime.now().astimezone()
    utc = now.astimezone(timezone.utc)
    return (f"Local time: {now.strftime('%A, %d %B %Y, %H:%M:%S')} ({now.tzname()}, UTC{now.strftime('%z')[:3]}:{now.strftime('%z')[3:]})\n"
            f"ISO 8601: {now.isoformat(timespec='seconds')}\n"
            f"UTC: {utc.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Unix timestamp: {int(now.timestamp())}")


async def tool_delete_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    rel = str(args.get("path") or "")
    if not rel:
        raise ValueError("path is required")
    p = workspace_path(ctx.workspace, rel, ctx.unrestricted)
    if p == ctx.workspace.resolve():
        raise ValueError("refusing to delete the workspace folder itself")
    if p.is_dir():
        if any(p.iterdir()):
            raise ValueError(f"'{rel}' is a folder that is not empty; delete its files first")
        p.rmdir()
        return f"Deleted the empty folder {_rel(ctx.workspace, p)}."
    if not p.is_file():
        raise ValueError(f"'{rel}' does not exist")
    size = p.stat().st_size
    p.unlink()
    return f"Deleted {_rel(ctx.workspace, p)} ({size:,} bytes)."


async def tool_move_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    src_rel, dst_rel = str(args.get("source") or ""), str(args.get("destination") or "")
    if not src_rel or not dst_rel:
        raise ValueError("source and destination are required")
    src = workspace_path(ctx.workspace, src_rel, ctx.unrestricted)
    dst = workspace_path(ctx.workspace, dst_rel, ctx.unrestricted)
    if not src.exists():
        raise ValueError(f"'{src_rel}' does not exist")
    if dst.is_dir():
        dst = dst / src.name
    if dst.exists() and not args.get("overwrite"):
        raise ValueError(f"'{_rel(ctx.workspace, dst)}' already exists; set overwrite to true to replace it")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dst)
    return f"Moved {_rel(ctx.workspace, src)} to {_rel(ctx.workspace, dst)}."


import ast as _ast
import math as _math
import operator as _op

_CALC_OPS = {_ast.Add: _op.add, _ast.Sub: _op.sub, _ast.Mult: _op.mul, _ast.Div: _op.truediv, _ast.FloorDiv: _op.floordiv,
             _ast.Mod: _op.mod, _ast.Pow: _op.pow, _ast.USub: _op.neg, _ast.UAdd: _op.pos}
_CALC_FUNCS = {n: getattr(_math, n) for n in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "log", "log10",
                                               "log2", "exp", "floor", "ceil", "degrees", "radians", "hypot", "gcd")}
_CALC_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow,
                    "factorial": lambda n: _math.factorial(int(n)) if 0 <= n <= 2000 else (_ for _ in ()).throw(ValueError("factorial argument must be 0-2000"))})
_CALC_NAMES = {"pi": _math.pi, "e": _math.e, "tau": _math.tau, "inf": _math.inf}


def _calc_eval(node: Any) -> float:
    if isinstance(node, _ast.Expression):
        return _calc_eval(node.body)
    if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, _ast.BinOp) and type(node.op) in _CALC_OPS:
        left, right = _calc_eval(node.left), _calc_eval(node.right)
        if isinstance(node.op, _ast.Pow) and abs(right) > 10_000:
            raise ValueError("exponent too large")
        return _CALC_OPS[type(node.op)](left, right)
    if isinstance(node, _ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_calc_eval(node.operand))
    if isinstance(node, _ast.Name) and node.id in _CALC_NAMES:
        return _CALC_NAMES[node.id]
    if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) and node.func.id in _CALC_FUNCS and not node.keywords:
        return _CALC_FUNCS[node.func.id](*(_calc_eval(a) for a in node.args))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


async def tool_calculate(ctx: ToolContext, args: dict[str, Any]) -> str:
    expr = str(args.get("expression") or "").strip().replace("^", "**")
    if not expr:
        raise ValueError("expression is required")
    if len(expr) > 500:
        raise ValueError("expression too long")
    try:
        tree = _ast.parse(expr, mode="eval")
        value = _calc_eval(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        raise ValueError(f"cannot evaluate: {e}")
    if isinstance(value, float):
        shown = f"{value:.12g}"
        if value.is_integer() and abs(value) < 1e15:
            shown = f"{int(value):,}"
    else:
        shown = f"{value:,}" if abs(value) < 10**30 else str(value)
    return f"{expr} = {shown}"


async def tool_read_attachment(ctx: ToolContext, args: dict[str, Any]) -> str:
    atts = ctx.attachments
    if not atts:
        return "No files are attached to this chat."
    name = str(args.get("name") or "").strip()
    if not name:
        return "Attached files:\n" + "\n".join(f"- {a.get('name')} ({a.get('chars', len(a.get('text') or '')):,} characters"
                                              f"{', ' + a['type'] if a.get('type') else ''})" for a in atts)
    match = next((a for a in atts if (a.get("name") or "").lower() == name.lower()), None) or \
        next((a for a in atts if name.lower() in (a.get("name") or "").lower()), None)
    if not match:
        raise ValueError(f"no attachment named '{name}'. Available: {', '.join(a.get('name', '?') for a in atts)}")
    text = match.get("text") or ""
    start = max(0, int(args.get("start") or 0))
    chunk = text[start:start + TOOL_RESULT_CHARS - 200]
    note = ""
    if start + len(chunk) < len(text):
        note = f"\n\n[characters {start:,}-{start + len(chunk):,} of {len(text):,}; call again with start={start + len(chunk)} for more]"
    elif start:
        note = f"\n\n[characters {start:,}-{len(text):,} of {len(text):,}; end of file]"
    return (chunk or "(empty)") + note


async def tool_recall_chats(ctx: ToolContext, args: dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    limit = max(1, min(int(args.get("max_results") or 8), 25))
    q = query.lower()
    words = [w for w in q.split() if len(w) > 1] or [q]

    def run() -> str:
        rows: list[tuple[str, str]] = []
        for p in sorted(CHATS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                chat = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if chat.get("id") == ctx.chat_id:
                continue
            snippets = []
            for m in chat.get("messages", []):
                content = m.get("content") or ""
                low = content.lower()
                # exact phrase first; otherwise every word somewhere in the message
                i = low.find(q)
                if i < 0 and all(w in low for w in words):
                    i = low.find(words[0])
                if i >= 0:
                    a, b = max(0, i - 120), min(len(content), i + len(query) + 160)
                    snippets.append(f"  [{m.get('role')}] …{content[a:b].replace(chr(10), ' ')}…")
                if len(snippets) >= 2:
                    break
            if snippets:
                when = (chat.get("updated_at") or "")[:10]
                rows.append((f"Chat \"{chat.get('title') or 'Untitled'}\" ({when}, id {chat.get('id')}):", "\n".join(snippets)))
            if len(rows) >= limit:
                break
        if not rows:
            return f"No saved chat mentions {query!r}."
        return f"{len(rows)} chat{'s' if len(rows) != 1 else ''} mention {query!r}:\n" + "\n".join(f"{h}\n{s}" for h, s in rows)

    return await asyncio.to_thread(run)


TOOL_SHELL_TIMEOUT_S = 60


async def tool_run_shell(ctx: ToolContext, args: dict[str, Any]) -> str:
    command = str(args.get("command") or "")
    if not command.strip():
        raise ValueError("command is required")
    timeout = max(1, min(int(args.get("timeout") or TOOL_SHELL_TIMEOUT_S), 300))
    ctx.workspace.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        argv = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command]
    else:
        argv = ["/bin/sh", "-c", command]
    proc = await asyncio.create_subprocess_exec(*argv, cwd=str(ctx.workspace), stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Timed out after {timeout} s and was killed."
    parts = [f"exit code {proc.returncode}"]
    if out:
        parts.append("stdout:\n" + out.decode("utf-8", "replace"))
    if err:
        parts.append("stderr:\n" + err.decode("utf-8", "replace"))
    return "\n".join(parts)[:TOOL_RESULT_CHARS]


async def tool_open_path(ctx: ToolContext, args: dict[str, Any]) -> str:
    target = str(args.get("target") or "").strip()
    if not target:
        raise ValueError("target is required")
    if re.match(r"^https?://", target):
        what = target
    else:
        p = workspace_path(ctx.workspace, target, ctx.unrestricted)
        if not p.exists():
            raise ValueError(f"'{target}' does not exist")
        what = str(p)
    if os.name == "nt":
        os.startfile(what)  # type: ignore[attr-defined]
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        await asyncio.create_subprocess_exec(opener, what, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    return f"Opened {what} with the default application."


NOTES_FILE = Path(os.environ.get("LLMHARNESS_NOTES", ROOT / "memory.md"))
NOTES_PROMPT_CHARS = 4000


def load_notes() -> str:
    try:
        return NOTES_FILE.read_text(encoding="utf-8") if NOTES_FILE.exists() else ""
    except OSError:
        return ""


async def tool_remember(ctx: ToolContext, args: dict[str, Any]) -> str:
    note = " ".join(str(args.get("note") or "").split())
    if not note:
        raise ValueError("note is required")
    if len(note) > 500:
        raise ValueError("keep a note under 500 characters")
    line = f"- [{datetime.now().strftime('%Y-%m-%d')}] {note}\n"
    with NOTES_FILE.open("a", encoding="utf-8") as f:
        f.write(line)
    return f"Saved. The notes file now has {len(load_notes().splitlines())} entries; they are shown to you at the start of every agent reply."


async def tool_ask_user(ctx: ToolContext, args: dict[str, Any]) -> str:
    # Never executed: the generation loop answers ask_user from the chat UI directly.
    raise ValueError("ask_user is handled by the chat")

# severity: 0 no side effects, 1 reads the workspace, 2 changes files or notes, 3 executes code (UI ordering / grouping).
# approval=False: runs without asking (web lookups are read-only and leave nothing on disk).
# approval=True: the UI shows Allow / Deny and the reply waits for the answer.
# default: whether the tool is on when agent tools are first enabled.
TOOLS: dict[str, dict[str, Any]] = {
    "get_datetime": {
        "description": "Return the current local date and time, the timezone, and the UTC time.",
        "parameters": {"type": "object", "properties": {}},
        "severity": 0, "approval": False, "default": True, "handler": tool_get_datetime,
    },
    "calculate": {
        "description": "Evaluate an arithmetic expression exactly (+ - * / // % ** and functions like sqrt, sin, log, "
                       "factorial; constants pi and e). Use it instead of doing arithmetic in your head.",
        "parameters": {"type": "object", "required": ["expression"], "properties": {
            "expression": {"type": "string", "description": "e.g. (1200 * 1.19) / 12 or sqrt(2) * pi"}}},
        "severity": 0, "approval": False, "default": True, "handler": tool_calculate,
    },
    "web_search": {
        "description": "Search the web for current information. Returns numbered results with title, URL and "
                       "extracted page text. Cite results inline as [1], [2], ...",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "A concise search query"},
            "max_results": {"type": "integer", "description": "How many results to return (1-10)"}}},
        "severity": 0, "approval": False, "default": True, "handler": tool_web_search,
    },
    "fetch_page": {
        "description": "Download one web page and return its readable text.",
        "parameters": {"type": "object", "required": ["url"], "properties": {
            "url": {"type": "string", "description": "Full http(s) URL"}}},
        "severity": 0, "approval": False, "default": True, "handler": tool_fetch_page,
    },
    "read_attachment": {
        "description": "Read the full text of a file the user attached to this chat. Call without a name to list the "
                       "attachments; use start to page through long files.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Attachment file name (omit to list them)"},
            "start": {"type": "integer", "description": "Character offset to continue from"}}},
        "severity": 0, "approval": False, "default": True, "handler": tool_read_attachment,
    },
    "recall_chats": {
        "description": "Search the user's saved chats for a keyword and return matching snippets with the chat title "
                       "and date, to recall earlier conversations.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "Keyword or phrase to look for"},
            "max_results": {"type": "integer", "description": "How many chats to return (1-25, default 8)"}}},
        "severity": 0, "approval": True, "default": True, "handler": tool_recall_chats,
    },
    "ask_user": {
        "description": "Ask the user a clarifying question and wait for the answer before continuing. Use it when the "
                       "task is ambiguous instead of guessing. Optionally offer a few choices.",
        "parameters": {"type": "object", "required": ["question"], "properties": {
            "question": {"type": "string", "description": "The question, in one or two sentences"},
            "options": {"type": "array", "items": {"type": "string"}, "description": "Up to 5 short answer choices"}}},
        "severity": 0, "approval": True, "fixed": True, "default": True, "handler": tool_ask_user,
    },
    "list_files": {
        "description": "List files and folders in the workspace directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Folder relative to the workspace; omit for the root"}}},
        "severity": 1, "approval": True, "default": True, "handler": tool_list_files,
    },
    "search_files": {
        "description": "Search text files in the workspace for a substring (or a regular expression with regex=true). "
                       "Returns matching lines as path:line: text.",
        "parameters": {"type": "object", "required": ["pattern"], "properties": {
            "pattern": {"type": "string", "description": "Text or regular expression to look for"},
            "path": {"type": "string", "description": "Folder or file relative to the workspace; omit for everything"},
            "glob": {"type": "string", "description": "Only files matching this name pattern, e.g. *.py"},
            "regex": {"type": "boolean", "description": "Treat pattern as a regular expression"},
            "case_sensitive": {"type": "boolean", "description": "Match case exactly (default: ignore case)"},
            "max_results": {"type": "integer", "description": "Maximum matching lines to return (1-200, default 50)"}}},
        "severity": 1, "approval": True, "default": True, "handler": tool_search_files,
    },
    "read_file": {
        "description": "Read a UTF-8 text file from the workspace directory.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace"}}},
        "severity": 1, "approval": True, "default": True, "handler": tool_read_file,
    },
    "remember": {
        "description": "Save a short note (a fact, preference or decision) to a persistent notes file that is shown to "
                       "you at the start of every agent reply, across chats.",
        "parameters": {"type": "object", "required": ["note"], "properties": {
            "note": {"type": "string", "description": "One sentence to remember"}}},
        "severity": 2, "approval": True, "default": True, "handler": tool_remember,
    },
    "open_path": {
        "description": "Open a workspace file or an http(s) URL on the user's screen with the default application "
                       "(browser, editor, viewer).",
        "parameters": {"type": "object", "required": ["target"], "properties": {
            "target": {"type": "string", "description": "File path relative to the workspace, or a URL"}}},
        "severity": 2, "approval": True, "default": False, "handler": tool_open_path,
    },
    "write_file": {
        "description": "Create or overwrite a text file in the workspace directory. Parent folders are created.",
        "parameters": {"type": "object", "required": ["path", "content"], "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace"},
            "content": {"type": "string", "description": "The complete file content"}}},
        "severity": 2, "approval": True, "default": True, "critical": True, "handler": tool_write_file,
    },
    "edit_file": {
        "description": "Replace an exact piece of text in a workspace file with new text. old_text must match the file "
                       "exactly (including indentation) and occur once, unless replace_all is true. Prefer this over "
                       "write_file for changes to existing files.",
        "parameters": {"type": "object", "required": ["path", "old_text", "new_text"], "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace"},
            "old_text": {"type": "string", "description": "The exact text to replace"},
            "new_text": {"type": "string", "description": "The replacement text (empty string deletes old_text)"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match"}}},
        "severity": 2, "approval": True, "default": True, "critical": True, "handler": tool_edit_file,
    },
    "move_file": {
        "description": "Move or rename a file or folder inside the workspace directory. Parent folders are created.",
        "parameters": {"type": "object", "required": ["source", "destination"], "properties": {
            "source": {"type": "string", "description": "Existing path relative to the workspace"},
            "destination": {"type": "string", "description": "New path (or an existing folder to move into)"},
            "overwrite": {"type": "boolean", "description": "Replace the destination if it already exists"}}},
        "severity": 2, "approval": True, "default": True, "critical": True, "handler": tool_move_file,
    },
    "delete_file": {
        "description": "Delete a file (or an empty folder) in the workspace directory.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace"}}},
        "severity": 2, "approval": True, "default": True, "critical": True, "handler": tool_delete_file,
    },
    "run_python": {
        "description": f"Run a Python script with the workspace as working directory ({TOOL_PYTHON_TIMEOUT_S} s limit). "
                       "Returns the exit code, stdout and stderr.",
        "parameters": {"type": "object", "required": ["code"], "properties": {
            "code": {"type": "string", "description": "Python source code to execute"}}},
        "severity": 3, "approval": True, "default": False, "critical": True, "handler": tool_run_python,
    },
    "run_shell": {
        "description": f"Run a shell command ({'PowerShell' if os.name == 'nt' else 'sh'}) with the workspace as working "
                       f"directory and return exit code, stdout and stderr. Default timeout {TOOL_SHELL_TIMEOUT_S} s.",
        "parameters": {"type": "object", "required": ["command"], "properties": {
            "command": {"type": "string", "description": "The command line to run"},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed (max 300)"}}},
        "severity": 3, "approval": True, "default": False, "critical": True, "handler": tool_run_shell,
    },
}


FILE_TOOLS = ("list_files", "read_file", "write_file", "edit_file", "search_files", "delete_file", "move_file", "open_path")


def tool_specs(names: list[str], unrestricted: bool = False) -> list[dict[str, Any]]:
    specs = []
    for n in names:
        if n not in TOOLS:
            continue
        desc = TOOLS[n]["description"]
        if unrestricted and n in FILE_TOOLS:
            desc += " Absolute paths anywhere on this computer are allowed as well."
        params = json.loads(json.dumps(TOOLS[n]["parameters"]))
        params.setdefault("properties", {})[PURPOSE_FIELD] = {
            "type": "string",
            "description": "One short sentence for the user: what this call does and why (e.g. "
                           "'List the src folder to find the entry point')."}
        params["required"] = list(params.get("required", [])) + [PURPOSE_FIELD]
        specs.append({"type": "function", "function": {"name": n, "description": desc, "parameters": params}})
    return specs


PURPOSE_FIELD = "purpose"


async def summarize_tool_call(client: httpx.AsyncClient, model: str, caps: list[str], name: str,
                              args: dict[str, Any], user_msg: str, extras: dict[str, Any] | None = None) -> str:
    """Fallback intent line when the model left `purpose` out. Never raises; returns "" on failure."""
    shown = {k: (v[:400] + "..." if isinstance(v, str) and len(v) > 400 else v) for k, v in args.items()}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "An assistant is about to call a tool while helping a user. Describe in ONE short "
                                          "sentence (max 20 words) what the call does and why, for the user to approve. "
                                          "Reply with the sentence only."},
            {"role": "user", "content": f"User's request: {user_msg[:600]}\n\nTool: {name}\nArguments: "
                                        f"{json.dumps(shown, ensure_ascii=False)}"},
        ],
        "options": {"temperature": 0, "num_predict": 48, **(extras or {}).get("options", {})},
        **{k: v for k, v in (extras or {}).items() if k != "options"},
    }
    tp = think_param("off", caps)
    if tp is not None:
        payload["think"] = tp
    try:
        raw = await ollama_chat_once(client, payload, timeout=30)
    except httpx.HTTPError as e:
        log.info("tool summary failed: %s", e)
        return ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    return raw.splitlines()[0].strip().strip('"')[:200] if raw else ""


TOOL_POLICIES = ("ask", "auto")


def tool_needs_approval(name: str, settings: dict[str, Any]) -> bool:
    """Effective approval rule: the server-wide policy if one is set, else the tool's default."""
    spec = TOOLS.get(name)
    if not spec:
        return False
    policy = (settings.get("tool_policies") or {}).get(name)
    if spec.get("fixed"):
        return bool(spec["approval"])
    if policy in TOOL_POLICIES:
        return policy == "ask"
    return bool(spec["approval"])


def clean_tool_policies(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {n: p for n, p in raw.items() if n in TOOLS and p in TOOL_POLICIES}


def public_tools(settings: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": n, "description": t["description"], "approval": tool_needs_approval(n, settings),
             "default_approval": t["approval"], "critical": bool(t.get("critical")), "fixed": bool(t.get("fixed")),
             "severity": int(t.get("severity", 0)), "default": t["default"]}
            for n, t in TOOLS.items()]


def agent_instructions(ctx: ToolContext, names: list[str]) -> str:
    lines = [
        "You can call tools. Use them when they help with the task and answer directly when they do not.",
        "Tool results are visible only to you: restate what matters for the user in your answer.",
        "Some tool calls need the user's approval. If a call is declined, do not repeat it; continue without it "
        "or explain what you would need.",
        "Every tool call includes a 'purpose' argument: one short sentence for the user saying what the call does "
        "and why (for example 'Read config.json to find the port number').",
    ]
    if "remember" in names:
        notes = load_notes().strip()
        if notes:
            lines.append("Notes you saved earlier with the remember tool:\n" + notes[-NOTES_PROMPT_CHARS:])
    if "ask_user" in names:
        lines.append("If the request is ambiguous in a way that changes the outcome, use ask_user once rather than guessing.")
    if any(n in names for n in FILE_TOOLS + ("run_python", "run_shell")):
        if ctx.unrestricted:
            lines.append(f"File tools default to the workspace folder {ctx.workspace} for relative paths, "
                         "and absolute paths anywhere on this computer are allowed.")
        else:
            lines.append(f"File tools work inside the workspace folder {ctx.workspace}. Use paths relative to it.")
    return "\n".join(lines)


def expand_tool_history(msgs: list[dict[str, Any]], with_tools: bool) -> list[dict[str, Any]]:
    """Turn stored assistant messages that used tools back into the assistant / tool turn sequence.

    Without tool support the calls are dropped and only the final text is sent.
    """
    out: list[dict[str, Any]] = []
    for m in msgs:
        calls = m.pop("tool_calls", None)
        if m.get("role") == "assistant" and calls and with_tools:
            out.append({"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": c.get("name", ""), "arguments": c.get("arguments") or {}}}
                                       for c in calls]})
            for c in calls:
                out.append({"role": "tool", "tool_name": c.get("name", ""), "content": c.get("result") or c.get("status") or ""})
            if m.get("content"):
                out.append({"role": "assistant", "content": m["content"]})
        elif m.get("content") or m.get("images") or m.get("attachments"):
            out.append(m)
    return out


def _call_key(name: str, args: Any) -> str:
    try:
        return name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return name + ":" + repr(args)


async def run_tool(ctx: ToolContext, name: str, args: Any) -> tuple[str, str]:
    """Execute one tool call; returns (status, result text). Never raises."""
    spec = TOOLS.get(name)
    if not spec:
        return "error", f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}."
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return "error", "arguments must be a JSON object"
    if not isinstance(args, dict):
        return "error", "arguments must be a JSON object"
    try:
        out = await spec["handler"](ctx, args)
        return "ok", (out or "(no output)")[:TOOL_RESULT_CHARS]
    except Exception as e:  # tool failures go back to the model as text, never to the user as a crash
        log.info("tool %s failed: %s: %s", name, e.__class__.__name__, e)
        return "error", f"{name} failed: {e.__class__.__name__}: {e}"


_tool_approvals: dict[tuple[str, str], asyncio.Future] = {}


async def wait_for_approval(gen_id: str, call_id: str, cancel: asyncio.Event, request: Request) -> dict[str, Any] | None:
    """Block until the UI answers ({"approved": bool, "answer": str}); None when the reply was stopped or the client left."""
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    _tool_approvals[(gen_id, call_id)] = fut
    try:
        while not fut.done():
            if cancel.is_set() or await request.is_disconnected():
                return None
            await asyncio.wait({fut}, timeout=0.5)
        return fut.result()
    finally:
        _tool_approvals.pop((gen_id, call_id), None)


DECLINED_NOTE = "The user declined this tool call. Do not call it again; continue without it or explain what you need."


# ----------------------------------------------------------------------------- generation

_cancel_events: dict[str, asyncio.Event] = {}


async def generate_stream(req: dict[str, Any], request: Request) -> AsyncIterator[str]:
    gen_id = str(req.get("gen_id") or uuid.uuid4().hex)
    cancel = _cancel_events.setdefault(gen_id, asyncio.Event())
    settings = load_settings()

    model = req.get("model") or ""
    system_prompt = (req.get("system_prompt") or "").strip()
    history: list[dict[str, Any]] = [
        {k: v for k, v in m.items() if k in ("role", "content", "images", "attachments", "tool_calls")}
        for m in req.get("messages", [])
        if m.get("role") in ("user", "assistant")
        and (m.get("content") or m.get("images") or m.get("attachments") or m.get("tool_calls"))
    ]
    options = clean_options(req.get("options"))
    think_mode = req.get("think", "default")
    loop_cfg = req.get("loop_guard") or {}
    search_cfg = req.get("web_search") or {}
    agent_cfg = req.get("agent") or {}
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

        # --- agent tools ----------------------------------------------------------------
        tool_names = [n for n in (agent_cfg.get("tools") or []) if n in TOOLS]
        agent_active = bool(agent_cfg.get("enabled")) and bool(tool_names)
        if agent_active and "tools" not in caps:
            agent_active = False
            yield sse({"type": "status", "phase": "tools",
                       "detail": f"{model} does not report tool support; agent tools are off for this reply"})
        max_steps = max(1, min(int(agent_cfg.get("max_steps") or TOOL_STEPS_DEFAULT), 25))
        tool_ctx = ToolContext(client, settings, search_cfg,
                               attachments=[a for m in history for a in (m.get("attachments") or []) if a.get("text")],
                               chat_id=chat_id)
        tool_records: list[dict[str, Any]] = []

        outgoing = [dict(m) for m in history]
        att_info = expand_attachments(outgoing, options)
        if att_info["chars"]:
            detail = f"Attached files use about {att_info['tokens']:,} tokens"
            if att_info["truncated"]:
                detail += "; some content was cut to fit the context length"
            yield sse({"type": "status", "phase": "attachments", "detail": detail})
        outgoing = expand_tool_history(outgoing, agent_active)

        # --- optional web search ------------------------------------------------------
        # With the web_search tool available the model decides itself, so the planner only
        # runs for "search every message"; without tools the pre-search works as before.
        mode = (search_cfg.get("mode") or "off").lower()
        pre_search = mode in ("auto", "always") and not (agent_active and "web_search" in tool_names and mode == "auto")
        if pre_search:
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
                    tool_ctx.sources = list(sources)      # later tool searches continue the numbering
                    tool_ctx.queries = [query]
                    yield sse({"type": "sources", "query": query, "items": sources})
                    outgoing[-1] = {**outgoing[-1],
                                    "content": build_search_context(query, results) + "\n\nUser's message:\n" + outgoing[-1]["content"]}
            else:
                yield sse({"type": "status", "phase": "planning", "detail": "No search needed"})

        # --- main streamed completion (looped while the model calls tools) --------------------
        if agent_active:
            extra = agent_instructions(tool_ctx, tool_names)
            system_prompt = f"{system_prompt}\n\n{extra}" if system_prompt else extra
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + outgoing
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True,
                                   "options": options, "keep_alive": settings.get("keep_alive", "5m")}
        tp = think_param(think_mode, caps)
        if tp is not None:
            payload["think"] = tp

        guard_on = loop_cfg.get("enabled", True)
        threshold = loop_cfg.get("threshold", 3)
        guard = LoopGuard(threshold) if guard_on else None
        think_guard = LoopGuard(threshold) if guard_on else None
        think_budget = int(req.get("think_budget") or 0)
        think_tokens = 0
        # Models with native thinking stream it in msg["thinking"]; a "<think>" in their content is prose.
        splitter = ThinkSplitter(enabled="thinking" not in caps)
        pending_sep = False     # a paragraph break goes before the first text after a tool step
        yield sse({"type": "status", "phase": "generating", "detail": "Waiting for the model"})

        def take(kind: str, text: str) -> tuple[str | None, list[str]]:
            """Append a piece to the assistant message; returns (abort_reason, events)."""
            nonlocal pending_sep
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
            if pending_sep:
                pending_sep = False
                if assistant["content"] and not assistant["content"].endswith("\n\n"):
                    text = "\n\n" + text
            assistant["content"] += text
            events.append(sse({"type": "content", "delta": text}))
            if guard:
                reason = guard.feed(text)
                if reason:
                    return reason, events
            return None, events

        def tool_event(rec: dict[str, Any], kind: str, **extra: Any) -> str:
            base = {"type": kind, "id": rec["id"], "name": rec["name"], "arguments": rec["arguments"],
                    "purpose": rec.get("purpose", ""), "step": rec["step"], "status": rec["status"]}
            return sse({**base, **extra})

        totals = {"tokens": 0, "eval_seconds": 0.0, "load_seconds": 0.0, "total_seconds": 0.0}
        prompt_tokens: int | None = None
        done_reason: str | None = None
        seen_calls: dict[str, int] = {}
        step = 0
        try:
            while True:
                step += 1
                step_calls: list[dict[str, Any]] = []
                step_start = len(assistant["content"])
                if agent_active:
                    if step < max_steps:
                        payload["tools"] = tool_specs(tool_names, tool_ctx.unrestricted)
                    else:
                        payload.pop("tools", None)
                        if step > 1:
                            yield sse({"type": "status", "phase": "generating",
                                       "detail": f"Step limit ({max_steps}) reached; asking for a final answer"})
                if step > 1:
                    pending_sep = True
                    if guard:
                        guard = LoopGuard(threshold)   # sentences repeated across tool steps are normal
                    yield sse({"type": "status", "phase": "generating", "detail": "Continuing after tools"})

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
                        if first_token_at is None and (msg.get("thinking") or msg.get("content") or msg.get("tool_calls")):
                            first_token_at = time.monotonic()
                        token_count += 1
                        if msg.get("thinking"):
                            think_tokens += 1
                        for tc in msg.get("tool_calls") or []:
                            step_calls.append(tc)
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
                            totals["tokens"] += chunk.get("eval_count") or 0
                            totals["eval_seconds"] += ev / 1e9
                            totals["load_seconds"] += (chunk.get("load_duration") or 0) / 1e9
                            totals["total_seconds"] += (chunk.get("total_duration") or 0) / 1e9
                            prompt_tokens = chunk.get("prompt_eval_count") or prompt_tokens
                            done_reason = chunk.get("done_reason")
                            break

                if aborted or not step_calls:
                    break

                # --- run the tool calls of this step --------------------------------------
                payload["messages"].append({"role": "assistant", "content": assistant["content"][step_start:],
                                            "tool_calls": step_calls})
                for tc in step_calls:
                    fn = tc.get("function") or {}
                    name = str(fn.get("name") or "")
                    args = fn.get("arguments")
                    if args is None:
                        args = {}
                    purpose = ""
                    if isinstance(args, dict):
                        args = dict(args)
                        purpose = str(args.pop(PURPOSE_FIELD, "") or "").strip()
                    spec = TOOLS.get(name)
                    needs_approval = tool_needs_approval(name, settings)
                    if not purpose and needs_approval:
                        # The model skipped the intent line; let the helper model write one before we ask the user.
                        yield sse({"type": "status", "phase": "approval", "detail": f"Summarising the {name} call"})
                        helper, helper_caps = await helper_model_for(client, settings, model, caps)
                        purpose = await summarize_tool_call(client, helper, helper_caps, name, args, history[-1]["content"],
                                                            extras=helper_extras(helper, model, options, settings))
                    rec: dict[str, Any] = {"id": f"c{len(tool_records) + 1}", "step": step, "name": name,
                                           "arguments": args, "purpose": purpose, "status": "pending", "result": ""}
                    tool_records.append(rec)
                    assistant["tool_calls"] = tool_records
                    yield tool_event(rec, "tool_call", approval=needs_approval)

                    key = _call_key(name, args)
                    seen_calls[key] = seen_calls.get(key, 0) + 1
                    if spec and seen_calls[key] > 2:
                        rec["status"], rec["result"] = "error", ("This exact call was already made in this reply; the result is above. "
                                                                 "Do not repeat it.")
                    elif needs_approval:
                        yield sse({"type": "status", "phase": "approval",
                                   "detail": "Waiting for your answer" if name == "ask_user" else f"Waiting for approval: {name}"})
                        answer = await wait_for_approval(gen_id, rec["id"], cancel, request)
                        if answer is None:
                            rec["status"], rec["result"] = "stopped", "Generation was stopped before this ran."
                            aborted = "stopped" if cancel.is_set() else "client disconnected"
                        elif name == "ask_user":
                            # The user's reply is the tool result; nothing runs.
                            if answer.get("approved") and answer.get("answer", "").strip():
                                rec["status"], rec["result"] = "ok", f"The user answered: {answer['answer'].strip()}"
                            else:
                                rec["status"], rec["result"] = "denied", "The user skipped the question. Proceed with your best judgement and say what you assumed."
                            yield tool_event(rec, "tool_status")
                        elif not answer.get("approved"):
                            rec["status"], rec["result"] = "denied", DECLINED_NOTE

                    if rec["status"] == "pending":
                        rec["status"] = "running"
                        yield tool_event(rec, "tool_status")
                        yield sse({"type": "status", "phase": "tool", "detail": f"Running {name}"})
                        t0 = time.monotonic()
                        rec["status"], rec["result"] = await run_tool(tool_ctx, name, args)
                        rec["seconds"] = round(time.monotonic() - t0, 2)
                        if name == "web_search" and rec["status"] == "ok" and tool_ctx.sources:
                            assistant["sources"] = tool_ctx.sources
                            assistant["search_query"] = " · ".join(tool_ctx.queries)
                            yield sse({"type": "sources", "query": assistant["search_query"], "items": tool_ctx.sources})
                    yield tool_event(rec, "tool_result", result=rec["result"], seconds=rec.get("seconds"))
                    payload["messages"].append({"role": "tool", "tool_name": name, "content": rec["result"]})
                    if aborted:
                        break
                if aborted:
                    break
                if cancel.is_set():
                    aborted = "stopped"
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
        else:
            ev = totals["eval_seconds"]
            stats = {
                "prompt_tokens": prompt_tokens,
                "tokens": totals["tokens"],
                "eval_seconds": round(ev, 2),
                "load_seconds": round(totals["load_seconds"], 2),
                "total_seconds": round(totals["total_seconds"], 2),
                "tokens_per_second": round(totals["tokens"] / ev, 1) if ev else None,
                "done_reason": done_reason,
            }
        if tool_records:
            stats["steps"] = step
            stats["tool_calls"] = len(tool_records)
        assistant["stats"] = stats
        assistant["think_mode"] = think_mode
        yield sse({"type": "done", "stats": stats, "aborted": aborted})

        # --- persist -------------------------------------------------------------------
        chat: dict[str, Any] | None = None
        if save and (assistant["content"] or assistant["thinking"] or tool_records):
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
    data["tool_policies"] = clean_tool_policies(data.get("tool_policies"))
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


@app.post("/api/generate/{gen_id}/tools/{call_id}")
async def answer_tool_call(gen_id: str, call_id: str, body: dict[str, Any]) -> JSONResponse:
    """The UI's Allow / Deny answer for a tool call that is waiting for approval."""
    fut = _tool_approvals.get((gen_id, call_id))
    if not fut or fut.done():
        raise HTTPException(404, "no tool call is waiting for an answer")
    fut.set_result({"approved": bool(body.get("approved")), "answer": str(body.get("answer") or "")})
    return JSONResponse({"ok": True})


@app.get("/api/tools")
async def list_tools() -> JSONResponse:
    settings = load_settings()
    return JSONResponse({"tools": public_tools(settings), "workspace": str(workspace_dir(settings)),
                         "allow_outside_workspace": bool(settings.get("allow_outside_workspace"))})


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
