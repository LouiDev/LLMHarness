# LlmHarness

A local chat harness for Ollama models: pick a model, shape it with a system prompt,
turn thinking on or off, keep it from looping, and save every conversation. Optional
web search grounds answers in fresh results.

## Showcase
![img](https://raw.githubusercontent.com/LouiDev/LLMHarness/assets/assets/showcase.png)

## Run

Double-click `start.bat`, or from a PowerShell prompt:

```powershell
.\run.ps1
```

The script creates `.venv` on first use, installs the dependencies, starts the server on
http://127.0.0.1:8766 and opens the browser. If Ollama is not already running, the
script starts `ollama serve` as a separate detached process, so Ollama keeps running after
the app is closed. Ollama is expected at `http://127.0.0.1:11434` by default; override with
the `OLLAMA_HOST` environment variable.

## Features

- **Model selection** from whatever Ollama has pulled, with capability badges (thinking,
  tools, vision) and the model's context length. Unload a model from memory with one click.
- **System prompt** per chat, with saved presets (stored server-side in `settings.json`).
- **Thinking mode**: model default, off, on, or an effort level for models that support it.
  Models that emit `<think>` tags without native support are parsed too. Thinking streams
  live into a collapsible block and shows how long the model thought.
- **Loop guard**: the reply is stopped when a sentence or paragraph repeats N times or the
  tail of the text is a short fragment repeated over and over. It watches the thinking stream
  as well as the answer. A thinking budget (tokens) and a per-reply time limit are available too.
- **Generation controls**: temperature, top p, top k, min p, repeat penalty and window,
  max tokens, context length, seed. Blank fields fall back to the model's defaults.
  "Use as defaults" stores the whole panel (model, prompt, thinking, search, loop guard)
  server-side in `settings.json`, so new chats start with it in any browser.
- **Saved chats** as JSON files in `chats/`. Chats are saved automatically after the first
  reply and get an auto-generated title. Rename, pin, delete, filter, and export as
  Markdown or JSON. Edit a message and resend, regenerate, or delete messages.
- **Web search**: off, "let the model decide", or "search every message". The model
  proposes a query, results come from DuckDuckGo (no API key) or a SearXNG instance, the
  pages are fetched and summarised into context, and citations like [1] link to the source.
  Messages that explicitly ask for a search ("search the web", "look it up online", "google")
  always trigger one in auto mode. A small helper model can be assigned in server settings to
  do the planning and titling instead of the (possibly slow) chat model.
- **Live readout** while generating: phase (planning, searching, thinking, writing),
  token count, tokens per second, and elapsed time. Stop at any time; partial replies are kept.
- **File attachments**: drop, paste, or pick files in the composer. Text, code, CSV, JSON,
  Markdown, PDF and DOCX are extracted server-side and placed in the prompt within the context
  budget (the composer shows the token cost and warns when the end will be cut). Images are
  only offered for models that report vision; other models refuse them with an explanation.
  Attached text is stored with the chat, so later turns and regenerations still see it.
- **Agent mode**: the Chat / Agent switch above the composer decides whether the model only writes
  replies or may call tools. "Agent options" next to it picks the tools and the step limit. In agent
  mode a tool-capable model can call `web_search`, `fetch_page`, `list_files`, `read_file`, `write_file`
  and `run_python` in a loop until it has an answer. Web lookups run on their own; every other
  tool shows an Allow / Deny prompt in the chat first, so nothing touches the disk without a
  click. File tools and Python are confined to a workspace folder (`workspace/` next to
  `server.py`, changeable in server settings). Each reply shows a collapsible list of the calls
  made with their arguments and results, and the loop stops after a configurable number of
  steps. Models that do not report tool support answer normally with the tools skipped. A server
  setting can lift the workspace restriction so file tools accept absolute paths anywhere on
  the machine; the approval prompt stays in place either way. Server settings also hold a per-tool
  approval rule ("asks first" or "runs on its own"), shared by every chat; switching a file-writing
  or code-running tool to run unattended asks for an extra confirmation.
- **LaTeX math** rendered with KaTeX: `$...$` and `\(...\)` inline, `$$...$$` and `\[...\]` as
  display formulas. Code blocks are left untouched.
- Dark and light themes.

Model capabilities come from Ollama's `/api/show`, because the model list endpoint
under-reports thinking and tool support for some imported GGUF models.

## Example system prompt
```
You are a highly precise and knowledgeable assistant. Your primary goal is factual accuracy. Answer questions directly and concisely.
If you are unsure about a fact or if a query is ambiguous, state your uncertainty clearly rather than guessing and consider doing a web search first if allowed.
Use Markdown (headers, bolding, lists, and code blocks) to organize complex information for maximum readability.
```

## Layout

```
start.bat          Double-click launcher (calls run.ps1)
server.py          FastAPI backend: Ollama proxy, streaming, loop guard, search, agent tools, chat storage
static/index.html  Single-page UI
static/style.css
static/app.js
chats/             Saved conversations (one JSON file each)
settings.json      Server-wide settings: search provider, presets, keep-alive, new-chat defaults
```

## API sketch

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/models` | Installed models with capabilities |
| GET/POST/PUT/DELETE | `/api/chats[/{id}]` | Chat storage |
| GET/PUT | `/api/settings` | Server settings, prompt presets, new-chat defaults |
| POST | `/api/generate` | Server-sent-event stream of a reply |
| POST | `/api/generate/{gen_id}/stop` | Cancel a running reply |
| POST | `/api/search` | Run a web search directly |
