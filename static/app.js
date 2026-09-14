/* LlmHarness front-end: one page, no framework. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v === true) node.setAttribute(k, "");
      else node.setAttribute(k, v);
    }
    for (const c of children.flat()) if (c != null) node.append(c.nodeType ? c : document.createTextNode(String(c)));
    return node;
  };
  const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const fmtTime = (iso) => { const d = new Date(iso); return isNaN(d) ? "" : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); };
  const fmtSecs = (s) => (s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${s.toFixed(1)} s`);
  const hostOf = (u) => { try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return ""; } };

  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts, body: opts.body != null ? JSON.stringify(opts.body) : undefined });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch { /* ignore */ }
      throw new Error(msg);
    }
    return res.status === 204 ? null : res.json();
  }

  let toastTimer;
  function toast(msg, isError = false) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = "toast" + (isError ? " error" : "");
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, isError ? 6000 : 2600);
  }

  // Styled confirmation popup; resolves true when the user confirms.
  function askConfirm(title, text, okLabel = "Confirm") {
    const dlg = $("#confirm-dialog");
    $("#confirm-title").textContent = title;
    $("#confirm-text").textContent = text;
    $("#confirm-ok").textContent = okLabel;
    return new Promise((resolve) => {
      const done = (v) => { dlg.close(); resolve(v); cleanup(); };
      const ok = () => done(true), cancel = () => done(false);
      const onClose = () => { resolve(false); cleanup(); };
      const cleanup = () => { $("#confirm-ok").removeEventListener("click", ok); $("#confirm-cancel").removeEventListener("click", cancel); dlg.removeEventListener("close", onClose); };
      $("#confirm-ok").addEventListener("click", ok);
      $("#confirm-cancel").addEventListener("click", cancel);
      dlg.addEventListener("close", onClose);
      dlg.showModal();
    });
  }

  // ------------------------------------------------------------------ markdown
  const canMarkdown = typeof marked !== "undefined" && typeof DOMPurify !== "undefined";
  if (canMarkdown) {
    marked.setOptions({ gfm: true, breaks: true });
    if (typeof hljs !== "undefined") {
      marked.use({
        renderer: {
          code(code, lang) {
            const text = typeof code === "object" ? code.text : code;
            const language = typeof code === "object" ? code.lang : lang;
            const valid = language && hljs.getLanguage(language) ? language : null;
            const highlighted = valid ? hljs.highlight(text, { language: valid }).value : hljs.highlightAuto(text).value;
            return `<pre><code class="hljs${valid ? " language-" + valid : ""}">${highlighted}</code></pre>`;
          },
        },
      });
    }
    if (typeof katex !== "undefined") {
      // LaTeX math: $$...$$ and \[...\] (display), \(...\) and $...$ (inline).
      // Captured as a marked token so Markdown does not touch the TeX source (backslashes, _, *).
      // Text inside code spans / fences is never matched because marked lexes those first.
      const MATH_PATTERNS = [
        { re: /^\$\$([\s\S]+?)\$\$/, display: true },
        { re: /^\\\[([\s\S]+?)\\\]/, display: true },
        { re: /^\\\(([\s\S]+?)\\\)/, display: false },
        // Single $: no space after the opening / before the closing $, and not followed by a digit ("$5 and $10").
        { re: /^\$(?!\s)((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)/, display: false },
      ];
      marked.use({
        extensions: [{
          name: "math",
          level: "inline",
          start(src) {
            const m = src.match(/\$|\\[\[(]/);
            return m ? m.index : undefined;
          },
          tokenizer(src) {
            for (const { re, display } of MATH_PATTERNS) {
              const m = re.exec(src);
              if (m) return { type: "math", raw: m[0], tex: m[1].trim(), display };
            }
          },
          renderer(token) {
            try {
              return katex.renderToString(token.tex, { displayMode: token.display, throwOnError: false, strict: "ignore" });
            } catch {
              return `<code>${escapeHtml(token.raw)}</code>`;
            }
          },
        }],
      });
    }
  }
  function renderMarkdown(text, sources) {
    if (!canMarkdown) return `<p style="white-space:pre-wrap">${escapeHtml(text)}</p>`;
    let html = DOMPurify.sanitize(marked.parse(text), { ADD_ATTR: ["target"] });
    if (sources && sources.length) {
      html = html.replace(/\[(\d{1,2})\](?![^<]*<\/(?:code|a)>)/g, (m, n) => {
        const s = sources.find((x) => String(x.n) === n);
        return s ? `<sup class="cite"><a href="${escapeHtml(s.url)}" target="_blank" rel="noopener" title="${escapeHtml(s.title)}">${n}</a></sup>` : m;
      });
    }
    return html;
  }
  function decorateCode(container) {
    for (const pre of $$("pre", container)) {
      if ($(".copy-code", pre)) continue;
      pre.append(el("button", { class: "copy-code", type: "button", text: "Copy", onclick: (e) => {
        navigator.clipboard.writeText(pre.querySelector("code")?.innerText || pre.innerText);
        e.currentTarget.textContent = "Copied"; setTimeout(() => (e.currentTarget && (e.currentTarget.textContent = "Copy")), 1200);
      } }));
    }
    for (const a of $$("a[href]", container)) { a.target = "_blank"; a.rel = "noopener"; }
  }

  // ------------------------------------------------------------------ settings model
  const GEN_FIELDS = [
    { key: "temperature", label: "Temperature", min: 0, max: 2, step: 0.05, help: "Higher is more random." },
    { key: "top_p", label: "Top p", min: 0, max: 1, step: 0.01 },
    { key: "top_k", label: "Top k", min: 0, max: 200, step: 1 },
    { key: "min_p", label: "Min p", min: 0, max: 0.5, step: 0.01 },
    { key: "repeat_penalty", label: "Repeat penalty", min: 0.8, max: 2, step: 0.01 },
    { key: "repeat_last_n", label: "Repeat window (tokens)", min: -1, max: 4096, step: 1, help: "-1 uses the whole context, 0 turns the penalty off." },
    { key: "num_predict", label: "Max tokens", plain: true, placeholder: "model default", help: "Includes thinking tokens." },
    { key: "num_ctx", label: "Context length", plain: true, placeholder: "model default" },
    { key: "seed", label: "Seed", plain: true, placeholder: "random" },
  ];

  const FACTORY_SETTINGS = {
    model: "",
    system_prompt: 'You are a helpful, knowledgeable assistant. Lead with the answer, then the reasoning if it adds something. Match the length and depth of your reply to the question; skip filler and do not restate the question. If you are unsure or the request is ambiguous, say so and state your assumption or ask one short question. Use Markdown for structure when it aids readability, fenced code blocks with a language tag for code, and LaTeX ($...$ inline, $$...$$ display) for mathematical notation.',
    think: "default",
    options: { temperature: 0.7, top_p: 0.9, top_k: 40, min_p: 0, repeat_penalty: 1.1, repeat_last_n: 256, num_predict: "", num_ctx: 8192, seed: "" },
    loop_guard: { enabled: true, threshold: 3 },
    think_budget: 0,
    timeout_s: 0,
    web_search: { mode: "off", max_results: 5, fetch_pages: true },
    agent: { enabled: false, tools: null, max_steps: 8, workspace: "" },   // tools: null = the server's default set; workspace: "" = server default
  };
  const deepClone = (o) => JSON.parse(JSON.stringify(o));
  const mergeSettings = (base, extra) => ({ ...deepClone(base), ...deepClone(extra || {}),
    options: { ...base.options, ...((extra || {}).options || {}) },
    loop_guard: { ...base.loop_guard, ...((extra || {}).loop_guard || {}) },
    web_search: { ...base.web_search, ...((extra || {}).web_search || {}) },
    agent: { ...base.agent, ...((extra || {}).agent || {}) } });
  const defaultTools = () => (state.tools || []).filter((t) => t.default).map((t) => t.name);
  const agentTools = (s) => (Array.isArray(s?.agent?.tools) ? s.agent.tools : defaultTools());

  function loadDefaults() {
    let saved = state.server?.chat_defaults;
    if (!saved || !Object.keys(saved).length) {
      try { saved = JSON.parse(localStorage.getItem("harness.defaults") || "{}"); } catch { saved = {}; }
    }
    try { return mergeSettings(FACTORY_SETTINGS, saved); } catch { return deepClone(FACTORY_SETTINGS); }
  }

  // ------------------------------------------------------------------ state
  const state = {
    models: [],
    tools: [],           // [{ name, description, approval, default }] from /api/tools
    chats: [],
    server: {},
    chat: null,          // { id, title, messages, settings, saved }
    streaming: null,     // { genId, index, startedAt, tokens, thinkStart, thinkEnd, phase }
    pendingFiles: [],    // { kind: "image", b64, name } | { kind: "text", name, type, chars, text, truncated }
    filter: "",
  };
  const ui = {
    app: $("#app"), sidebar: $("#sidebar"), panel: $("#panel"), chatList: $("#chat-list"), messages: $("#messages"), thread: $("#thread"),
    input: $("#input"), send: $("#send"), readout: $("#readout"), readoutPhase: $("#readout-phase"), readoutMetrics: $("#readout-metrics"),
    ctxMeter: $("#ctx-meter"), ctxFill: $("#ctx-fill"), ctxText: $("#ctx-text"),
    title: $("#chat-title"), topbarModel: $("#topbar-model"), conn: $("#conn"), footInfo: $("#foot-info"),
    model: $("#model"), modelCaps: $("#model-caps"), preset: $("#preset"), systemPrompt: $("#system-prompt"), think: $("#think"), thinkHelp: $("#think-help"),
    genFields: $("#gen-fields"), loopEnabled: $("#loop-enabled"), loopThreshold: $("#loop-threshold"), thinkBudget: $("#think-budget"), timeout: $("#timeout"),
    searchMode: $("#search-mode"), searchMax: $("#search-max"), searchFetch: $("#search-fetch"),
    agentTools: $("#agent-tools"), agentSteps: $("#agent-steps"), agentHelp: $("#agent-help"),
    modeSwitch: $("#mode-switch"), modeNote: $("#mode-note"), agentOptionsBtn: $("#agent-options-btn"), agentPopover: $("#agent-popover"),
    wsBtn: $("#ws-btn"), wsBtnName: $("#ws-btn-name"),
    quickThink: $("#quick-think"), quickSearch: $("#quick-search"), attachLabel: $("#attach-label"), attach: $("#attach"), attachments: $("#attachments"), attNote: $("#att-note"), composer: $("#composer"),
  };

  // ------------------------------------------------------------------ layout & theme
  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("harness.theme", theme);
    const link = $("#hljs-theme");
    if (link) link.href = `https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/${theme === "light" ? "github" : "github-dark"}.min.css`;
  }
  setTheme(localStorage.getItem("harness.theme") || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));
  $("#theme-toggle").onclick = () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");

  const narrow = () => matchMedia("(max-width: 900px)").matches;
  function applyLayout() {
    ui.app.classList.toggle("sidebar-hidden", localStorage.getItem("harness.sidebar") === "hidden" || (narrow() && localStorage.getItem("harness.sidebar") !== "shown"));
    ui.app.classList.toggle("panel-hidden", localStorage.getItem("harness.panel") === "hidden" || (narrow() && localStorage.getItem("harness.panel") !== "shown"));
  }
  const togglePane = (key, cls) => {
    const hidden = !ui.app.classList.contains(cls);
    localStorage.setItem(key, hidden ? "hidden" : "shown");
    applyLayout();
  };
  $("#toggle-sidebar").onclick = () => togglePane("harness.sidebar", "sidebar-hidden");
  $("#toggle-panel").onclick = () => togglePane("harness.panel", "panel-hidden");
  $("#close-panel").onclick = () => { localStorage.setItem("harness.panel", "hidden"); applyLayout(); };
  window.addEventListener("resize", applyLayout);
  if (narrow()) { localStorage.setItem("harness.panel", "hidden"); }
  applyLayout();

  // ------------------------------------------------------------------ health & models
  async function checkHealth() {
    try {
      const h = await api("/api/health");
      ui.conn.className = "conn " + (h.ok ? "ok" : "bad");
      ui.conn.title = h.ok ? `Ollama ${h.ollama} at ${h.host}` : `Cannot reach Ollama at ${h.host}: ${h.error}`;
      ui.footInfo.textContent = h.ok ? `ollama ${h.ollama}` : "offline";
    } catch { ui.conn.className = "conn bad"; ui.footInfo.textContent = "offline"; }
  }
  async function loadModels() {
    try {
      state.models = await api("/api/models");
    } catch (e) { toast(e.message, true); state.models = []; }
    ui.model.replaceChildren(
      ...(state.models.length ? [] : [el("option", { value: "", text: "No models found" })]),
      ...state.models.map((m) => el("option", { value: m.name, text: `${m.name}  (${m.parameter_size || "?"})` }))
    );
    if (state.chat) {
      if (!state.chat.settings.model || !state.models.some((m) => m.name === state.chat.settings.model)) {
        state.chat.settings.model = state.models[0]?.name || "";
      }
      ui.model.value = state.chat.settings.model;
      renderModelInfo();
    }
  }
  async function loadTools() {
    try {
      const r = await api("/api/tools");
      state.tools = r.tools || [];
      state.workspace = r.workspace || "";
    } catch (e) { toast(`Tool list unavailable: ${e.message}`, true); state.tools = []; }
    buildAgentTools();
    if (state.chat) settingsToPanel(state.chat.settings);
  }
  const currentModel = () => state.models.find((m) => m.name === (state.chat?.settings.model));
  function renderModelInfo() {
    const m = currentModel();
    ui.topbarModel.textContent = m ? m.name : "";
    if (!m) { ui.modelCaps.replaceChildren(); ui.thinkHelp.textContent = ""; ui.attach.accept = TEXT_ACCEPT; return; }
    const caps = m.capabilities || [];
    ui.modelCaps.replaceChildren(...[
      el("span", { class: caps.includes("thinking") ? "yes" : "", text: "thinking" }),
      el("span", { class: caps.includes("tools") ? "yes" : "", text: "tools" }),
      el("span", { class: caps.includes("vision") ? "yes" : "", text: "vision" }),
      m.quantization ? el("span", { text: m.quantization }) : null,
      m.context_length ? el("span", { text: `ctx ${m.context_length.toLocaleString()}` }) : null,
    ].filter(Boolean));
    ui.thinkHelp.textContent = caps.includes("thinking")
      ? "This model reports native thinking. Effort levels apply only to models that support them."
      : "This model does not report native thinking. If it writes <think> tags anyway, they are shown as thinking.";
    const vision = caps.includes("vision");
    ui.attach.accept = vision ? `${TEXT_ACCEPT},image/*` : TEXT_ACCEPT;
    ui.attachLabel.title = vision ? "Attach documents, code or images" : "Attach documents or code (this model cannot see images)";
    if (!vision && state.pendingFiles.some((f) => f.kind === "image")) {
      state.pendingFiles = state.pendingFiles.filter((f) => f.kind !== "image");
      renderAttachments();
      toast(`${m.name} cannot see images, so the attached pictures were removed.`);
    }
  }
  $("#refresh-models").onclick = () => loadModels().then(() => toast("Model list refreshed"));
  $("#unload-model").onclick = async () => {
    if (!state.chat?.settings.model) return;
    try { await api("/api/models/unload", { method: "POST", body: { model: state.chat.settings.model } }); toast("Model unloaded"); }
    catch (e) { toast(e.message, true); }
  };

  // ------------------------------------------------------------------ server settings dialog
  const dialog = $("#settings-dialog");
  async function loadServerSettings() {
    try { state.server = await api("/api/settings"); } catch { state.server = { system_prompt_presets: [] }; }
    renderPresets();
  }
  $("#open-settings").onclick = () => {
    $("#s-provider").value = state.server.search_provider || "ddgs";
    $("#s-searxng").value = state.server.searxng_url || "";
    $("#s-region").value = state.server.search_region || "wt-wt";
    $("#s-keepalive").value = state.server.keep_alive || "5m";
    $("#s-helper").replaceChildren(el("option", { value: "", text: "Same as the chat model" }),
      ...state.models.map((m) => el("option", { value: m.name, text: m.name })));
    $("#s-helper").value = state.models.some((m) => m.name === state.server.helper_model) ? state.server.helper_model : "";
    $("#s-workspace").value = state.server.workspace_dir || "";
    $("#s-outside").checked = !!state.server.allow_outside_workspace;
    buildPolicyRows(state.server.tool_policies || {});
    $("#s-test-result").textContent = "";
    $("#s-searxng-wrap").hidden = $("#s-provider").value !== "searxng";
    if (!$(".settings-tab.active", dialog)) showSettingsTab(localStorage.getItem("harness.settingsTab") || "models");
    dialog.showModal();
  };
  $("#s-provider").onchange = (e) => { $("#s-searxng-wrap").hidden = e.target.value !== "searxng"; };
  $("#s-cancel").onclick = () => dialog.close();
  $("#s-close").onclick = () => dialog.close();
  function showSettingsTab(name) {
    for (const t of $$(".settings-tab", dialog)) t.classList.toggle("active", t.dataset.tab === name);
    for (const p of $$(".settings-panel", dialog)) p.classList.toggle("active", p.dataset.tab === name);
    try { localStorage.setItem("harness.settingsTab", name); } catch { /* ignore */ }
  }
  for (const t of $$(".settings-tab", dialog)) t.onclick = () => showSettingsTab(t.dataset.tab);
  // Open on the tab the caller wants (or the last one used).
  function openSettings(tab) {
    $("#open-settings").click();
    showSettingsTab(tab || localStorage.getItem("harness.settingsTab") || "models");
  }
  $("#s-save").onclick = async (e) => {
    e.preventDefault();
    try {
      state.server = await api("/api/settings", { method: "PUT", body: {
        search_provider: $("#s-provider").value, searxng_url: $("#s-searxng").value.trim(),
        search_region: $("#s-region").value.trim() || "wt-wt", keep_alive: $("#s-keepalive").value.trim() || "5m",
        helper_model: $("#s-helper").value, workspace_dir: $("#s-workspace").value.trim(),
        allow_outside_workspace: $("#s-outside").checked, tool_policies: readPolicyRows(),
      } });
      dialog.close(); toast("Settings saved");
      loadTools();
    } catch (err) { toast(err.message, true); }
  };
  $("#s-test-search").onclick = async () => {
    const out = $("#s-test-result");
    out.textContent = "Searching…";
    try {
      // Save provider fields first so the test uses them.
      await api("/api/settings", { method: "PUT", body: { search_provider: $("#s-provider").value, searxng_url: $("#s-searxng").value.trim(), search_region: $("#s-region").value.trim() || "wt-wt" } });
      const r = await api("/api/search", { method: "POST", body: { query: "ollama local models", max_results: 3 } });
      out.textContent = r.length ? `Working: ${r.length} results, first from ${hostOf(r[0].url)}` : "The provider returned no results.";
    } catch (err) { out.textContent = `Failed: ${err.message}`; }
  };

  // ------------------------------------------------------------------ tool approval policies
  function buildPolicyRows(policies) {
    const box = $("#s-policies");
    const rows = [];
    for (const [sev, tools] of bySeverity(state.tools)) {
      rows.push(el("div", { class: "policy-group", text: SEVERITY_LABEL[sev] || "Other" }));
      rows.push(...tools.map(policyRow));
    }
    box.replaceChildren(...rows);
    function policyRow(t) {
      const sel = el("select", { "data-tool": t.name },
        el("option", { value: "ask", text: "Asks first" }),
        el("option", { value: "auto", text: "Runs on its own" }));
      sel.value = policies[t.name] || (t.default_approval ? "ask" : "auto");
      if (t.fixed) { sel.disabled = true; sel.title = "This tool always waits for you by nature."; }
      const risky = () => sel.value === "auto" && (t.critical || t.default_approval);   // web tools default to auto; no alarm for those
      sel.classList.toggle("auto", risky());
      sel.onchange = async () => {
        if (sel.value === "auto" && t.critical) {
          const ok = await askConfirm(`Let ${t.name} run without asking?`,
            `The model could then ${CRITICAL_VERB[t.name] || "change things"} on this computer with no chance for you to check the call first. A misfiring or heavily quantized model can do real damage this way. You can change this back at any time.`,
            "Yes, run without asking");
          if (!ok) sel.value = "ask";
        }
        sel.classList.toggle("auto", risky());
      };
      const name = el("span", { class: "policy-name", title: t.description }, t.name, TOOL_BADGE[t.name] ? el("span", { class: "tool-badge", text: TOOL_BADGE[t.name] }) : null);
      return el("div", { class: "policy-row" }, name, sel);
    }
  }
  function readPolicyRows() {
    const out = {};
    for (const sel of $$("#s-policies select:not(:disabled)")) out[sel.dataset.tool] = sel.value;
    return out;
  }

  // ------------------------------------------------------------------ presets
  // Built-in presets come from the server (read-only); the user's own live in settings.json.
  const builtinPresets = () => state.server.builtin_presets || [];
  const userPresets = () => (state.server.system_prompt_presets || []).filter((p) => !builtinPresets().some((b) => b.name === p.name && b.prompt === p.prompt));
  function renderPresets() {
    const user = userPresets(), builtin = builtinPresets();
    ui.preset.replaceChildren(
      el("option", { value: "", text: "Custom prompt" }),
      builtin.length ? el("optgroup", { label: "Built-in" }, ...builtin.map((p, i) => el("option", { value: `b${i}`, text: p.name }))) : null,
      user.length ? el("optgroup", { label: "Yours" }, ...user.map((p, i) => el("option", { value: `u${i}`, text: p.name }))) : null,
    );
    syncPresetSelection();
  }
  function presetByValue(v) {
    if (!v) return null;
    const list = v[0] === "b" ? builtinPresets() : userPresets();
    return list[Number(v.slice(1))] || null;
  }
  function syncPresetSelection() {
    const current = (state.chat?.settings.system_prompt || "").trim();
    const bi = builtinPresets().findIndex((p) => p.prompt.trim() === current);
    const ui_ = userPresets().findIndex((p) => p.prompt.trim() === current);
    ui.preset.value = ui_ >= 0 ? `u${ui_}` : bi >= 0 ? `b${bi}` : "";
    $("#delete-preset").hidden = ui_ < 0;      // built-ins cannot be deleted
  }
  ui.preset.onchange = () => {
    const p = presetByValue(ui.preset.value);
    if (p) { ui.systemPrompt.value = p.prompt; onPanelChange(); }
  };
  $("#save-preset").onclick = async () => {
    const name = prompt("Preset name");
    if (!name) return;
    const presets = (state.server.system_prompt_presets || []).filter((p) => p.name !== name);
    presets.push({ name, prompt: ui.systemPrompt.value });
    try { state.server = await api("/api/settings", { method: "PUT", body: { system_prompt_presets: presets } }); renderPresets(); toast(`Preset "${name}" saved`); }
    catch (e) { toast(e.message, true); }
  };
  $("#delete-preset").onclick = async () => {
    const target = presetByValue(ui.preset.value);
    if (!target || ui.preset.value[0] !== "u" || !confirm(`Delete preset "${target.name}"?`)) return;
    const presets = (state.server.system_prompt_presets || []).filter((p) => !(p.name === target.name && p.prompt === target.prompt));
    try { state.server = await api("/api/settings", { method: "PUT", body: { system_prompt_presets: presets } }); renderPresets(); }
    catch (e) { toast(e.message, true); }
  };

  // ------------------------------------------------------------------ controls panel binding
  function buildGenFields() {
    ui.genFields.replaceChildren(...GEN_FIELDS.map((f) => {
      if (f.plain) {
        return el("label", { class: "field inline" },
          el("span", { text: f.label, title: f.help || "" }),
          el("input", { type: "number", id: `opt-${f.key}`, placeholder: f.placeholder || "", step: "1", oninput: onPanelChange }));
      }
      const val = el("span", { class: "val" });
      const range = el("input", { type: "range", id: `opt-${f.key}`, min: f.min, max: f.max, step: f.step, oninput: (e) => { val.textContent = e.target.value; onPanelChange(); } });
      return el("label", { class: "field", title: f.help || "" }, el("span", {}, f.label, val), range);
    }));
  }
  buildGenFields();

  function settingsToPanel(s) {
    ui.model.value = s.model || "";
    ui.systemPrompt.value = s.system_prompt || "";
    ui.think.value = s.think || "default";
    for (const f of GEN_FIELDS) {
      const input = $(`#opt-${f.key}`);
      const v = s.options?.[f.key];
      input.value = v === undefined || v === null ? "" : v;
      if (!f.plain) input.previousElementSibling.querySelector(".val").textContent = input.value;
    }
    ui.loopEnabled.checked = s.loop_guard?.enabled !== false;
    ui.loopThreshold.value = s.loop_guard?.threshold ?? 3;
    ui.thinkBudget.value = s.think_budget || "";
    ui.timeout.value = s.timeout_s || "";
    ui.searchMode.value = s.web_search?.mode || "off";
    ui.searchMax.value = s.web_search?.max_results ?? 5;
    ui.searchFetch.checked = s.web_search?.fetch_pages !== false;
    ui.modeSwitch.dataset.mode = s.agent?.enabled ? "agent" : "chat";
    ui.agentSteps.value = s.agent?.max_steps ?? 8;
    const enabledTools = new Set(agentTools(s));
    for (const box of $$("input[type=checkbox]", ui.agentTools)) box.checked = enabledTools.has(box.value);
    renderModelInfo();
    syncPresetSelection();
    renderQuickChips();
  }
  const SEVERITY_LABEL = ["No side effects", "Reads the workspace", "Changes files or notes", "Runs code"];
  const bySeverity = (tools) => { const g = new Map(); for (const t of tools) { const k = t.severity ?? 0; if (!g.has(k)) g.set(k, []); g.get(k).push(t); } return [...g.entries()].sort((a, b) => a[0] - b[0]); };
  function buildAgentTools() {
    ui.agentTools.replaceChildren(...bySeverity(state.tools).map(([sev, tools]) => el("div", { class: "tool-group" },
      el("div", { class: "tool-group-head" }, el("span", { text: SEVERITY_LABEL[sev] || "Other" }),
        el("button", { type: "button", class: "tool-expand", text: "all", onclick: () => { for (const b of $$("input", ui.agentTools)) if (tools.some((t) => t.name === b.value)) b.checked = true; onPanelChange(); } }),
        el("button", { type: "button", class: "tool-expand", text: "none", onclick: () => { for (const b of $$("input", ui.agentTools)) if (tools.some((t) => t.name === b.value)) b.checked = false; onPanelChange(); } })),
      el("div", { class: "tool-grid" }, ...tools.map((t) => el("label", { class: "check tool-opt", title: `${t.description}\n\n${t.fixed ? "Always asks you." : t.approval ? "Asks first (Allow / Deny in the chat)." : "Runs on its own."}` },
        el("input", { type: "checkbox", value: t.name }),
        el("span", { class: "tool-opt-name", text: t.name }),
        t.approval ? el("span", { class: "tool-ask", text: t.fixed ? "asks you" : "asks" }) : null))))));
    ui.agentTools.append(el("p", { class: "help tool-note" },
      "All tools are on by default; anything that changes files or runs code asks you first. Untick tools only to give a small model fewer options to choose from. ",
      el("button", { type: "button", class: "tool-expand inline", text: "Approval rules are in server settings.",
        onclick: () => { closeAgentPopover(); openSettings("tools"); } })));
    for (const box of $$("input", ui.agentTools)) box.addEventListener("change", onPanelChange);
  }
  function panelToSettings() {
    const options = {};
    for (const f of GEN_FIELDS) {
      const v = $(`#opt-${f.key}`).value;
      options[f.key] = v === "" ? "" : Number(v);
    }
    return {
      model: ui.model.value,
      system_prompt: ui.systemPrompt.value,
      think: ui.think.value,
      options,
      loop_guard: { enabled: ui.loopEnabled.checked, threshold: Number(ui.loopThreshold.value) || 3 },
      think_budget: Number(ui.thinkBudget.value) || 0,
      timeout_s: Number(ui.timeout.value) || 0,
      web_search: { mode: ui.searchMode.value, max_results: Number(ui.searchMax.value) || 5, fetch_pages: ui.searchFetch.checked },
      agent: { enabled: ui.modeSwitch.dataset.mode === "agent", tools: $$("input:checked", ui.agentTools).map((b) => b.value),
        max_steps: Math.min(25, Math.max(1, Number(ui.agentSteps.value) || 8)), workspace: state.chat?.settings.agent?.workspace || "" },
    };
  }
  const persistChatSettings = debounce(async () => {
    if (!state.chat || !state.chat.saved) return;
    try { await api(`/api/chats/${state.chat.id}`, { method: "PUT", body: { settings: state.chat.settings } }); }
    catch (e) { toast(e.message, true); }
  }, 600);
  function onPanelChange() {
    if (!state.chat) return;
    state.chat.settings = panelToSettings();
    renderModelInfo();
    syncPresetSelection();
    renderQuickChips();
    updateContextMeter();
    persistChatSettings();
  }
  for (const node of [ui.model, ui.systemPrompt, ui.think, ui.loopEnabled, ui.loopThreshold, ui.thinkBudget, ui.timeout, ui.searchMode, ui.searchMax, ui.searchFetch, ui.agentSteps]) {
    node.addEventListener("input", onPanelChange);
    node.addEventListener("change", onPanelChange);
  }
  $("#save-defaults").onclick = async () => {
    const defaults = panelToSettings();
    try {
      state.server = await api("/api/settings", { method: "PUT", body: { chat_defaults: defaults } });
      localStorage.removeItem("harness.defaults");
      toast("New chats will start with these settings");
    } catch (e) {
      // Server unreachable: keep them at least for this browser.
      localStorage.setItem("harness.defaults", JSON.stringify(defaults));
      toast(`Saved in this browser only (server: ${e.message})`, true);
    }
  };
  $("#reset-settings").onclick = () => {
    const d = deepClone(FACTORY_SETTINGS);
    d.model = state.chat?.settings.model || state.models[0]?.name || "";
    settingsToPanel(d);
    onPanelChange();
  };

  function renderQuickChips() {
    const s = state.chat?.settings;
    if (!s) return;
    const m = currentModel();
    const think = s.think || "default";
    ui.quickThink.textContent = think === "default" ? "Thinking: default" : think === "off" ? "Thinking: off" : think === "on" ? "Thinking: on" : `Thinking: ${think}`;
    ui.quickThink.className = "chip" + (think !== "default" && think !== "off" ? " on" : "") + (m && !(m.capabilities || []).includes("thinking") ? " muted" : "");
    const mode = s.web_search?.mode || "off";
    ui.quickSearch.textContent = mode === "off" ? "Web search: off" : mode === "auto" ? "Web search: auto" : "Web search: always";
    ui.quickSearch.className = "chip" + (mode !== "off" ? " on" : "");
    renderModeBar();
  }

  // ------------------------------------------------------------------ chat / agent mode
  function renderModeBar() {
    const s = state.chat?.settings;
    if (!s) return;
    const m = currentModel();
    const agent = !!s.agent?.enabled;
    const modelHasTools = !m || (m.capabilities || []).includes("tools");
    ui.modeSwitch.dataset.mode = agent ? "agent" : "chat";
    for (const b of $$(".mode-btn", ui.modeSwitch)) {
      const on = b.dataset.mode === (agent ? "agent" : "chat");
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    }
    ui.modeSwitch.classList.toggle("unsupported", agent && !modelHasTools);
    ui.agentOptionsBtn.hidden = !agent;
    const tools = agentTools(s);
    const asking = tools.filter((n) => state.tools.find((t) => t.name === n)?.approval).length;
    ui.modeNote.textContent = !agent ? ""
      : !modelHasTools ? `${m.name} does not report tool support; replies stay in chat mode`
      : !tools.length ? "No tools selected"
      : `${tools.length} tool${tools.length === 1 ? "" : "s"}${asking ? `, ${asking} ask${asking === 1 ? "s" : ""} before running` : ""}`;
    renderWorkspaceButton(agent);
    // The popover explains itself through the tags and the settings link; the help line only appears for a model without tools.
    ui.agentHelp.hidden = modelHasTools;
    ui.agentHelp.textContent = modelHasTools ? "" : `${m.name} does not report tool support; agent mode has no effect until you pick a model that does.`;
    if (!agent) closeAgentPopover();
  }
  const folderName = (p) => { const parts = String(p).replace(/[\\/]+$/, "").split(/[\\/]/); return parts[parts.length - 1] || p; };
  function renderWorkspaceButton(agent) {
    const ws = state.chat?.settings.agent?.workspace || "";
    ui.wsBtn.hidden = !agent;
    ui.wsBtnName.textContent = ws ? folderName(ws) : `default${state.workspace ? ` (${folderName(state.workspace)})` : ""}`;
    ui.wsBtn.classList.toggle("custom", !!ws);
    ui.wsBtn.title = (ws || state.workspace || "") + "\nFolder the agent's file tools and scripts work in. Click to change.";
  }
  function setWorkspace(path) {
    if (!state.chat) return;
    state.chat.settings.agent = { ...(state.chat.settings.agent || {}), workspace: path || "" };
    onPanelChange();
  }
  ui.wsBtn.onclick = () => openFolderDialog(state.chat?.settings.agent?.workspace || "");

  // ------------------------------------------------------------------ folder picker
  const fd = { dialog: $("#folder-dialog"), path: $("#fd-path"), list: $("#fd-list"), shortcuts: $("#fd-shortcuts"), recent: $("#fd-recent"), recentHead: $("#fd-recent-head"), status: $("#fd-status"), up: $("#fd-up"), use: $("#fd-use") };
  let fdCurrent = null;   // last listing from the server
  async function fdLoad(path) {
    fd.status.textContent = "Loading…";
    try {
      const r = await api(`/api/fs/dirs?path=${encodeURIComponent(path || "")}`);
      fdCurrent = r;
      fd.path.value = r.path;
      requestAnimationFrame(() => { fd.path.scrollLeft = fd.path.scrollWidth; });   // long paths: show the folder name end
      fd.up.disabled = !r.parent;
      fd.list.replaceChildren(...(r.dirs.length ? r.dirs.map((d) => el("button", { type: "button", class: "folder-item", text: d.name, title: d.path, onclick: () => fdLoad(d.path) }))
        : [el("div", { class: "folder-empty", text: "No subfolders" })]));
      const link = (label, p, active) => el("button", { type: "button", class: "folder-link" + (active ? " active" : ""), text: label, title: p, onclick: () => fdLoad(p) });
      fd.shortcuts.replaceChildren(...r.shortcuts.map((s) => link(s.label, s.path, s.path === r.path)));
      fd.recentHead.hidden = !r.recent.length;
      fd.recent.replaceChildren(...r.recent.map((p) => link(folderName(p), p, p === r.path)));
      fd.status.textContent = r.is_default ? "This is the default workspace from server settings." : `${r.dirs.length} subfolder${r.dirs.length === 1 ? "" : "s"}`;
    } catch (e) {
      fd.status.textContent = `Cannot open that folder: ${e.message}`;
    }
  }
  function openFolderDialog(start) {
    fdLoad(start || "");
    fd.dialog.showModal();
    setTimeout(() => fd.path.focus(), 50);
  }
  fd.up.onclick = () => { if (fdCurrent?.parent) fdLoad(fdCurrent.parent); };
  fd.path.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); fdLoad(fd.path.value.trim()); } });
  fd.use.onclick = () => {
    if (!fdCurrent) return;
    // Choosing the server default folder means "use the default", so the chat follows later changes of it.
    setWorkspace(fdCurrent.is_default ? "" : fdCurrent.path);
    fd.dialog.close();
    toast(fdCurrent.is_default ? "Using the default workspace" : `Workspace: ${fdCurrent.path}`);
  };
  $("#fd-cancel").onclick = () => fd.dialog.close();
  $("#fd-default").onclick = () => { setWorkspace(""); fd.dialog.close(); toast("Using the default workspace"); };
  $("#fd-close").onclick = () => fd.dialog.close();

  function setMode(mode) {
    ui.modeSwitch.dataset.mode = mode;
    onPanelChange();
  }
  for (const b of $$(".mode-btn", ui.modeSwitch)) {
    b.onclick = () => {
      if (b.dataset.mode === "agent" && ui.modeSwitch.dataset.mode === "agent") { toggleAgentPopover(); return; }
      setMode(b.dataset.mode);
    };
  }
  function openAgentPopover() { ui.agentPopover.hidden = false; ui.agentOptionsBtn.setAttribute("aria-expanded", "true"); }
  function closeAgentPopover() { ui.agentPopover.hidden = true; ui.agentOptionsBtn.setAttribute("aria-expanded", "false"); }
  function toggleAgentPopover() { if (ui.agentPopover.hidden) openAgentPopover(); else closeAgentPopover(); }
  ui.agentOptionsBtn.onclick = toggleAgentPopover;
  $("#agent-popover-close").onclick = closeAgentPopover;
  document.addEventListener("pointerdown", (e) => {
    if (ui.agentPopover.hidden) return;
    if (ui.agentPopover.contains(e.target) || ui.agentOptionsBtn.contains(e.target) || ui.modeSwitch.contains(e.target)) return;
    if (e.target.closest && e.target.closest("dialog[open]")) return;   // the folder picker and other dialogs belong to the popover's flow
    closeAgentPopover();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !ui.agentPopover.hidden) closeAgentPopover(); });
  ui.quickThink.onclick = () => {
    const order = ["default", "off", "on"];
    ui.think.value = order[(order.indexOf(ui.think.value) + 1) % order.length] || "default";
    onPanelChange();
  };
  ui.quickSearch.onclick = () => {
    const order = ["off", "auto", "always"];
    ui.searchMode.value = order[(order.indexOf(ui.searchMode.value) + 1) % order.length];
    onPanelChange();
  };

  // ------------------------------------------------------------------ chats
  async function refreshChatList() {
    try { state.chats = await api("/api/chats"); } catch (e) { toast(e.message, true); }
    renderChatList();
  }
  function groupLabel(iso) {
    const d = new Date(iso), now = new Date();
    const start = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const days = Math.round((start(now) - start(d)) / 86400000);
    if (days <= 0) return "Today";
    if (days === 1) return "Yesterday";
    if (days < 7) return "This week";
    if (days < 30) return "This month";
    return "Older";
  }
  function renderChatList() {
    const q = state.filter.trim().toLowerCase();
    const items = state.chats.filter((c) => !q || (c.title || "").toLowerCase().includes(q) || (c.model || "").toLowerCase().includes(q));
    const nodes = [];
    let lastGroup = null;
    for (const c of items) {
      const group = c.pinned ? "Pinned" : groupLabel(c.updated_at);
      if (group !== lastGroup) { nodes.push(el("div", { class: "chat-group", text: group })); lastGroup = group; }
      nodes.push(el("div", { class: "chat-item" + (state.chat?.id === c.id ? " active" : ""), role: "button", tabindex: "0",
        onclick: () => openChat(c.id), onkeydown: (e) => { if (e.key === "Enter") openChat(c.id); } },
        c.pinned ? el("span", { class: "ci-pin", text: "●" }) : null,
        el("span", { class: "ci-title" + (c.title ? "" : " untitled"), text: c.title || "Untitled chat", title: c.model || "" }),
        el("span", { class: "ci-actions" },
          el("button", { title: c.pinned ? "Unpin" : "Pin", text: c.pinned ? "Unpin" : "Pin", onclick: (e) => { e.stopPropagation(); togglePin(c); } }),
          el("button", { title: "Delete", text: "Delete", onclick: (e) => { e.stopPropagation(); deleteChat(c); } }))));
    }
    if (!nodes.length) nodes.push(el("div", { class: "empty-list", text: q ? "No chats match." : "Your saved chats will appear here." }));
    ui.chatList.replaceChildren(...nodes);
  }
  $("#chat-filter").oninput = (e) => { state.filter = e.target.value; renderChatList(); };

  async function togglePin(c) {
    try { await api(`/api/chats/${c.id}`, { method: "PUT", body: { pinned: !c.pinned } }); await refreshChatList(); }
    catch (e) { toast(e.message, true); }
  }
  async function deleteChat(c) {
    if (!confirm(`Delete "${c.title || "this chat"}"? This cannot be undone.`)) return;
    try {
      await api(`/api/chats/${c.id}`, { method: "DELETE" });
      if (state.chat?.id === c.id) newChat();
      await refreshChatList();
    } catch (e) { toast(e.message, true); }
  }

  function newChat() {
    if (state.streaming) stopGeneration();
    const settings = loadDefaults();
    if (!settings.model || !state.models.some((m) => m.name === settings.model)) settings.model = state.models[0]?.name || "";
    state.chat = { id: uid(), title: "", messages: [], settings, saved: false };
    localStorage.removeItem("harness.lastChat");
    ui.title.value = "";
    settingsToPanel(settings);
    renderMessages();
    renderChatList();
    ui.input.focus();
  }
  async function openChat(id) {
    if (state.streaming) stopGeneration();
    try {
      const chat = await api(`/api/chats/${id}`);
      state.chat = { ...chat, settings: mergeSettings(loadDefaults(), chat.settings), saved: true };
      if (!state.models.some((m) => m.name === state.chat.settings.model)) state.chat.settings.model = state.models[0]?.name || "";
      localStorage.setItem("harness.lastChat", id);
      ui.title.value = chat.title || "";
      settingsToPanel(state.chat.settings);
      renderMessages();
      renderChatList();
      if (narrow()) { localStorage.setItem("harness.sidebar", "hidden"); applyLayout(); }
    } catch (e) { toast(e.message, true); }
  }
  // The server finishes the auto-title in the background even if the stream was cut off;
  // if it did not arrive over the stream, fetch it a little later.
  function pickUpTitleLater(chat, delays = [12000, 30000]) {
    const [delay, ...rest] = delays;
    if (delay == null) return;
    setTimeout(async () => {
      if (chat.title) return;
      try {
        const fresh = await api(`/api/chats/${chat.id}`);
        if (fresh.title) {
          chat.title = fresh.title;
          if (state.chat?.id === chat.id) { state.chat.title = fresh.title; ui.title.value = fresh.title; }
          refreshChatList();
          return;
        }
      } catch { /* chat may have been deleted */ return; }
      pickUpTitleLater(chat, rest);
    }, delay);
  }
  async function saveChatMessages() {
    if (!state.chat) return;
    try {
      await api(`/api/chats/${state.chat.id}`, { method: "PUT", body: { title: state.chat.title, messages: state.chat.messages, settings: state.chat.settings } });
      state.chat.saved = true;
      refreshChatList();
    } catch (e) { toast(e.message, true); }
  }
  $("#new-chat").onclick = newChat;
  ui.title.addEventListener("change", async () => {
    if (!state.chat) return;
    state.chat.title = ui.title.value.trim();
    if (state.chat.saved) { await api(`/api/chats/${state.chat.id}`, { method: "PUT", body: { title: state.chat.title } }).catch((e) => toast(e.message, true)); refreshChatList(); }
  });
  ui.title.addEventListener("keydown", (e) => { if (e.key === "Enter") ui.title.blur(); });

  // export
  const exportMenu = $("#export-menu");
  $("#export-btn").onclick = (e) => { e.stopPropagation(); exportMenu.hidden = !exportMenu.hidden; };
  document.addEventListener("click", () => { exportMenu.hidden = true; });
  for (const b of $$("button", exportMenu)) b.onclick = () => exportChat(b.dataset.format);
  function exportChat(format) {
    if (!state.chat || !state.chat.messages.length) return toast("Nothing to export yet");
    const name = (state.chat.title || "chat").replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "-") || "chat";
    let blob;
    if (format === "json") {
      blob = new Blob([JSON.stringify({ ...state.chat, saved: undefined }, null, 2)], { type: "application/json" });
    } else {
      const lines = [`# ${state.chat.title || "Chat"}`, "", `Model: ${state.chat.settings.model}`, ""];
      if (state.chat.settings.system_prompt) lines.push("## System prompt", "", state.chat.settings.system_prompt, "");
      for (const m of state.chat.messages) {
        lines.push(`## ${m.role === "user" ? "User" : "Assistant"}`, "");
        if (m.attachments?.length) lines.push(`Attached: ${m.attachments.map((a) => `${a.name} (${a.chars} chars)`).join(", ")}`, "");
        if (m.thinking) lines.push("<details><summary>Thinking</summary>", "", m.thinking, "", "</details>", "");
        for (const c of m.tool_calls || []) {
          lines.push(`<details><summary>Tool: ${c.name} (${c.status})</summary>`, "", ...(c.purpose ? [`Intent: ${c.purpose}`, ""] : []), "```json", JSON.stringify(c.arguments || {}, null, 2), "```", "");
          if (c.result) lines.push("```", c.result, "```", "");
          lines.push("</details>", "");
        }
        lines.push(m.content, "");
        if (m.sources?.length) { lines.push("Sources:", ""); for (const s of m.sources) lines.push(`${s.n}. [${s.title}](${s.url})`); lines.push(""); }
      }
      blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    }
    const a = el("a", { href: URL.createObjectURL(blob), download: `${name}.${format}` });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  // ------------------------------------------------------------------ messages
  const STARTERS = [
    "Explain how transformers use attention, briefly.",
    "Write a PowerShell one-liner that lists the ten largest files in a folder.",
    "What changed in the latest Ollama release?",
    "Help me plan a three-day trip to Lisbon.",
  ];
  function renderMessages() {
    updateContextMeter();
    const msgs = state.chat?.messages || [];
    if (!msgs.length) {
      ui.messages.replaceChildren(el("div", { class: "welcome" },
        el("h1", { text: "What are we working on?" }),
        el("p", { text: "Everything runs on your machine through Ollama. Pick a model and tune it in the controls; chats are saved automatically after the first reply." }),
        el("div", { class: "starters" }, ...STARTERS.map((s) => el("button", { type: "button", text: s, onclick: () => { ui.input.value = s; autoGrow(); ui.input.focus(); } })))));
      return;
    }
    ui.messages.replaceChildren(...msgs.map((m, i) => renderMessage(m, i)));
    scrollToBottom(true);
  }
  function renderMessage(m, index) {
    const node = el("article", { class: `msg ${m.role}`, "data-index": index });
    const who = m.role === "user" ? "You" : (m.model || state.chat.settings.model || "Assistant");
    node.append(el("div", { class: "msg-head" }, el("span", { class: "who", text: who }), el("span", { class: "when", text: fmtTime(m.created_at) })));
    if (m.role === "user") {
      if (m.attachments?.length) {
        node.append(el("div", { class: "files" }, ...m.attachments.map((a) => el("details", {},
          el("summary", { title: a.type || "" }, el("span", { class: "att-icon", text: fileTag(a) }), el("span", { text: a.name }),
            el("span", { class: "att-size", text: fmtChars(a.chars) })),
          el("pre", { text: a.text || "(no text)" })))));
      }
      const bubble = el("div", { class: "bubble" });
      for (const img of m.images || []) bubble.append(el("img", { src: `data:image/*;base64,${img}`, alt: "attached image" }));
      if (m.content) bubble.append(document.createTextNode(m.content));
      if (!m.content && !(m.images || []).length) bubble.append(el("em", { text: "(files only)" }));
      node.append(bubble);
    } else {
      node.append(renderThinking(m, false));
      if (m.tool_calls?.length) node.append(renderTools(m.tool_calls, null));
      const body = el("div", { class: "body", html: renderMarkdown(m.content || "", m.sources) });
      decorateCode(body);
      node.append(body);
      if (m.sources?.length) node.append(renderSources(m.search_query, m.sources));
    }
    node.append(renderFooter(m, index));
    return node;
  }
  function renderThinking(m, live) {
    if (!m.thinking && !live) return document.createComment("no thinking");
    const secs = m.thinking_seconds;
    const label = live ? "Thinking…" : secs ? `Thought for ${fmtSecs(secs)}` : "Thinking";
    const d = el("details", { class: "think" + (live ? " live" : ""), open: live || null },
      el("summary", { text: label }),
      el("div", { class: "think-body", text: m.thinking || "" }));
    return d;
  }
  // Tool activity block. `live` is the gen id while streaming (enables Allow / Deny), null for saved messages.
  // What a tool call is about, for the header: the path, query or URL rather than a blob of content.
  const str = (v) => (typeof v === "string" ? v.trim() : v == null ? "" : String(v));
  const lineCount = (s) => (s ? s.replace(/\r?\n$/, "").split(/\r?\n/).length : 0);
  const shortPath = (p) => { const s = str(p).replace(/\\/g, "/"); return s.length > 60 ? "…" + s.slice(-57) : s; };
  function toolHeadline(rec) {
    const a = rec.arguments || {};
    switch (rec.name) {
      case "write_file": case "read_file": return shortPath(a.path);
      case "list_files": return shortPath(a.path) || "workspace root";
      case "edit_file": return shortPath(a.path);
      case "search_files": return `"${str(a.pattern)}"${a.path ? ` in ${shortPath(a.path)}` : ""}${a.glob ? ` (${str(a.glob)})` : ""}`;
      case "get_datetime": return "";
      case "calculate": return str(a.expression).slice(0, 60);
      case "read_attachment": return str(a.name) || "list attachments";
      case "recall_chats": return `"${str(a.query)}"`;
      case "ask_user": return "";   // the question is shown in full in the card body
      case "remember": return str(a.note).slice(0, 60);
      case "delete_file": return shortPath(a.path);
      case "move_file": return `${shortPath(a.source)} → ${shortPath(a.destination)}`;
      case "run_shell": return str(a.command).replace(/\s+/g, " ").slice(0, 60);
      case "open_path": return str(a.target).slice(0, 60);
      case "web_search": return str(a.query);
      case "fetch_page": return str(a.url);
      case "run_python": return `${lineCount(str(a.code))} lines of Python`;
      default: { const f = Object.values(a).find((v) => typeof v === "string" && v.trim()); return f ? f.replace(/\s+/g, " ").slice(0, 60) : ""; }
    }
  }
  // One specific sentence for the approval prompt.
  function toolDescription(rec) {
    const a = rec.arguments || {};
    const where = state.server.allow_outside_workspace ? "" : " in the workspace";
    switch (rec.name) {
      case "write_file": { const c = str(a.content); return `Create or overwrite ${str(a.path) || "a file"}${where} with ${lineCount(c)} line${lineCount(c) === 1 ? "" : "s"} (${c.length.toLocaleString()} characters).`; }
      case "read_file": return `Read the file ${str(a.path) || "?"}${where}.`;
      case "list_files": return a.path && str(a.path) !== "." ? `List the folder ${str(a.path)}${where}.` : "List the workspace folder.";
      case "run_python": return `Run a ${lineCount(str(a.code))}-line Python script with the workspace as working directory.`;
      case "edit_file": return `Replace ${lineCount(str(a.old_text))} line${lineCount(str(a.old_text)) === 1 ? "" : "s"} with ${lineCount(str(a.new_text))} in ${str(a.path) || "?"}${a.replace_all ? ", every occurrence" : ""}${where}.`;
      case "search_files": return `Search ${a.path ? str(a.path) : "the workspace"} for "${str(a.pattern)}"${a.regex ? " as a regular expression" : ""}.`;
      case "get_datetime": return "Read the current date and time.";
      case "calculate": return `Evaluate ${str(a.expression)}.`;
      case "read_attachment": return a.name ? `Read the attached file ${str(a.name)}.` : "List the files attached to this chat.";
      case "recall_chats": return `Search your saved chats for "${str(a.query)}".`;
      case "ask_user": return str(a.question);
      case "remember": return `Add a note to the persistent notes file: “${str(a.note)}”.`;
      case "delete_file": return `Delete ${str(a.path) || "?"}${where}. This cannot be undone.`;
      case "move_file": return `Move ${str(a.source) || "?"} to ${str(a.destination) || "?"}${where}${a.overwrite ? ", replacing what is there" : ""}.`;
      case "run_shell": return `Run a shell command with the workspace as working directory.`;
      case "open_path": return `Open ${str(a.target)} on your screen with its default application.`;
      case "web_search": return `Search the web for “${str(a.query)}”.`;
      case "fetch_page": return `Download ${str(a.url)}.`;
      default: return `Run the tool ${rec.name}.`;
    }
  }
  // Arguments as labelled fields; long or multi-line text gets a block with real line breaks.
  const DIFF_CLASS = { old_text: "diff-old", new_text: "diff-new" };
  const ARG_ORDER = ["path", "source", "destination", "target", "name", "url", "query", "pattern", "glob", "regex", "case_sensitive", "max_results", "overwrite", "replace_all", "expression", "question", "options", "note", "old_text", "new_text", "content", "code", "command"];
  function renderArgs(args) {
    const isBlock = (k, text) => !!DIFF_CLASS[k] || /\n/.test(text) || text.length > 90;
    const entries = Object.entries(args || {}).map(([k, v]) => [k, typeof v === "string" ? v : JSON.stringify(v, null, 2)]);
    if (!entries.length) return el("div", { class: "tool-args-empty", text: "No arguments" });
    // Short fields first, long text last; within that, a sensible fixed order (path before old before new).
    const rank = (k) => { const i = ARG_ORDER.indexOf(k); return i < 0 ? ARG_ORDER.length : i; };
    entries.sort((x, y) => (Number(isBlock(...x)) - Number(isBlock(...y))) || (rank(x[0]) - rank(y[0])));
    return el("div", { class: "tool-args" }, ...entries.map(([k, text]) => {
      const block = isBlock(k, text);
      let body;
      if (block || DIFF_CLASS[k]) {
        const long = lineCount(text) > 8;
        body = el("pre", { class: (long ? "clamped " : "") + (DIFF_CLASS[k] || ""), text: text || (DIFF_CLASS[k] ? "(nothing: the text is removed)" : "") });
        const row = el("div", { class: "tool-arg-row block" }, el("span", { class: "tool-arg-key", text: k }), body);
        if (long) {
          const btn = el("button", { type: "button", class: "tool-expand", text: `Show all ${lineCount(text)} lines` });
          btn.onclick = () => { const open = body.classList.toggle("clamped"); btn.textContent = open ? `Show all ${lineCount(text)} lines` : "Show less"; };
          row.append(btn);
        }
        return row;
      }
      return el("div", { class: "tool-arg-row" }, el("span", { class: "tool-arg-key", text: k }), el("span", { class: "tool-arg-val", text }));
    }));
  }
  const STATUS_LABEL = { pending: "", running: "running…", ok: "done", error: "failed", denied: "denied", stopped: "stopped" };
  const statusLabel = (rec) => rec.name === "ask_user" ? ({ ok: "answered", denied: "skipped" }[rec.status] ?? STATUS_LABEL[rec.status] ?? rec.status) : (STATUS_LABEL[rec.status] ?? rec.status);
  const TOOL_BADGE = { write_file: "writes files", edit_file: "writes files", delete_file: "deletes files", move_file: "moves files", run_python: "runs code", run_shell: "runs commands" };
  const CRITICAL_VERB = { run_python: "execute arbitrary Python code", run_shell: "run arbitrary shell commands", write_file: "create or overwrite files", edit_file: "modify files", delete_file: "delete files", move_file: "move or rename files" };
  function renderToolCall(rec, live) {
    const details = el("details", { class: `tool-call ${rec.status}`, "data-call": rec.id, open: rec.status === "pending" || null });
    const summary = el("summary", {},
      el("span", { class: "tool-name", text: rec.name }),
      TOOL_BADGE[rec.name] ? el("span", { class: "tool-badge", text: TOOL_BADGE[rec.name], title: "This tool can change files on disk" }) : null,
      el("span", { class: "tool-arg", text: toolHeadline(rec), title: toolHeadline(rec) }),
      el("span", { class: `tool-status ${rec.status}`, text: statusLabel(rec) }));
    details.append(summary);
    // One sentence about the call: the model's intent, or the factual description when it gave none.
    const intent = el("div", { class: "tool-intent" }, el("b", { text: "Intent: " }), rec.purpose || toolDescription(rec));
    if (rec.name === "ask_user" && rec.status === "pending" && live) {
      const reply = async (text) => {
        for (const b of $$("button, input", details)) b.disabled = true;
        try { await api(`/api/generate/${live}/tools/${rec.id}`, { method: "POST", body: { approved: !!text, answer: text || "" } }); }
        catch (e) { toast(e.message, true); }
      };
      const input = el("input", { type: "text", class: "ask-input", placeholder: "Type your answer", "aria-label": "Your answer" });
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); if (input.value.trim()) reply(input.value.trim()); } });
      const options = Array.isArray(rec.arguments?.options) ? rec.arguments.options.slice(0, 5).map(String) : [];
      details.append(el("div", { class: "tool-question" },
        el("div", { class: "tool-intent" }, el("b", { text: "Question: " }), str(rec.arguments?.question)),
        options.length ? el("div", { class: "ask-options" }, ...options.map((o) => el("button", { type: "button", class: "btn ghost small", text: o, onclick: () => reply(o) }))) : null,
        el("div", { class: "ask-row" }, input,
          el("button", { type: "button", class: "btn primary small", text: "Answer", onclick: () => { if (input.value.trim()) reply(input.value.trim()); } }),
          el("button", { type: "button", class: "btn ghost small", text: "Skip", onclick: () => reply("") }))));
      setTimeout(() => input.focus(), 50);
    } else if (rec.approval && rec.status === "pending" && live) {
      const answer = async (approved) => {
        for (const b of $$("button", details)) b.disabled = true;
        try { await api(`/api/generate/${live}/tools/${rec.id}`, { method: "POST", body: { approved } }); }
        catch (e) { toast(e.message, true); }
      };
      details.append(el("div", { class: "tool-ask" }, intent,
        el("div", { class: "tool-actions" },
          el("button", { class: "btn ghost small", text: "Deny", onclick: () => answer(false) }),
          el("button", { class: "btn primary small", text: "Allow", onclick: () => answer(true) }))));
    } else if (rec.purpose) {
      details.append(intent);
    }
    if (rec.name !== "ask_user") details.append(renderArgs(rec.arguments));
    if (rec.result) {
      const r = rec.result.length > 4000 ? rec.result.slice(0, 4000) + `\n… (${rec.result.length.toLocaleString()} characters total)` : rec.result;
      details.append(el("div", { class: "tool-section", text: "Result" + (rec.seconds ? ` (${fmtSecs(rec.seconds)})` : "") }), el("pre", { text: r }));
    }
    return details;
  }
  function renderTools(calls, live) {
    const pending = calls.filter((c) => c.status === "pending").length;
    const wrap = el("details", { class: "tools" + (pending ? " needs-answer" : ""), open: live || pending ? true : null },
      el("summary", { text: pending ? (calls.some((c) => c.status === "pending" && c.name === "ask_user") ? "The model has a question for you" : "Tool call waiting for your approval") : `${calls.length} tool call${calls.length === 1 ? "" : "s"}` }),
      el("div", { class: "tools-body" }, ...calls.map((c) => renderToolCall(c, live))));
    return wrap;
  }
  function renderSources(query, sources) {
    return el("div", { class: "sources" },
      el("div", { class: "sq" }, "Searched for ", el("b", { text: query || "" })),
      el("ol", {}, ...sources.map((s) => el("li", {},
        el("span", { class: "n", text: `[${s.n}]` }),
        el("a", { href: s.url, target: "_blank", rel: "noopener", text: s.title || s.url, title: s.snippet || "" }),
        el("span", { class: "host", text: hostOf(s.url) })))));
  }
  function statsText(m) {
    const st = m.stats || {};
    const parts = [];
    if (st.tokens) parts.push(`${st.tokens} tok`);
    if (st.tokens_per_second) parts.push(`${st.tokens_per_second} tok/s`);
    if (st.total_seconds) parts.push(fmtSecs(st.total_seconds));
    if (st.prompt_tokens) parts.push(`${st.prompt_tokens} in`);
    return parts.join("   ");
  }
  // ------------------------------------------------------------------ context meter
  // Ollama reports the prompt size (prompt_eval_count) with every reply, so after a reply the number is
  // measured; everything typed or attached since then is estimated at 3.2 chars per token.
  const CHARS_PER_TOKEN = 3.2;
  const estTokens = (chars) => Math.round(chars / CHARS_PER_TOKEN);
  const fmtK = (n) => n >= 10000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
  function messageChars(m) {
    let chars = (m.content || "").length;
    for (const a of m.attachments || []) chars += Math.min(a.chars || 0, (a.text || "").length);
    return chars;
  }
  function contextUsage() {
    const chat = state.chat; if (!chat) return null;
    const limit = Number(chat.settings.options?.num_ctx) || 4096;
    const msgs = chat.messages || [];
    let base = -1;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "assistant" && msgs[i].stats?.prompt_tokens) { base = i; break; }
    }
    let used = 0;
    if (base >= 0) used = msgs[base].stats.prompt_tokens + (msgs[base].stats.tokens || 0);
    else used = estTokens((chat.settings.system_prompt || "").length);
    for (let i = base + 1; i < msgs.length; i++) used += estTokens(messageChars(msgs[i]));
    used += estTokens(ui.input.value.length);
    used += estTokens(state.pendingFiles.reduce((n, f) => n + (f.kind === "text" ? Math.min(f.chars, f.text.length) : 0), 0));
    return { used, limit, measured: base >= 0 };
  }
  function updateContextMeter() {
    const u = contextUsage();
    if (!u || (!u.measured && u.used === 0)) { ui.ctxMeter.hidden = true; return; }
    const pct = u.used / u.limit;
    ui.ctxMeter.hidden = false;
    ui.ctxMeter.className = "ctx-meter" + (pct >= 1 ? " over" : pct >= 0.8 ? " warn" : "") + (u.measured ? "" : " estimate");
    ui.ctxFill.style.width = `${Math.min(100, pct * 100).toFixed(1)}%`;
    ui.ctxText.textContent = `${fmtK(u.used)} / ${fmtK(u.limit)}`;
    const how = u.measured ? "Measured after the last reply, plus an estimate for what was typed or attached since." : "Estimated from the text length; the first reply gives a measured value.";
    const over = pct >= 1 ? " The oldest part of the conversation will be dropped by the model." : "";
    ui.ctxMeter.title = `Context window: about ${u.used.toLocaleString()} of ${u.limit.toLocaleString()} tokens (${Math.round(pct * 100)}%). ${how}${over} Change the limit with "Context length" in the controls.`;
  }

  function renderFooter(m, index) {
    const foot = el("div", { class: "msg-foot" });
    const actions = el("span", { class: "msg-actions" });
    actions.append(el("button", { text: "Copy", onclick: () => { navigator.clipboard.writeText(m.content || ""); toast("Copied"); } }));
    if (m.role === "user") {
      actions.append(el("button", { text: "Edit", onclick: () => editMessage(index) }));
    } else {
      actions.append(el("button", { text: "Regenerate", onclick: () => regenerate(index) }));
    }
    actions.append(el("button", { text: "Delete", onclick: () => deleteMessage(index) }));
    if (m.role === "assistant") {
      const s = statsText(m);
      if (s) foot.append(el("span", { class: "stats", text: s }));
      if (m.aborted && m.aborted !== "error") foot.append(el("span", { class: "aborted", text: `Stopped: ${m.aborted}` }));
      else if (m.stats?.done_reason === "length") foot.append(el("span", { class: "aborted", text: "Reached the max tokens limit" }));
      if (m.error) foot.append(el("span", { class: "error-note", text: m.error }));
    }
    foot.append(actions);
    return foot;
  }
  function scrollToBottom(force = false) {
    const t = ui.thread;
    const nearBottom = t.scrollHeight - t.scrollTop - t.clientHeight < 140;
    if (force || nearBottom) t.scrollTop = t.scrollHeight;
  }

  function editMessage(index) {
    const m = state.chat.messages[index];
    const node = $(`.msg[data-index="${index}"]`, ui.messages);
    if (!node) return;
    const ta = el("textarea", { rows: 4 }); ta.value = m.content;
    const box = el("div", { class: "edit-box" }, ta, el("div", { class: "row-actions" },
      el("button", { class: "btn ghost small", text: "Cancel", onclick: () => renderMessages() }),
      el("button", { class: "btn primary small", text: "Save and resend", onclick: () => {
        m.content = ta.value.trim();
        state.chat.messages = state.chat.messages.slice(0, index + 1);
        renderMessages();
        generate();
      } })));
    node.querySelector(".bubble").replaceWith(box);
    ta.focus();
  }
  function regenerate(index) {
    if (state.streaming) return;
    state.chat.messages = state.chat.messages.slice(0, index);
    if (!state.chat.messages.length || state.chat.messages.at(-1).role !== "user") return toast("Nothing to regenerate from");
    renderMessages();
    generate();
  }
  async function deleteMessage(index) {
    if (state.streaming) return;
    state.chat.messages.splice(index, 1);
    renderMessages();
    await saveChatMessages();
  }

  // ------------------------------------------------------------------ composer
  function autoGrow() {
    ui.input.style.height = "auto";
    ui.input.style.height = Math.min(ui.input.scrollHeight, 260) + "px";
  }
  ui.input.addEventListener("input", () => { autoGrow(); updateContextMeter(); });
  ui.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
    if (e.key === "Escape" && state.streaming) stopGeneration();
  });
  ui.send.onclick = () => (state.streaming ? stopGeneration() : send());

  const TEXT_ACCEPT = ".txt,.md,.markdown,.rst,.csv,.tsv,.json,.jsonl,.xml,.yaml,.yml,.html,.htm,.log,.ini,.toml,.cfg,.conf,.tex,.srt,"
    + ".py,.js,.mjs,.ts,.tsx,.jsx,.java,.kt,.cs,.cpp,.cc,.c,.h,.hpp,.go,.rs,.rb,.php,.swift,.sh,.ps1,.bat,.cmd,.sql,.r,.lua,.dart,.scala,.css,.scss,.vue,.svelte,"
    + ".pdf,.docx,text/*,application/pdf,application/json";
  const fileTag = (a) => { const n = (a.name || "").toLowerCase(); const ext = n.includes(".") ? n.split(".").pop() : ""; return (ext || "txt").slice(0, 5); };
  const fmtChars = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k chars` : `${n} chars`);
  const modelCanSee = () => (currentModel()?.capabilities || []).includes("vision");

  async function addFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    const images = files.filter((f) => f.type.startsWith("image/"));
    const docs = files.filter((f) => !f.type.startsWith("image/"));
    if (images.length && !modelCanSee()) {
      toast(`${currentModel()?.name || "This model"} cannot see images. Switch to a vision model to attach pictures.`, true);
    } else {
      for (const file of images) {
        const b64 = await new Promise((res) => { const r = new FileReader(); r.onload = () => res(r.result.split(",")[1]); r.readAsDataURL(file); });
        state.pendingFiles.push({ kind: "image", b64, name: file.name });
      }
    }
    if (docs.length) {
      const form = new FormData();
      for (const f of docs) form.append("files", f, f.name);
      ui.composer.classList.add("uploading");
      try {
        const res = await fetch("/api/extract", { method: "POST", body: form });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        for (const r of await res.json()) {
          if (r.kind === "text") state.pendingFiles.push({ kind: "text", name: r.name, type: r.type, chars: r.chars, text: r.text, truncated: r.truncated });
          else if (r.kind === "image") toast(`${r.name}: attach images through their own file type`, true);
          else toast(`${r.name}: ${r.error || "not supported"}`, true);
        }
      } catch (e) { toast(`Could not read files: ${e.message}`, true); }
      finally { ui.composer.classList.remove("uploading"); }
    }
    renderAttachments();
  }
  ui.attach.onchange = async () => { await addFiles(ui.attach.files); ui.attach.value = ""; };
  ui.composer.addEventListener("dragover", (e) => { if (e.dataTransfer?.types?.includes("Files")) { e.preventDefault(); ui.composer.classList.add("dragover"); } });
  ui.composer.addEventListener("dragleave", () => ui.composer.classList.remove("dragover"));
  ui.composer.addEventListener("drop", (e) => { e.preventDefault(); ui.composer.classList.remove("dragover"); addFiles(e.dataTransfer.files); });
  ui.input.addEventListener("paste", (e) => { const files = Array.from(e.clipboardData?.files || []); if (files.length) { e.preventDefault(); addFiles(files); } });

  function renderAttachments() {
    updateContextMeter();
    ui.attachments.replaceChildren(...state.pendingFiles.map((f, i) => {
      const remove = el("button", { type: "button", text: "×", title: "Remove", onclick: () => { state.pendingFiles.splice(i, 1); renderAttachments(); } });
      if (f.kind === "image") return el("div", { class: "att" }, el("img", { src: `data:image/*;base64,${f.b64}`, alt: "" }), remove);
      return el("div", { class: "att att-file", title: f.name },
        el("span", { class: "att-icon", text: fileTag(f) }),
        el("span", { class: "att-meta" }, el("span", { class: "att-name", text: f.name }), el("span", { class: "att-size", text: fmtChars(f.chars) + (f.truncated ? ", cut" : "") })),
        remove);
    }));
    const chars = state.pendingFiles.reduce((n, f) => n + (f.kind === "text" ? Math.min(f.chars, f.text.length) : 0), 0);
    if (!chars) { ui.attNote.hidden = true; return; }
    const tokens = Math.round(chars / 3.2);
    const ctx = Number(state.chat?.settings.options?.num_ctx) || 4096;
    const fits = tokens < ctx - 1500;
    ui.attNote.hidden = false;
    ui.attNote.className = "att-note" + (fits ? "" : " warn");
    ui.attNote.textContent = fits
      ? `Files add about ${tokens.toLocaleString()} tokens to the prompt.`
      : `Files add about ${tokens.toLocaleString()} tokens, more than the ${ctx.toLocaleString()}-token context allows. The end will be cut; raise the context length in the controls to fit more.`;
  }
  function clearAttachments() { state.pendingFiles = []; renderAttachments(); }

  async function send() {
    if (state.streaming || !state.chat) return;
    const text = ui.input.value.trim();
    if (!text && !state.pendingFiles.length) return;
    if (!state.chat.settings.model) return toast("Pick a model in the controls first", true);
    const msg = { role: "user", content: text, created_at: new Date().toISOString() };
    const images = state.pendingFiles.filter((f) => f.kind === "image").map((f) => f.b64);
    const docs = state.pendingFiles.filter((f) => f.kind === "text").map(({ name, type, chars, text: t, truncated }) => ({ name, type, chars, text: t, truncated }));
    if (images.length) msg.images = images;
    if (docs.length) msg.attachments = docs;
    state.chat.messages.push(msg);
    ui.input.value = ""; autoGrow(); clearAttachments();
    renderMessages();
    await generate();
  }

  // ------------------------------------------------------------------ generation stream
  let readoutTimer = null;
  function setPhase(text) { ui.readoutPhase.textContent = text; }
  function updateReadout() {
    const s = state.streaming; if (!s) return;
    const elapsed = (performance.now() - s.startedAt) / 1000;
    const genElapsed = s.firstTokenAt ? (performance.now() - s.firstTokenAt) / 1000 : 0;
    const rate = genElapsed > 0.5 && s.tokens ? (s.tokens / genElapsed).toFixed(1) : "–";
    ui.readoutMetrics.textContent = `${s.tokens} tok   ${rate} tok/s   ${elapsed.toFixed(1)} s`;
    updateContextMeter();
  }
  function startReadout() {
    ui.readout.hidden = false;
    ui.send.textContent = "Stop"; ui.send.classList.add("stop");
    updateReadout();
    readoutTimer = setInterval(updateReadout, 120);
  }
  function stopReadout() {
    clearInterval(readoutTimer); readoutTimer = null;
    ui.readout.hidden = true;
    ui.send.textContent = "Send"; ui.send.classList.remove("stop");
  }
  async function stopGeneration() {
    const s = state.streaming; if (!s) return;
    setPhase("Stopping");
    try { await api(`/api/generate/${s.genId}/stop`, { method: "POST" }); } catch { /* stream will end anyway */ }
  }

  async function generate() {
    const chat = state.chat;
    const history = chat.messages.map(({ role, content, images, attachments, created_at, tool_calls }) => ({ role, content, images, attachments, created_at, tool_calls }));
    const assistant = { role: "assistant", content: "", thinking: "", model: chat.settings.model, created_at: new Date().toISOString() };
    chat.messages.push(assistant);
    const index = chat.messages.length - 1;
    const genId = uid();
    state.streaming = { genId, index, startedAt: performance.now(), tokens: 0, firstTokenAt: null, thinkStart: null, thinkEnd: null };

    // Build the live message node.
    const node = el("article", { class: `msg assistant streaming`, "data-index": index });
    node.append(el("div", { class: "msg-head" }, el("span", { class: "who", text: chat.settings.model }), el("span", { class: "when", text: fmtTime(assistant.created_at) })));
    const thinkWrap = el("div");
    const toolsWrap = el("div");
    const body = el("div", { class: "body" });
    const extra = el("div");
    node.append(thinkWrap, toolsWrap, body, extra);
    assistant.tool_calls = [];
    const renderToolsLive = () => {
      if (!assistant.tool_calls.length) return;
      const openIds = new Set($$("details.tool-call[open]", toolsWrap).map((d) => d.dataset.call));
      const block = renderTools(assistant.tool_calls, genId);
      for (const d of $$("details.tool-call", block)) if (openIds.has(d.dataset.call)) d.open = true;
      toolsWrap.replaceChildren(block);
      scrollToBottom();
    };
    const toolRecord = (id) => assistant.tool_calls.find((c) => c.id === id);
    const welcome = $(".welcome", ui.messages); if (welcome) welcome.remove();
    ui.messages.append(node);
    scrollToBottom(true);
    startReadout();
    setPhase("Waiting for the model");

    let renderQueued = false;
    let thinkDetails = null, thinkBody = null;
    const flushRender = () => {
      renderQueued = false;
      if (assistant.thinking) {
        if (!thinkDetails) {
          thinkDetails = renderThinking(assistant, true);
          thinkBody = $(".think-body", thinkDetails);
          thinkWrap.append(thinkDetails);
        }
        thinkBody.textContent = assistant.thinking;
        thinkBody.scrollTop = thinkBody.scrollHeight;
      }
      body.innerHTML = renderMarkdown(assistant.content, assistant.sources);
      scrollToBottom();
    };
    const queueRender = () => { if (!renderQueued) { renderQueued = true; requestAnimationFrame(flushRender); } };

    const body_ = {
      gen_id: genId, chat_id: chat.id, model: chat.settings.model, system_prompt: chat.settings.system_prompt,
      messages: history, options: chat.settings.options, think: chat.settings.think, loop_guard: chat.settings.loop_guard,
      web_search: chat.settings.web_search, think_budget: chat.settings.think_budget, timeout_s: chat.settings.timeout_s, save: true, chat_settings: chat.settings,
      agent: { ...(chat.settings.agent || {}), tools: agentTools(chat.settings) },
    };

    let finalized = false;
    const finalize = () => {
      if (finalized) return; finalized = true;
      state.streaming = null;
      stopReadout();
      if (assistant.thinking && state.thinkTimes?.start) {
        assistant.thinking_seconds = ((state.thinkTimes.end || performance.now()) - state.thinkTimes.start) / 1000;
      }
      state.thinkTimes = null;
      if (!assistant.tool_calls?.length) delete assistant.tool_calls;
      if (!assistant.content && !assistant.thinking && !assistant.error && !assistant.tool_calls) {
        chat.messages.splice(index, 1);
      }
      renderMessages();
      ui.input.focus();
      if (chat.saved && !chat.title) pickUpTitleLater(chat);
    };

    try {
      const res = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body_) });
      if (!res.ok || !res.body) throw new Error(`Server returned ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      state.thinkTimes = { start: null, end: null };
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep;
        while ((sep = buffer.indexOf("\n\n")) >= 0) {
          const frame = buffer.slice(0, sep); buffer = buffer.slice(sep + 2);
          const line = frame.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const ev = JSON.parse(line.slice(6));
          handleEvent(ev);
        }
      }
    } catch (e) {
      assistant.error = assistant.error || `Connection lost: ${e.message}`;
      toast(assistant.error, true);
    }
    finalize();

    function handleEvent(ev) {
      const s = state.streaming;
      switch (ev.type) {
        case "status":
          setPhase(ev.detail || ev.phase);
          break;
        case "sources":
          assistant.sources = ev.items; assistant.search_query = ev.query;
          extra.replaceChildren(renderSources(ev.query, ev.items));
          setPhase(`Reading ${ev.items.length} sources`);
          break;
        case "thinking":
          if (s && !s.firstTokenAt) s.firstTokenAt = performance.now();
          if (s) s.tokens++;
          if (!state.thinkTimes.start) state.thinkTimes.start = performance.now();
          assistant.thinking += ev.delta;
          setPhase("Thinking");
          queueRender();
          break;
        case "content":
          if (s && !s.firstTokenAt) s.firstTokenAt = performance.now();
          if (s) s.tokens++;
          if (state.thinkTimes.start && !state.thinkTimes.end) state.thinkTimes.end = performance.now();
          assistant.content += ev.delta;
          setPhase("Writing");
          queueRender();
          break;
        case "tool_call": {
          assistant.tool_calls.push({ id: ev.id, step: ev.step, name: ev.name, arguments: ev.arguments, purpose: ev.purpose || "", status: ev.status, approval: !!ev.approval, result: "" });
          setPhase(ev.name === "ask_user" ? "The model has a question for you" : ev.approval ? `Waiting for approval: ${ev.name}` : `Running ${ev.name}`);
          if (ev.name === "ask_user") toast("The model has a question for you. Answer it in the chat.");
          else if (ev.approval) toast(`The model wants to run ${ev.name}. Allow or deny it in the chat.`);
          renderToolsLive();
          break;
        }
        case "tool_status": {
          const rec = toolRecord(ev.id); if (rec) rec.status = ev.status;
          setPhase(`Running ${ev.name}`);
          renderToolsLive();
          break;
        }
        case "tool_result": {
          const rec = toolRecord(ev.id);
          if (rec) { rec.status = ev.status; rec.result = ev.result || ""; rec.seconds = ev.seconds; }
          setPhase("Continuing after tools");
          renderToolsLive();
          break;
        }
        case "done":
          assistant.stats = ev.stats || {};
          if (ev.aborted) assistant.aborted = ev.aborted;
          updateContextMeter();
          if (state.thinkTimes.start && !state.thinkTimes.end) state.thinkTimes.end = performance.now();
          setPhase(ev.aborted ? `Stopped: ${ev.aborted}` : "Done");
          break;
        case "saved":
          chat.saved = true;
          localStorage.setItem("harness.lastChat", chat.id);
          Object.assign(assistant, ev.message, { thinking_seconds: assistant.thinking_seconds });
          refreshChatList();
          break;
        case "title":
          chat.title = ev.title; ui.title.value = ev.title;
          refreshChatList();
          break;
        case "error":
          assistant.error = ev.message;
          toast(ev.message, true);
          break;
      }
    }
  }

  // ------------------------------------------------------------------ boot
  (async () => {
    await Promise.all([checkHealth(), loadServerSettings(), loadTools()]);
    await loadModels();
    await refreshChatList();
    const last = new URLSearchParams(location.search).get("chat") || localStorage.getItem("harness.lastChat");
    if (last && state.chats.some((c) => c.id === last)) await openChat(last); else newChat();
    setInterval(checkHealth, 30000);
    if (!canMarkdown) toast("Markdown libraries did not load; showing plain text.");
  })();
})();
