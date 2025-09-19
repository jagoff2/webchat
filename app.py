#!/usr/bin/env python3
"""
Flask UI + LLaMA (llama-server) + DuckDuckGo + http_get
Persistent chats, Markdown UI, citation chips styling, multi‑chat sidebar.

All original features are kept; the code has been refactored and hardened.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import os
import re
import sqlite3
import sys
import time
import uuid
import logging
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union
from urllib.parse import unquote, parse_qs, urlparse

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import requests
from bs4 import BeautifulSoup
from flask import ( Flask, abort, g, jsonify, render_template_string,
                    request, session, url_for, Response )
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
class Config:
    """Centralised Flask configuration."""
    # Paths
    DB_PATH: Path = Path(os.getenv("CHAT_DB_PATH", "chat_store.sqlite3"))
    # Secrets
    SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", uuid.uuid4().hex)
    # Limits
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024          # 16 MiB upload limit
    RATE_LIMIT: int = 5                               # requests per minute per IP
    # LLaMA server endpoint
    LLAMA_ENDPOINT: str = "http://127.0.0.1:8000/v1/chat/completions"
    # HTTP session for external calls (re‑use TCP connections)
    HTTP_TIMEOUT: int = 60
    # Logging
    LOG_LEVEL = logging.INFO

# ----------------------------------------------------------------------
# Flask app creation
# ----------------------------------------------------------------------
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=Config.SECRET_KEY,
    MAX_CONTENT_LENGTH=Config.MAX_CONTENT_LENGTH,
)
app.logger.setLevel(Config.LOG_LEVEL)

# ----------------------------------------------------------------------
# Simple in‑memory rate limiter (IP → timestamps)
# ----------------------------------------------------------------------
_ip_hits: defaultdict[str, List[float]] = defaultdict(list)

def _rate_limited() -> bool:
    """Return True if the current request exceeds the configured rate limit."""
    ip = request.remote_addr or "unknown"
    now = time.time()
    window = 60.0
    hits = _ip_hits[ip]

    # prune old entries
    while hits and hits[0] < now - window:
        hits.pop(0)
    if len(hits) >= Config.RATE_LIMIT:
        return True
    hits.append(now)
    return False

# ----------------------------------------------------------------------
# Content‑Security‑Policy header (added to every response)
# ----------------------------------------------------------------------
@app.after_request
@app.after_request
def add_security_headers(resp: Response) -> Response:
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self' https://cdn.jsdelivr.net; "
    )
    return resp


# ----------------------------------------------------------------------
# DB helpers – use a single connection per request
# ----------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    """Return a SQLite connection bound to Flask's `g`."""
    if "db" not in g:
        g.db = sqlite3.connect(str(Config.DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc: Optional[BaseException]) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db() -> None:
    """Create tables if they do not exist."""
    db = get_db()
    db.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS sessions (
        sid TEXT PRIMARY KEY,
        created REAL NOT NULL,
        updated REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        sid TEXT NOT NULL,
        name TEXT NOT NULL,
        created REAL NOT NULL,
        updated REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chats_sid ON chats(sid);
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,          -- system|user|assistant|tool
        content TEXT,                -- raw text or JSON payload for tools
        tool_name TEXT,
        tool_call_id TEXT,
        api_extra TEXT,              -- JSON for tool_calls etc.
        ts REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sid TEXT NOT NULL,
        filename TEXT NOT NULL,
        content TEXT NOT NULL,
        ts REAL NOT NULL
    );
    """)
    db.commit()

def now() -> float:
    return time.time()

# ----------------------------------------------------------------------
# Session / chat helpers
# ----------------------------------------------------------------------
def _ensure_sid() -> str:
    """Create a stable session identifier stored in Flask's session."""
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex[:16]
    return session["sid"]

def _ensure_session_row(sid: str) -> None:
    db = get_db()
    cur = db.execute("SELECT sid FROM sessions WHERE sid=?", (sid,))
    if not cur.fetchone():
        db.execute(
            "INSERT INTO sessions(sid,created,updated) VALUES(?,?,?)",
            (sid, now(), now()),
        )
    else:
        db.execute("UPDATE sessions SET updated=? WHERE sid=?", (now(), sid))
    db.commit()

def _create_chat(sid: str, name: Optional[str] = None) -> str:
    chat_id = uuid.uuid4().hex[:8]
    name = name or "Chat"
    db = get_db()
    db.execute(
        "INSERT INTO chats(id,sid,name,created,updated) VALUES(?,?,?,?,?)",
        (chat_id, sid, name, now(), now()),
    )
    db.commit()
    return chat_id

def _list_chats(sid: str) -> List[Dict[str, Any]]:
    db = get_db()
    cur = db.execute(
        "SELECT id,name,created,updated FROM chats WHERE sid=? ORDER BY updated DESC, created DESC",
        (sid,),
    )
    return [dict(row) for row in cur.fetchall()]

def _rename_chat(chat_id: str, name: str) -> None:
    db = get_db()
    db.execute(
        "UPDATE chats SET name=?, updated=? WHERE id=?", (name, now(), chat_id)
    )
    db.commit()

def _delete_chat(chat_id: str) -> None:
    db = get_db()
    db.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    db.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    db.commit()

def _insert_message(
    chat_id: str,
    role: str,
    content: str,
    *,
    tool_name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    api_extra: Optional[Dict[str, Any]] = None,
) -> None:
    db = get_db()
    db.execute(
        """INSERT INTO messages(
            chat_id, role, content, tool_name, tool_call_id, api_extra, ts
        ) VALUES (?,?,?,?,?,?,?)""",
        (
            chat_id,
            role,
            content,
            tool_name,
            tool_call_id,
            json.dumps(api_extra, ensure_ascii=False) if api_extra else None,
            now(),
        ),
    )
    db.execute("UPDATE chats SET updated=? WHERE id=?", (now(), chat_id))
    db.commit()

def _get_messages(chat_id: str) -> List[Dict[str, Any]]:
    db = get_db()
    cur = db.execute(
        "SELECT role,content,tool_name,tool_call_id,api_extra,ts FROM messages "
        "WHERE chat_id=? ORDER BY id ASC",
        (chat_id,),
    )
    msgs = []
    for row in cur.fetchall():
        extra = json.loads(row["api_extra"]) if row["api_extra"] else None
        msgs.append(
            {
                "role": row["role"],
                "content": row["content"],
                "tool_name": row["tool_name"],
                "tool_call_id": row["tool_call_id"],
                "api_extra": extra,
                "ts": row["ts"],
            }
        )
    return msgs

def _clear_chat(chat_id: str) -> None:
    db = get_db()
    db.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    db.commit()

def _list_uploads(sid: str) -> List[str]:
    db = get_db()
    cur = db.execute(
        "SELECT filename FROM uploads WHERE sid=? ORDER BY id DESC", (sid,)
    )
    return [row["filename"] for row in cur.fetchall()]

def _add_upload(sid: str, filename: str, content: str) -> None:
    db = get_db()
    db.execute(
        "INSERT INTO uploads(sid,filename,content,ts) VALUES(?,?,?,?)",
        (sid, filename, content, now()),
    )
    db.commit()

def _latest_web_search(chat_id: str) -> List[Dict[str, Any]]:
    """Return the last `web_search` tool payload as Python list."""
    db = get_db()
    cur = db.execute(
        """SELECT content FROM messages 
           WHERE chat_id=? AND role='tool' AND tool_name='web_search'
           ORDER BY id DESC LIMIT 1""",
        (chat_id,),
    )
    row = cur.fetchone()
    if not row or not row["content"]:
        return []
    try:
        data = json.loads(row["content"])
        return data if isinstance(data, list) else []
    except Exception:
        return []

# ----------------------------------------------------------------------
# Current‑chat handling (stored in Flask session)
# ----------------------------------------------------------------------
def _get_or_create_current_chat() -> str:
    sid = _get_sid()
    _ensure_session_row(sid)
    if "current_chat_id" not in session:
        existing = _list_chats(sid)
        if existing:
            session["current_chat_id"] = existing[0]["id"]
        else:
            session["current_chat_id"] = _create_chat(sid, "Chat 1")
            _insert_message(session["current_chat_id"], "system", SYSTEM_PROMPT)
    return session["current_chat_id"]

def _switch_chat(chat_id: str) -> bool:
    sid = _get_sid()
    available = {c["id"] for c in _list_chats(sid)}
    if chat_id in available:
        session["current_chat_id"] = chat_id
        return True
    return False

def _get_sid() -> str:
    return _ensure_sid()

# ----------------------------------------------------------------------
# Tools (DuckDuckGo search + HTTP GET)
# ----------------------------------------------------------------------
# Re‑use one `requests.Session` with retry logic for all external calls
_http = requests.Session()
_adapter = requests.adapters.HTTPAdapter(max_retries=3)
_http.mount("http://", _adapter)
_http.mount("https://", _adapter)

def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a DuckDuckGo search.

    Returns a list of dictionaries with keys: ``title``, ``url`` and ``snippet``.
    """
    api = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    try:
        r = _http.get(api, params=params, timeout=Config.HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results: List[Dict[str, str]] = []
        for item in data.get("RelatedTopics", []):
            # Some entries have nested "Topics"
            for sub in item.get("Topics", [item]):
                if "Text" in sub and "FirstURL" in sub:
                    results.append(
                        {
                            "title": sub["Text"].split(" - ")[0],
                            "url": sub["FirstURL"],
                            "snippet": sub["Text"],
                        }
                    )
                if len(results) >= max_results:
                    return results
        if results:
            return results
    except Exception as exc:
        app.logger.debug("DuckDuckGo JSON API error: %s", exc)

    # Fallback to HTML scrape if the API fails
    fallback_url = "https://html.duckduckgo.com/html/"
    try:
        r = _http.get(
            fallback_url,
            params={"q": query},
            timeout=Config.HTTP_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for div in soup.select("div.result"):
            a = div.select_one("a.result__a")
            snippet = div.select_one("a.result__snippet")
            if a:
                results.append(
                    {
                        "title": a.get_text(strip=True),
                        "url": a["href"],
                        "snippet": snippet.get_text(strip=True) if snippet else "",
                    }
                )
            if len(results) >= max_results:
                break
        return results or [
            {
                "title": "No results",
                "url": "",
                "snippet": "DuckDuckGo returned nothing for the query.",
            }
        ]
    except Exception as exc:
        app.logger.debug("DuckDuckGo HTML fallback error: %s", exc)
        return [
            {
                "title": "No results",
                "url": "",
                "snippet": "DuckDuckGo returned nothing for the query.",
            }
        ]

def http_get_fetch(url: str, max_bytes: int = 200_000) -> Dict[str, Any]:
    """
    Retrieve a URL and return a short visible‑text extract.

    The returned dict always contains ``ok`` (bool) and either ``text`` or ``error``.
    """
    try:
        r = _http.get(
            url,
            timeout=Config.HTTP_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        content = r.content[:max_bytes]
        soup = BeautifulSoup(content, "html.parser")
        for bad in soup(["script", "style", "noscript"]):
            bad.extract()
        text = " ".join(soup.get_text(separator=" ").split())
        return {"ok": True, "status": r.status_code, "url": url, "text": text[:max_bytes]}
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}

WEB_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the public web for up‑to‑date information using DuckDuckGo.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (1‑10).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

HTTP_GET_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "http_get",
        "description": "Fetch a URL and return visible page text for grounding.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_bytes": {
                    "type": "integer",
                    "default": 200_000,
                    "minimum": 10_000,
                    "maximum": 400_000,
                },
            },
            "required": ["url"],
        },
    },
}
TOOLS = [WEB_SEARCH_TOOL, HTTP_GET_TOOL]

# ----------------------------------------------------------------------
# System prompt (model‑led, never altered by server)
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful assistant powered by a local LLaMA model.

You may call tools. When information is time‑sensitive or uncertain, call web_search.
If you need page contents from a search result, call http_get on one of the URLs.
After using tools as needed, produce a concise, human‑readable answer.
Never fabricate URLs or facts. Cite sources plainly or with clear links.
Do not include chain‑of‑thought.
"""

# ----------------------------------------------------------------------
# LLaMA orchestration
# ----------------------------------------------------------------------
WINDOW_USER_TURNS = 6      # last N user+assistant messages
WINDOW_TOOL_PAIRS = 6      # last N tool call/result pairs
MAX_TOOL_STEPS = 64        # safety guard

def _build_windowed_messages(full: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Trim the full history to a manageable window that still contains the
    latest system message, recent user/assistant turns and recent tool
    interactions.
    """
    # Keep the most recent system message only
    system = [m for m in full if m["role"] == "system"]
    others = [m for m in full if m["role"] != "system"]

    result: List[Dict[str, Any]] = system[-1:]  # latest system only
    compact: List[Dict[str, Any]] = []
    tool_kept = 0
    ua_kept = 0

    for m in reversed(others):
        if m["role"] == "tool":
            if tool_kept < WINDOW_TOOL_PAIRS:
                compact.insert(0, m)
                tool_kept += 1
        elif m["role"] == "assistant" and (m.get("api_extra") or {}).get("tool_calls"):
            if tool_kept < WINDOW_TOOL_PAIRS:
                compact.insert(0, m)
                tool_kept += 1
        else:  # user or plain assistant
            if ua_kept < 2 * WINDOW_USER_TURNS:
                compact.insert(0, m)
                ua_kept += 1

    result.extend(compact)

    # Convert to OpenAI‑compatible payload for llama‑server
    api_msgs: List[Dict[str, Any]] = []
    for m in result:
        role = m["role"]
        if role in ("system", "user", "assistant"):
            entry: Dict[str, Any] = {"role": role, "content": m.get("content") or ""}
            extra = m.get("api_extra") or {}
            if role == "assistant" and extra.get("tool_calls"):
                entry["tool_calls"] = extra["tool_calls"]
            api_msgs.append(entry)
        elif role == "tool":
            api_msgs.append(
                {
                    "role": "tool",
                    "content": m.get("content") or "",
                    "name": m.get("tool_name") or "",
                    "tool_call_id": m.get("tool_call_id") or "",
                }
            )
    return api_msgs

def _call_llama(messages: List[Dict[str, Any]], *, tools: Optional[List[Dict[str, Any]]] = None,
                tool_choice: str = "auto") -> Dict[str, Any]:
    """
    Send a request to the local LLaMA server. Retries are handled by the
    underlying ``requests.Session``.
    """
    payload: Dict[str, Any] = {
        "model": "local-llama",
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": 64000,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    app.logger.debug("→ LLaMA payload: %s", json.dumps(payload, ensure_ascii=False)[:800])
    resp = _http.post(Config.LLAMA_ENDPOINT, json=payload, timeout=300)  # generous timeout
    resp.raise_for_status()
    result = resp.json()
    app.logger.debug("← LLaMA response: %s", json.dumps(result, ensure_ascii=False)[:800])
    return result

def _run_until_answer(chat_id: str) -> Tuple[str, bool]:
    """
    Execute the model/orchestrator loop until a final assistant message is produced.
    Returns (assistant_text, tool_was_used).
    """
    steps = 0
    full = _get_messages(chat_id)

    while steps < MAX_TOOL_STEPS:
        steps += 1
        msgs = _build_windowed_messages(full)
        resp = _call_llama(msgs, tools=TOOLS, tool_choice="auto")
        message = resp["choices"][0]["message"]
        app.logger.info("Step %d – model output: %s", steps, json.dumps(message, ensure_ascii=False)[:600])

        # ----------------------- TOOL CALLS -----------------------
        if message.get("tool_calls"):
            _insert_message(
                chat_id,
                "assistant",
                message.get("content") or "",
                api_extra={"tool_calls": message["tool_calls"]},
            )
            for tc in message["tool_calls"]:
                fn = tc["function"]["name"]
                args = json.loads(tc["function"].get("arguments") or "{}")
                call_id = tc.get("id")
                if fn == "web_search":
                    query = args.get("query", "")
                    top_k = int(args.get("top_k", 5))
                    top_k = max(1, min(top_k, 10))
                    try:
                        results = duckduckgo_search(query, max_results=top_k)
                        output = json.dumps(results, ensure_ascii=False)
                    except Exception as e:
                        output = json.dumps(
                            {"error": str(e), "query": query}, ensure_ascii=False
                        )
                    _insert_message(
                        chat_id,
                        "tool",
                        output,
                        tool_name="web_search",
                        tool_call_id=call_id,
                    )
                elif fn == "http_get":
                    url = args.get("url", "")
                    max_bytes = int(args.get("max_bytes", 200_000))
                    result = http_get_fetch(url, max_bytes=max_bytes)
                    output = json.dumps(result, ensure_ascii=False)
                    _insert_message(
                        chat_id,
                        "tool",
                        output,
                        tool_name="http_get",
                        tool_call_id=call_id,
                    )
                else:
                    _insert_message(
                        chat_id,
                        "tool",
                        json.dumps({"error": f"unknown tool {fn}"}, ensure_ascii=False),
                        tool_name=fn,
                        tool_call_id=call_id,
                    )
            # Refresh history (including tool results) and continue the loop
            full = _get_messages(chat_id)
            continue

        # -------------------- FINAL ANSWER --------------------
        answer = (message.get("content") or "").strip()
        _insert_message(chat_id, "assistant", answer)
        # Was any tool used in this round?
        tool_used = any(m["role"] == "tool" for m in full)
        return answer, tool_used

    raise RuntimeError("Exceeded maximum tool steps – likely runaway loop.")

# ----------------------------------------------------------------------
# Flask routes
# ----------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index() -> Response:
    _get_or_create_current_chat()
    sid = _get_sid()
    chat_id = session["current_chat_id"]
    msgs = _get_messages(chat_id)

    # Build display history for the UI (user & assistant as‐is, tool as fenced block)
    display: List[Dict[str, Any]] = []
    for m in msgs:
        if m["role"] in ("user", "assistant"):
            display.append({"role": m["role"], "content": m["content"]})
        elif m["role"] == "tool":
            display.append(
                {
                    "role": "tool",
                    "content": f"```{m.get('tool_name') or 'tool'}\n{m.get('content')}\n```",
                    "tool_banner": f"✅ {m.get('tool_name') or 'tool'} executed",
                }
            )

    last_web = _latest_web_search(chat_id)

    return render_template_string(
        HTML_TEMPLATE,
        display_history=display,
        uploaded_files=_list_uploads(sid),
        last_web_results=last_web,
    )

@app.route("/chat", methods=["POST"])
def chat_endpoint() -> Response:
    if _rate_limited():
        abort(429, description="Too many requests – please wait a moment.")
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    try:
        chat_id = _get_or_create_current_chat()
        # guarantee a system message exists at the top of the chat
        msgs = _get_messages(chat_id)
        if not msgs or msgs[0]["role"] != "system":
            _insert_message(chat_id, "system", SYSTEM_PROMPT)
        _insert_message(chat_id, "user", user_msg)

        reply, tool_used = _run_until_answer(chat_id)

        return jsonify(
            {
                "reply": reply,
                "tool_banner": "🔧 tool was called" if tool_used else None,
                "last_web_results": _latest_web_search(chat_id),
            }
        )
    except Exception as exc:
        app.logger.error("Exception in /chat: %s", traceback.format_exc())
        return jsonify({"error": str(exc)}), 500

@app.route("/upload", methods=["POST"])
def upload() -> Response:
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    # Basic sanitisation – strip path components and limit length
    filename = os.path.basename(f.filename)
    filename = filename[:255]  # filesystem safe limit

    try:
        content = f.read().decode(errors="replace")
        _add_upload(_get_sid(), filename, content)
        return jsonify({"success": True})
    except Exception as exc:
        app.logger.error("Upload failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500

# ----------------------- Chat management (sidebar) -----------------------
@app.route("/chats", methods=["GET"])
def chats_list() -> Response:
    sid = _get_sid()
    items = _list_chats(sid)
    cur = session.get("current_chat_id")
    return jsonify({"items": items, "current_id": cur})

@app.route("/new_chat", methods=["POST"])
def new_chat() -> Response:
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    sid = _get_sid()
    new_id = _create_chat(sid, name or f"Chat {len(_list_chats(sid)) + 1}")
    _insert_message(new_id, "system", SYSTEM_PROMPT)
    session["current_chat_id"] = new_id
    return jsonify({"success": True, "id": new_id})

@app.route("/switch_chat/<chat_id>", methods=["POST"])
def switch_chat(chat_id: str) -> Response:
    ok = _switch_chat(chat_id)
    if ok:
        msgs = _get_messages(chat_id)
        if not msgs or msgs[0]["role"] != "system":
            _insert_message(chat_id, "system", SYSTEM_PROMPT)
    return jsonify({"success": ok})

@app.route("/rename_chat", methods=["POST"])
def rename_chat() -> Response:
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"success": False, "error": "Missing name"}), 400
    cid = _get_or_create_current_chat()
    _rename_chat(cid, name)
    return jsonify({"success": True})

@app.route("/clear_chat", methods=["POST"])
def clear_chat() -> Response:
    cid = _get_or_create_current_chat()
    _clear_chat(cid)
    _insert_message(cid, "system", SYSTEM_PROMPT)
    return jsonify({"success": True})

@app.route("/delete_chat/<chat_id>", methods=["POST"])
def delete_chat(chat_id: str) -> Response:
    _delete_chat(chat_id)
    sid = _get_sid()
    remaining = _list_chats(sid)
    if remaining:
        session["current_chat_id"] = remaining[0]["id"]
    else:
        session["current_chat_id"] = _create_chat(sid, "Chat 1")
        _insert_message(session["current_chat_id"], "system", SYSTEM_PROMPT)
    return jsonify({"success": True})

@app.route("/debug", methods=["GET"])
def debug() -> Response:
    cid = _get_or_create_current_chat()
    msgs = _get_messages(cid)
    return jsonify(
        {
            "messages_count": len(msgs),
            "window_cfg": {"user_turns": WINDOW_USER_TURNS, "tool_pairs": WINDOW_TOOL_PAIRS},
            "current_chat_id": cid,
        }
    )

@app.route("/health", methods=["GET"])
def health() -> Response:
    """Simple health check used by orchestration platforms."""
    return jsonify({"status": "ok", "timestamp": now()})

# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------
@app.errorhandler(404)
def not_found(err) -> Response:
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(429)
def too_many_requests(err) -> Response:
    return jsonify({"error": str(err)}), 429

@app.errorhandler(500)
def internal_error(err) -> Response:
    app.logger.error("Internal server error: %s", traceback.format_exc())
    return jsonify({"error": "Internal server error"}), 500

# ----------------------------------------------------------------------
# UI template (unchanged except minor styling tweaks)
# ----------------------------------------------------------------------
HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><title>🦙 LLaMA Chat + DuckDuckGo</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {background:#f8f9fa;}
    .layout {display:flex; min-height:100vh;}
    .sidebar {width:280px; background:#fff; border-right:1px solid #e9ecef; padding:1rem; overflow-y:auto;}
    .chat-area {flex:1; display:flex; flex-direction:column;}
    .chat-box {max-height:70vh; overflow-y:auto;}
    .msg {margin:.5rem 0;}
    .msg-user {text-align:right;}
    .msg-assistant {text-align:left;}
    .msg-tool {text-align:left; opacity:.9;}
    .msg-content {display:inline-block;padding:.6rem 1rem;border-radius:.8rem;max-width:75%;word-wrap:break-word;white-space:normal;}
    .msg-user .msg-content {background:#0d6efd;color:#fff;}
    .msg-assistant .msg-content {background:#e9ecef;color:#212529;}
    .msg-tool .msg-content {background:#fff;color:#212529;border:1px dashed #adb5bd;}
    .typing .msg-content {font-style:italic;color:#666;}
    .tool-banner {font-size:.85rem;color:#555;margin:.2rem 0;}
    pre {background:#f1f3f5;padding:.8rem;border-radius:.4rem;overflow:auto;}
    code {font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;}
    .cite {vertical-align:super;font-size:.75em;margin-left:.15rem;}
    .cite-link {text-decoration:none;border-bottom:1px dotted rgba(13,110,253,.5);opacity:.75;padding:0 .15rem;}
    .cite-link:hover,.cite-link:focus {opacity:1;border-bottom-color:rgba(13,110,253,.9);}
    .chat-item{display:flex;align-items:center;justify-content:space-between;padding:.4rem .5rem;border-radius:.5rem;cursor:pointer;}
    .chat-item:hover{background:#f8f9fa;}
    .chat-item.active{background:#e7f1ff;border:1px solid #cfe2ff;}
    .chat-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:170px;}
  </style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="d-flex align-items-center justify-content-between mb-2">
      <h5 class="mb-0">Conversations</h5>
      <button id="new-chat" class="btn btn-sm btn-primary">New</button>
    </div>
    <div id="chat-list" class="vstack gap-1"></div>
  </aside>

  <main class="chat-area">
    <div class="container py-4 flex-grow-1 d-flex flex-column">
      <div class="d-flex align-items-center justify-content-between mb-3">
        <div class="chat-title" id="current-chat-title">Chat</div>
        <div>
          <button id="rename-chat" class="btn btn-sm btn-outline-secondary">Rename</button>
          <button id="clear-chat" class="btn btn-sm btn-outline-danger">Clear</button>
        </div>
      </div>

      <div class="chat-box flex-grow-1 mb-3 p-3 bg-white rounded shadow-sm" id="chat">
        {% for entry in display_history %}
          <div class="msg {% if entry.role == 'user' %}msg-user{% elif entry.role == 'assistant' %}msg-assistant{% else %}msg-tool{% endif %}">
            <div class="msg-content render-md" data-raw="{{ entry.content | tojson }}"></div>
            {% if entry.tool_banner %}
              <div class="tool-banner">{{ entry.tool_banner }}</div>
            {% endif %}
          </div>
        {% endfor %}
      </div>

      <form id="chat-form" class="row g-2">
        <div class="col-9">
          <textarea rows="3" class="form-control" id="user-input"
            placeholder="Shift+Enter for newline, Enter to send"
            autocomplete="off" required></textarea>
        </div>
        <div class="col-3 d-grid"><button type="submit" class="btn btn-primary">Send</button></div>
      </form>

      <hr class="my-4">

      <h5>Upload a file (optional)</h5>
      <form id="upload-form" enctype="multipart/form-data" class="row g-2">
        <div class="col-9"><input type="file" class="form-control" name="file" required></div>
        <div class="col-3 d-grid"><button type="submit" class="btn btn-secondary">Upload</button></div>
      </form>

      {% if uploaded_files %}
        <div class="mt-3">
          <h6>Uploaded files (available to future tools):</h6>
          <ul class="list-group file-list">
            {% for f in uploaded_files %}
              <li class="list-group-item py-1">{{ f }}</li>
            {% endfor %}
          </ul>
        </div>
      {% endif %}
    </div>
    <footer class="text-center py-3"><small>© 2025 – Flask + LLaMA + DuckDuckGo</small></footer>
  </main>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<script>
let latestToolUrl = '';
let lastSearchResults = {{ (last_web_results or []) | tojson }};

const chatBox = document.getElementById('chat');
const chatList = document.getElementById('chat-list');
const currentTitleEl = document.getElementById('current-chat-title');

function initTooltips(root) {
  const els = root.querySelectorAll('[data-bs-toggle="tooltip"]');
  els.forEach(el => new bootstrap.Tooltip(el));
}
const sanitizeCfg = {ADD_ATTR:['data-bs-toggle','data-bs-title','target','rel']};

/* ---------- Citation helpers ---------- */
function unwrapDDG(url){
  try{
    const u = new URL(url);
    if (u.hostname.includes('duckduckgo.com')) {
      const real = u.searchParams.get('uddg');
      if (real) return decodeURIComponent(real);
    }
    return url;
  }catch{return url;}
}
function makeCiteHTML(n,url){
  const clean = unwrapDDG(url||'');
  const tip = clean||'Source';
  return `<small><sup class="cite"><a class="cite-link" href="${clean}" target="_blank" rel="noopener" data-bs-toggle="tooltip" data-bs-title="${tip}">[${n}]</a></sup></small>`;
}
function resultUrl(idx){ return (lastSearchResults[idx]||{}).url||''; }
function replaceBracketDagger(html){
  return html.replace(/【\s*(\d+)\s*†[^】]*】/g,(_,num)=>{
    const i = Math.max(0,parseInt(num,10)-1);
    return makeCiteHTML(i+1, resultUrl(i));
  });
}
function replacePlainBracketUrl(html){
  let cnt = lastSearchResults.length;
  html = html.replace(/【\s*(https?:\/\/[^】\s]+)\s*】/g,(m,url)=>{
    cnt+=1; return makeCiteHTML(cnt,url);
  });
  html = html.replace(/【\s*([^\s\}]+)\s*】/g,(m,token)=>{
    let match='';
    for(const r of lastSearchResults) if(r.url && r.url.toLowerCase().includes(token.toLowerCase())){ match=r.url; break; }
    if(!match) match = latestToolUrl || resultUrl(0);
    cnt+=1; return makeCiteHTML(cnt, match);
  });
  return html;
}
function replaceBracketJSON(html){
  return html.replace(/【\s*(\{[^】]*\})\s*】/g,(_,jsonStr)=>{
    try{
      const obj = JSON.parse(jsonStr);
      const idx = obj.id||obj.index||obj.result||obj.i||0;
      const i = Math.max(0,idx);
      return makeCiteHTML(i+1, resultUrl(i));
    }catch{ return makeCiteHTML(1,resultUrl(0)); }
  });
}
function replaceSourceAnchors(root){
  const anchors = root.querySelectorAll('a');
  anchors.forEach(a=>{
    const txt = (a.textContent||'').trim();
    const m = /^source\s+(\d+)$/i.exec(txt);
    if(!m) return;
    const n = parseInt(m[1],10);
    const url = a.getAttribute('href')||resultUrl(n-1);
    const html = makeCiteHTML(n,url);
    const span = document.createElement('span');
    span.innerHTML = html;
    a.replaceWith(span.firstChild);
  });
}
function prettifyCitations(el){
  let html = replaceBracketJSON(replaceBracketDagger(el.innerHTML));
  html = replacePlainBracketUrl(html);
  el.innerHTML = html;
  replaceSourceAnchors(el);
}
function renderMarkdown(el){
  const raw = el.dataset.raw ?? el.innerText ?? '';
  const html = marked.parse(raw);
  el.innerHTML = DOMPurify.sanitize(html, sanitizeCfg);
  prettifyCitations(el);
  initTooltips(el);
}
function renderAll(){
  document.querySelectorAll('.render-md').forEach(renderMarkdown);
  chatBox.scrollTop = chatBox.scrollHeight;
}
function appendMessage(role, raw, banner=null){
  const div = document.createElement('div');
  const roleCls = role==='user'?'msg-user':role==='assistant'?'msg-assistant':'msg-tool';
  div.className = `msg ${roleCls}`;

  const content = document.createElement('div');
  content.className = 'msg-content';
  content.dataset.raw = raw||'';
  renderMarkdown(content);
  div.appendChild(content);

  if(banner){
    const b = document.createElement('div');
    b.className = 'tool-banner';
    b.textContent = banner;
    div.appendChild(b);
  }

  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;

  if(role==='tool'){
    try{
      const data = JSON.parse(raw);
      const url = Array.isArray(data) && data[0]?.url ? data[0].url :
                  data?.url ? data.url : '';
      if(url) latestToolUrl = url;
    }catch{}
  }
}

/* ---------- UI interactions ---------- */
const userInput = document.getElementById('user-input');
userInput.addEventListener('keydown', e=>{
  if(e.key==='Enter' && !e.shiftKey){
    e.preventDefault();
    document.getElementById('chat-form').requestSubmit();
  }
});
document.getElementById('chat-form').addEventListener('submit', async e=>{
  e.preventDefault();
  const txt = userInput.value.trim();
  if(!txt) return;
  appendMessage('user', txt);
  userInput.value = '';

  const typing = document.createElement('div');
  typing.className = 'msg msg-assistant typing';
  const tcontent = document.createElement('div');
  tcontent.className = 'msg-content';
  tcontent.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> <em>Assistant is thinking…</em>`;
  typing.appendChild(tcontent);
  chatBox.appendChild(typing);
  chatBox.scrollTop = chatBox.scrollHeight;

  const resp = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:txt})
  });
  const data = await resp.json();
  typing.remove();
  if(data.error){ alert(data.error); return; }
  if(Array.isArray(data.last_web_results)) lastSearchResults = data.last_web_results;
  appendMessage('assistant', data.reply||'');
  if(data.tool_banner) appendMessage('tool','`tool activity`',data.tool_banner);
  updateSidebar();
});

document.getElementById('upload-form').addEventListener('submit', async e=>{
  e.preventDefault();
  const fd = new FormData(e.target);
  const resp = await fetch('/upload', {method:'POST', body:fd});
  const data = await resp.json();
  if(!data.success) alert('Upload failed: '+data.error);
});

async function updateSidebar(){
  const r = await fetch('/chats');
  const data = await r.json();
  chatList.innerHTML = '';
  data.items.forEach(item=>{
    const row = document.createElement('div');
    row.className = 'chat-item'+(data.current_id===item.id?' active':'');
    const left = document.createElement('div');
    left.className='chat-name';
    left.textContent=item.name;
    const actions = document.createElement('div');
    const del = document.createElement('button');
    del.className='btn btn-sm btn-outline-danger';
    del.textContent='Delete';
    del.addEventListener('click', async ev=>{
      ev.stopPropagation();
      if(!confirm('Delete this chat?')) return;
      const rr = await fetch('/delete_chat/'+item.id,{method:'POST'});
      const dj = await rr.json();
      if(dj.success) location.reload(); else alert('Delete failed');
    });
    row.addEventListener('click', async ()=>{
      const rr = await fetch('/switch_chat/'+item.id,{method:'POST'});
      const dj = await rr.json();
      if(dj.success) location.reload();
    });
    actions.appendChild(del);
    row.appendChild(left);
    row.appendChild(actions);
    chatList.appendChild(row);
    if(data.current_id===item.id) currentTitleEl.textContent=item.name;
  });
}
document.getElementById('new-chat').addEventListener('click', async ()=>{
  const name = prompt('Name this chat:', '');
  const r = await fetch('/new_chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  const j = await r.json();
  if(j.success) location.reload();
});
document.getElementById('rename-chat').addEventListener('click', async ()=>{
  const name = prompt('New name:', currentTitleEl.textContent||'');
  if(!name) return;
  const r = await fetch('/rename_chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  const j = await r.json();
  if(j.success){ currentTitleEl.textContent=name; updateSidebar(); }
});
document.getElementById('clear-chat').addEventListener('click', async ()=>{
  if(!confirm('Clear all messages in this chat?')) return;
  const r = await fetch('/clear_chat',{method:'POST'});
  const j = await r.json();
  if(j.success) location.reload();
});
renderAll();
updateSidebar();
</script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# Application entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure DB exists before first request
    with app.app_context():
        init_db()
    # Use production‑ready host/port when run via a WSGI server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)