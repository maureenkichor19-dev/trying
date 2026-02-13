
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import re
from typing import List, Literal, Optional, Dict, Any, Tuple
import os
import json
import time
import httpx
from datetime import datetime, timezone
import asyncio
from dotenv import load_dotenv
from urllib.parse import urlparse
from app.services.rag_service import retrieve_rag_context, is_rag_domain, is_rag_available
from app.services.domain_classifier import classify_domain
from app.services.moderation_service import moderate_message
from app.services.domain_classifier import classify_domain, validate_domain
from app.services.internet_rag_service import internet_rag_search_and_extract, build_web_evidence_block

load_dotenv()

router = APIRouter(prefix="/llamachats-multi", tags=["llamachat-plus"])

# -----------------------
# ENV
# -----------------------
LLAMA_CLOUDRUN_URL = os.getenv("LLAMA_CLOUDRUN_URL", "")

# RunPod Serverless (job-mode)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
RUNPOD_AUTH_HEADER = os.getenv("RUNPOD_AUTH_HEADER", "Authorization").strip()
RUNPOD_RUN_ENDPOINT = os.getenv("RUNPOD_RUN_ENDPOINT", "").strip()
RUNPOD_STATUS_ENDPOINT = os.getenv("RUNPOD_STATUS_ENDPOINT", "").strip()
RUNPOD_MAX_WAIT_SEC = float(os.getenv("RUNPOD_MAX_WAIT_SEC", "600"))
RUNPOD_POLL_INTERVAL_SEC = float(os.getenv("RUNPOD_POLL_INTERVAL_SEC", "1.5"))

# Optional: cap total time spent in *additional* RunPod passes (sanity/web second-pass).
# Keeps one request from submitting several jobs and hitting local 504s under cold starts/queueing.
RUNPOD_TOTAL_BUDGET_SEC = float(os.getenv("RUNPOD_TOTAL_BUDGET_SEC", "240"))

# Debug logging (prompt previews may contain user text)
AV_LOG_PROMPTS = os.getenv("AV_LOG_PROMPTS", "1") == "1"
AV_LOG_PROMPT_CHARS = int(os.getenv("AV_LOG_PROMPT_CHARS", "900"))
AV_LOG_LEARNING_PREF = os.getenv("AV_LOG_LEARNING_PREF", "1") == "1"

# One-time debug snapshot so log gating is obvious.
_AV_FLAGS_SNAPSHOT_PRINTED = False


def _log_av_flags_snapshot_once() -> None:
    """Print effective AV_LOG_* values once per process.

    This helps diagnose why prompt/learning-preference logs appear missing.
    Safe: does not include user content.
    """
    global _AV_FLAGS_SNAPSHOT_PRINTED
    if _AV_FLAGS_SNAPSHOT_PRINTED:
        return
    _AV_FLAGS_SNAPSHOT_PRINTED = True
    try:
        print(
            "🧾 AV_LOG flags snapshot:",
            {
                "AV_LOG_PROMPTS": AV_LOG_PROMPTS,
                "AV_LOG_PROMPT_CHARS": AV_LOG_PROMPT_CHARS,
                "AV_LOG_LEARNING_PREF": AV_LOG_LEARNING_PREF,
                "env_AV_LOG_PROMPTS": os.getenv("AV_LOG_PROMPTS"),
                "env_AV_LOG_PROMPT_CHARS": os.getenv("AV_LOG_PROMPT_CHARS"),
                "env_AV_LOG_LEARNING_PREF": os.getenv("AV_LOG_LEARNING_PREF"),
                "runpod_enabled": bool(RUNPOD_RUN_ENDPOINT),
            },
            flush=True,
        )
    except Exception:
        # Never allow debug logging to break requests.
        pass

# Supabase (REST + Storage)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "chat-images")

# Google APIs
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")  # Custom Search Engine ID
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# Internet RAG provider (optional)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

USE_SUPABASE_STORAGE_FOR_IMAGES = os.getenv("USE_SUPABASE_STORAGE_FOR_IMAGES", "0") == "1"
FORCE_WEB_SOURCES = os.getenv("FORCE_WEB_SOURCES", "1") == "1"
FORCE_YOUTUBE = os.getenv("FORCE_YOUTUBE", "0") == "1"
FORCE_IMAGES = os.getenv("FORCE_IMAGES", "0") == "1"
MODEL_KNOWLEDGE_CUTOFF_YEAR = int(os.getenv("MODEL_KNOWLEDGE_CUTOFF_YEAR", "2023"))
BLOCK_YT_CHANNELS = [s.strip() for s in (os.getenv("BLOCK_YT_CHANNELS", "Formula 1,F1,Formula One").split(",")) if s.strip()]
BLOCK_IMAGE_HOSTS = [s.strip().lower() for s in (os.getenv(
    "BLOCK_IMAGE_HOSTS",
    "instagram.com,fbcdn.net,facebook.com,x.com,twitter.com,tiktok.com,linkedin.com,reddit.com,redd.it"
).split(",")) if s.strip()]

# Output style (friendly tone + emoji) — configurable
AV_EMOJI_STYLE = (os.getenv("AV_EMOJI_STYLE", "light") or "").strip().lower()  # off | light | strong
AV_FRIENDLY_OPENERS = os.getenv("AV_FRIENDLY_OPENERS", "1") == "1"  # default ON; brief greeting guidance

# Explicitly log CloudRun usage (no local model by default)
if LLAMA_CLOUDRUN_URL:
    print(
        f"☁️ CloudRun mode enabled. Target: {LLAMA_CLOUDRUN_URL} (local model disabled)",
        flush=True,
    )

# One-time meta logging
cloud_meta_logged: bool = False

async def _log_cloudrun_meta_once() -> None:
    global cloud_meta_logged
    if cloud_meta_logged or not LLAMA_CLOUDRUN_URL:
        return
    try:
        base = LLAMA_CLOUDRUN_URL.rstrip('/')
        if base.endswith('/chat'):
            base = base[: -len('/chat')]
        meta_url = f"{base}/meta"
        root_url = f"{base}/"
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(meta_url)
            if r.status_code >= 400:
                r = await client.get(root_url)
            data = r.json() if r.headers.get('content-type','').startswith('application/json') else {}
            n_ctx = data.get('n_ctx')
            n_threads = data.get('n_threads')
            max_tokens = data.get('max_tokens')
            system = data.get('system')
            model_path = data.get('model_path')
            if any(v is not None for v in (n_ctx, n_threads, max_tokens, model_path, system)):
                print(
                    "☁️ CloudRun LLaMA config:",
                    {
                        "n_ctx": n_ctx,
                        "n_threads": n_threads,
                        "max_tokens": max_tokens,
                        "model_path": model_path,
                        "system": system,
                    },
                    flush=True,
                )
            else:
                print(
                    "☁️ CloudRun meta endpoint not exposing config. Optional: add /meta to Cloud app to return n_ctx, n_threads, max_tokens.",
                    flush=True,
                )
    except Exception:
        # Silent fail; do not block startup if meta unreachable
        pass
    finally:
        cloud_meta_logged = True

# -----------------------
# Local Llama (optional debug)
# Mirrors Cloud Run app.py style loading
# -----------------------
ENABLE_LOCAL_LLAMA = os.getenv("ENABLE_LOCAL_LLAMA", "0") == "1"
local_llm = None
if ENABLE_LOCAL_LLAMA:
    try:
        # Import locally to avoid crashing if library is missing
        from llama_cpp import Llama
        # Config similar to Cloud app.py
        LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "./model.gguf")
        N_CTX = int(os.getenv("N_CTX", "2048"))
        N_THREADS = int(os.getenv("N_THREADS", "4"))

        if os.path.exists(LOCAL_MODEL_PATH):
            print("--- Llama Configuration ---", flush=True)
            print(f"💻 Loading local model from {LOCAL_MODEL_PATH}...", flush=True)
            try:
                local_llm = Llama(
                    model_path=LOCAL_MODEL_PATH,
                    n_ctx=N_CTX,
                    n_threads=N_THREADS,
                    n_gpu_layers=0,  # CPU-only by default; set to -1 if you have GPU layers
                    verbose=False,
                )
                print("✅ Local Model Loaded Successfully.", flush=True)
            except Exception as e:
                print(f"⚠️ Failed to load local Llama model: {e}", flush=True)
        # If path is missing, stay silent and keep using Cloud
    except Exception as e:
        # If llama_cpp isn't installed, silently skip local mode
        pass

# -----------------------
# PROMPT SIZE GUARDS
# -----------------------
MAX_PROMPT_CHARS = 10000
MAX_ARTICLE_CHARS = 1200
MAX_RAG_CHARS = 600
MAX_EVIDENCE_CHARS = 3500
MAX_HISTORY_CHARS = 900
MAX_MESSAGE_CHARS = 800

# -----------------------
# PROMPTS
# -----------------------
FORMAT_INSTRUCTION = (
    "Always format your answer clearly.\n"
    "- When giving multiple suggestions, use a numbered list with ONE item per line.\n"
    "- Avoid giant wall-of-text paragraphs.\n"
)

# -----------------------
# Learning preference prompt shaping
# -----------------------
_LEARNING_PREF_ALLOWED = {"secondary", "tertiary", "university", "leisure"}
_learning_pref_cache: Dict[str, Tuple[float, str]] = {}
_LEARNING_PREF_TTL_SEC = float(os.getenv("LEARNING_PREF_TTL_SEC", "300"))


def _normalize_learning_pref(pref: Optional[str]) -> Optional[str]:
    p = (pref or "").strip().lower()
    return p if p in _LEARNING_PREF_ALLOWED else None


async def fetch_learning_preference(user_id: Optional[str]) -> Optional[str]:
    """Fetch user's learning_preference from Supabase (fail-soft + cached)."""
    uid = (user_id or "").strip()
    if not uid:
        return None
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
        return None

    now = time.perf_counter()
    cached = _learning_pref_cache.get(uid)
    if cached and (now - cached[0] <= _LEARNING_PREF_TTL_SEC):
        return cached[1]

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{SUPABASE_URL}/rest/v1/profiles?id=eq.{uid}&select=learning_preference&limit=1"
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(url, headers=headers)
        if r.status_code >= 400:
            return None
        rows = r.json() or []
        row = rows[0] if rows and isinstance(rows[0], dict) else {}
        pref = _normalize_learning_pref(row.get("learning_preference"))
        if pref:
            _learning_pref_cache[uid] = (now, pref)
        return pref
    except Exception:
        return None


def build_learning_preference_instruction(pref: Optional[str]) -> str:
    """Return system-instruction text that shapes depth/structure based on preference."""
    p = _normalize_learning_pref(pref)
    if not p:
        return ""

    if p == "secondary":
        return (
            "LEARNING PREFERENCE: Secondary.\n"
            "- Explain in simple terms, avoid jargon, and define any necessary terms.\n"
            "- Use short paragraphs and step-by-step guidance.\n"
            "- Give 1–2 concrete examples; keep math/light formalism minimal unless asked.\n"
        )
    if p == "tertiary":
        return (
            "LEARNING PREFERENCE: Tertiary.\n"
            "- Use moderately technical language and introduce key terms with brief definitions.\n"
            "- Prefer structured explanations (steps, bullets) and include practical examples.\n"
            "- Include important details, but avoid overly long proofs/derivations unless asked.\n"
        )
    if p == "university":
        return (
            "LEARNING PREFERENCE: University.\n"
            "- Provide a rigorous, higher-level explanation with assumptions and precise terminology.\n"
            "- When useful, include deeper reasoning, edge cases, and (optional) equations/derivations.\n"
            "- Keep structure clear: definitions → intuition → details → example(s) or algorithm.\n"
        )
    # leisure
    return (
        "LEARNING PREFERENCE: Leisure Learning.\n"
        "- Keep it friendly and intuitive; focus on big-picture understanding.\n"
        "- Use simple examples and avoid heavy technical depth unless asked.\n"
    )


def build_emoji_style_instruction(style: Optional[str]) -> str:
    """Return a short instruction to control emoji usage in model outputs.

    This is especially important for RunPod, where your serverless handler wraps
    `input.prompt` as the user message and uses a fixed system prompt.
    """
    s = (style or "").strip().lower()
    if s == "off":
        return "EMOJI STYLE: Off. Do not use emojis."
    if s == "strong":
        return (
            "EMOJI STYLE: Strong. Use a few relevant emojis sparingly (about 1–3 total) "
            "to make the response engaging. Avoid emojis in code blocks, JSON, or formal/academic responses."
        )
    # Default/light
    return (
        "EMOJI STYLE: Light. You may use at most one light emoji in an opening or closing when helpful. "
        "Skip emojis for formal, code-heavy, or JSON-only responses."
    )


def build_backend_system_prompt(
    *,
    chat_mode: bool,
    learning_preference: Optional[str] = None,
) -> str:
    """Backend-owned system prompt.

    IMPORTANT: This is used when sending a full Llama-Instruct template to RunPod.
    The RunPod handler will *not* wrap prompts that start with <|begin_of_text|>,
    so this prompt becomes the effective system behavior.
    """

    pref_instruction = build_learning_preference_instruction(learning_preference).strip()
    emoji_instruction = build_emoji_style_instruction(AV_EMOJI_STYLE).strip()

    # Shared behavioral guidance.
    base = (
        "You are AskVox, the assistant inside the AskVox app. "
        "Explain topics in a natural, human, tutor-like way. "
        "Prefer clear paragraphs with context, reasoning, and examples. "
        "Avoid giant wall-of-text paragraphs. "
        "If a line is in the form 'Title - details', render it as '**Title** — details'.\n"
        "Do not mention training data, model cutoffs, or internal implementation details.\n"
    )

    # Chat-mode is for the best human answer; JSON/structured mode is for tool routing + evidence.
    if chat_mode:
        chat_specific = (
            "Use bullet points or numbered lists only when they genuinely improve clarity "
            "(rankings, comparisons, step-by-step instructions). "
            "When providing steps, use a numbered list with ONE item per line and keep each step concise.\n"
            "Optionally include one short closing line inviting follow-up; avoid boilerplate and omit closings for long/formal answers.\n"
        )
        parts = [base]
        if pref_instruction:
            parts.append(pref_instruction)
        if emoji_instruction:
            parts.append(emoji_instruction)
        parts.append(chat_specific)
        return "\n".join([p for p in parts if p]).strip()

    # Structured mode: enforce JSON contract.
    parts = [base]
    if pref_instruction:
        parts.append(pref_instruction)
    if emoji_instruction:
        parts.append(emoji_instruction)
    parts.append(MODEL_JSON_INSTRUCTION.strip())
    return "\n\n".join([p for p in parts if p]).strip()


def build_runpod_user_prompt(
    message: str,
    history: List[HistoryItem],
    rag_block: str = "",
    web_evidence_block: str = "",
    article_block: str = "",
    chat_mode: bool = False,
    learning_preference: Optional[str] = None,
) -> str:
    """Build a RunPod prompt.

    This workspace's RunPod handler bypasses wrapping when the prompt begins with
    `<|begin_of_text|>`. Since you want the backend to own the system prompt,
    we always send a full Llama-Instruct formatted prompt here.
    """
    def _truncate(text: str, limit: int) -> str:
        if not text:
            return ""
        return text if len(text) <= limit else text[:limit] + "..."

    safe_message = _truncate(message, MAX_MESSAGE_CHARS)
    safe_article = _truncate(article_block, MAX_ARTICLE_CHARS)
    safe_rag = _truncate(rag_block, MAX_RAG_CHARS)
    safe_evidence = _truncate(web_evidence_block, MAX_EVIDENCE_CHARS)

    system_text = build_backend_system_prompt(chat_mode=chat_mode, learning_preference=learning_preference)

    p = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_text}<|eot_id|>"
    )

    # Context blocks (best effort). For chat_mode, include them in the user message.
    user_context_parts: List[str] = []
    if safe_article:
        user_context_parts.append("[ARTICLE_CONTEXT] (use when relevant):")
        user_context_parts.append(safe_article)
    if safe_rag:
        user_context_parts.append(safe_rag)
    if safe_evidence:
        user_context_parts.append(safe_evidence)

    # Compact recent history, avoid duplicates.
    filtered_history: List[HistoryItem] = []
    seen = set()
    for h in history:
        content = (getattr(h, "content", "") or "").strip()
        role = getattr(h, "role", None)
        if not content or role not in ("user", "assistant"):
            continue
        # Avoid duplicating the current user message if it was already included.
        if role == "user" and content == (safe_message or "").strip():
            continue
        key = (role, content)
        if key in seen:
            continue
        seen.add(key)
        filtered_history.append(HistoryItem(role=role, content=content))
    filtered_history = filtered_history[-4:]

    remaining_history_chars = MAX_HISTORY_CHARS
    for h in filtered_history:
        if remaining_history_chars <= 0:
            break
        role = "user" if h.role == "user" else "assistant"
        content = _truncate(h.content, 240)
        if len(content) > remaining_history_chars:
            content = _truncate(content, max(0, remaining_history_chars))
        remaining_history_chars -= len(content)
        p += (
            f"<|start_header_id|>{role}<|end_header_id|>\n"
            f"{content}<|eot_id|>"
        )

    # User turn
    user_payload_parts: List[str] = []
    if user_context_parts:
        user_payload_parts.append("\n".join([x for x in user_context_parts if x]).strip())
    user_payload_parts.append(safe_message)
    user_payload = "\n\n".join([x for x in user_payload_parts if x]).strip()

    p += (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_payload}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    if len(p) > MAX_PROMPT_CHARS:
        p = p[:MAX_PROMPT_CHARS] + "..."
    return p

CITATION_TOKEN_RULES = """
CITATIONS RULES (VERY IMPORTANT):
- When you use a fact from the provided evidence, add a citation token immediately after the sentence, like: [[cite:1]]
- Only cite from the sources provided in [SOURCES] below.
- Do NOT use [1] style. ONLY use [[cite:1]] tokens.
- If you are not sure / not supported by evidence, say so and do not cite.
"""

MODEL_JSON_INSTRUCTION = (
    "You are AskVox, the assistant inside the AskVox app.\n"
    "If the user asks about you (e.g., 'tell me about yourself'), start with \"I'm AskVox\" and give a short, friendly 2–4 sentence introduction focused on what you can do for them. "
    "Do not describe yourself as \"an AI language model\" and do not mention training data/corpus or internal implementation details.\n\n"
    "When appropriate, respond using a VALID JSON object matching the schema below.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT rewrite, summarize, shorten, or rephrase the answer.\n"
        "- Preserve ALL tone, emojis, formatting, markdown, lists, and wording exactly.\n"
        "- Simply PLACE the answer inside \"answer_markdown\".\n"
        "- No text before or after JSON.\n\n"
        "Schema:\n"
        "{\n"
        "  \"answer_markdown\": \"string\",\n"
        "  \"need_web_sources\": true/false,\n"
        "  \"need_images\": true/false,\n"
        "  \"need_youtube\": true/false,\n\n"
        "  \"web_query\": \"string (short query if need_web_sources)\",\n"
        "  \"image_query\": \"string (short query if need_images)\",\n"
        "  \"youtube_query\": \"string (short query if need_youtube)\"\n"
        "}\n\n"
        "Additional guidance:\n"
        "- answer_markdown is the final answer the user sees.\n"
        "- Apply these formatting rules:\n"
        + FORMAT_INSTRUCTION
        + "\n\n"
        + CITATION_TOKEN_RULES
        + "\n\n"
        "- If need_web_sources=false then web_query must be \"\" (same for image/youtube).\n"
        "- Do not invent citations. Only cite if evidence exists.\n"
        "- If the user includes a specific year (e.g., 2026), the web_query MUST include that year when relevant.\n"
)

# -----------------------
# Second-pass (sanity) prompt builder — non-JSON
# -----------------------
def build_second_pass_prompt_chat(
    message: str,
    history: List[HistoryItem],
    original_answer: str,
    web_evidence_block: str = "",
    article_block: str = "",
    learning_preference: Optional[str] = None,
) -> str:
    """Construct a chat-style prompt to REWRITE/UPDATE the original answer using fresh web evidence.

    This prompt outputs plain markdown (no JSON), suitable for LLaMA chat mode.
    """
    def _truncate(text: str, limit: int) -> str:
        if not text:
            return ""
        return text if len(text) <= limit else text[:limit] + "..."

    safe_msg = _truncate(message, MAX_MESSAGE_CHARS)
    safe_orig = _truncate(original_answer, MAX_ARTICLE_CHARS)
    safe_article = _truncate(article_block, MAX_ARTICLE_CHARS)
    safe_evidence = _truncate(web_evidence_block, MAX_EVIDENCE_CHARS)
    pref_instruction = build_learning_preference_instruction(learning_preference).strip()

    tone_lines = ""
    if AV_EMOJI_STYLE != "off":
        tone_lines += (
            "Adopt a warm, friendly tone. Use at most one light emoji in openings/closings when helpful; skip emojis for formal or long answers.\n"
        )
    if AV_FRIENDLY_OPENERS:
        tone_lines += (
            "For casual questions, a brief, varied greeting is okay; for academic questions, start directly.\n"
        )
    tone_lines += (
        "Optionally include a short closing inviting follow-up; avoid boilerplate; omit for formal or long answers.\n"
    )

    p = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are AskVox, a helpful assistant. Update and correct the original answer using the latest facts from the provided sources. "
        "Your internal knowledge may be outdated beyond 2023; when sources conflict with prior knowledge, trust the latest [SOURCES]. "
        "Do not mention training cutoffs or model limitations in the answer. "
        "Write a complete, well-structured markdown response (NOT JSON). Prefer concise paragraphs; use lists only when they improve clarity. "
        "If a line is 'Title - details', render as '**Title** — details'. Do not include any [SOURCES] section at the end.\n"
        + (pref_instruction + "\n" if pref_instruction else "")
        + tone_lines +
        "<|eot_id|>"
    )

    # Minimal recent history (up to 3 entries)
    filtered_history: List[HistoryItem] = []
    seen = set()
    for h in history[-3:]:
        if not h.content or not h.content.strip():
            continue
        key = (h.role, h.content.strip())
        if key in seen:
            continue
        seen.add(key)
        filtered_history.append(h)

    for h in filtered_history:
        role = "user" if h.role == "user" else "assistant"
        p += (
            f"<|start_header_id|>{role}<|end_header_id|>\n" + _truncate(h.content, 240) + "<|eot_id|>"
        )

    # Current user message
    p += (
        "<|start_header_id|>user<|end_header_id|>\n" + safe_msg + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    # Context blocks
    if safe_article:
        p += "[ARTICLE_CONTEXT]\n" + safe_article + "\n\n"
    if safe_orig:
        p += "[ORIGINAL_ANSWER]\n" + safe_orig + "\n\n"
    if safe_evidence:
        p += safe_evidence + "\n\n"

    # Revision rules (inline, non-JSON)
    p += (
        "[REVISION_RULES]\n"
        "- Integrate corrections and the latest facts from [SOURCES]; prefer them on conflicts.\n"
        "- Produce a complete, informative answer (do NOT shorten/paraphrase excessively); match or exceed original detail.\n"
        "- Preserve friendly tone and formatting; where appropriate, add [[cite:N]] after sentences supported by [SOURCES].\n"
        "- Output plain markdown only; no JSON.\n"
    )

    return p[:MAX_PROMPT_CHARS] if len(p) > MAX_PROMPT_CHARS else p

# -----------------------
# Sanity web query builder — non-JSON
# -----------------------
def build_sanity_web_query(
    message: str,
    original_answer: str,
    context_query_base: Optional[str] = None,
) -> str:
    """Derive a concise web query for sanity checking (non-JSON), without touching plan.web_query.

    - Prefer context_query_base, else user message
    - Remove media verbs (show me, images, videos, etc.)
    - If no explicit year is present and query implies recency, append the current year
    """
    base = (context_query_base or message or "").strip()
    q = re.sub(
        r"\b(show me|give me|find|search|look for|video|videos|youtube|yt|image|images|picture|pictures|photo|photos)\b",
        "",
        base,
        flags=re.I,
    )
    years = re.findall(r"\b(20\d{2})\b", (message or "") + " " + (original_answer or ""))
    q = re.sub(r"\s+", " ", q).strip()
    current_year = datetime.now(timezone.utc).year
    recency_intent = bool(re.search(r"\b(now|currently|today|this year|latest|trending|airing|released|premiered)\b", (message or ""), re.I))
    prefer_year = None
    if years:
        try:
            prefer_year = max(int(y) for y in years if y.isdigit())
        except Exception:
            prefer_year = None
    elif recency_intent:
        prefer_year = current_year
    if prefer_year and str(prefer_year) not in q:
        q = f"{q} {prefer_year}".strip()
    return q[:120]

# -----------------------
# PERFORMANCE: lightweight in-memory caches and time budgets
# -----------------------
_CACHE_TTL_SEC = 300  # 5 minutes
_cache_google: Dict[Tuple[str, int], Tuple[float, List['SourceItem']]] = {}
_cache_tavily: Dict[Tuple[str, int], Tuple[float, Tuple[List['SourceItem'], List[Dict[str, str]]]]] = {}
_cache_images: Dict[Tuple[str, int], Tuple[float, List[Dict[str, str]]]] = {}
_cache_youtube: Dict[Tuple[str, int], Tuple[float, List['YouTubeItem']]] = {}

async def fast_google_web_search(query: str, num: int = 6, timeout_sec: float = 3.5, date_restrict: Optional[str] = None) -> List['SourceItem']:
    key = (query or "", int(num))
    now = time.perf_counter()
    cached = _cache_google.get(key)
    if cached and (now - cached[0] <= _CACHE_TTL_SEC):
        return cached[1]
    try:
        res = await asyncio.wait_for(google_web_search(query, num=num, date_restrict=date_restrict), timeout=timeout_sec)
    except Exception:
        res = []
    _cache_google[key] = (now, res)
    return res

async def fast_tavily(query: str, max_sources: int = 6, timeout_sec: float = 4.5) -> Tuple[List['SourceItem'], List[Dict[str, str]]]:
    key = (query or "", int(max_sources))
    now = time.perf_counter()
    cached = _cache_tavily.get(key)
    if cached and (now - cached[0] <= _CACHE_TTL_SEC):
        return cached[1]
    try:
        res = await asyncio.wait_for(internet_rag_search_and_extract(query, max_sources=max_sources), timeout=timeout_sec)
    except Exception:
        res = ([], [])
    _cache_tavily[key] = (now, res)
    return res

async def fast_images(query: str, num: int = 4, timeout_sec: float = 3.0) -> List[Dict[str, str]]:
    key = (query or "", int(num))
    now = time.perf_counter()
    cached = _cache_images.get(key)
    if cached and (now - cached[0] <= _CACHE_TTL_SEC):
        return cached[1]
    try:
        res = await asyncio.wait_for(google_image_search(query, num=num), timeout=timeout_sec)
    except Exception:
        res = []
    _cache_images[key] = (now, res)
    return res

async def fast_youtube(query: str, num: int = 2, timeout_sec: float = 3.0) -> List['YouTubeItem']:
    key = (query or "", int(num))
    now = time.perf_counter()
    cached = _cache_youtube.get(key)
    if cached and (now - cached[0] <= _CACHE_TTL_SEC):
        return cached[1]
    try:
        res = await asyncio.wait_for(youtube_search(query, num=num), timeout=timeout_sec)
    except Exception:
        res = []
    _cache_youtube[key] = (now, res)
    return res

# Overall time budget for a single multimodal request (soft)
_TOTAL_BUDGET_SEC = float(os.getenv("MM_TOTAL_BUDGET_SEC", "9.0"))
_BUDGET_WEB_MIN_REMAIN_SEC = float(os.getenv("MM_BUDGET_WEB_MIN_REMAIN_SEC", "1.8"))
_BUDGET_WEB_SECOND_PASS_MIN_REMAIN_SEC = float(os.getenv("MM_BUDGET_WEB_SECOND_PASS_MIN_REMAIN_SEC", "3.0"))
_BUDGET_MEDIA_MIN_REMAIN_SEC = float(os.getenv("MM_BUDGET_MEDIA_MIN_REMAIN_SEC", "2.2"))

# YouTube result sizing (hard cap + default returned count)
_YOUTUBE_MAX_RESULTS = max(1, min(int(os.getenv("MM_YOUTUBE_MAX_RESULTS", "5")), 10))
_YOUTUBE_DEFAULT_RESULTS = max(1, min(int(os.getenv("MM_YOUTUBE_DEFAULT_RESULTS", "2")), _YOUTUBE_MAX_RESULTS))

# -----------------------
# DATA MODELS
# -----------------------
class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []
    query_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    domain: Optional[str] = None
    article_title: Optional[str] = None
    article_url: Optional[str] = None

class SourceItem(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    icon_url: Optional[str] = None

class ImageItem(BaseModel):
    url: str
    storage_path: Optional[str] = None
    alt: Optional[str] = None
    source_url: Optional[str] = None

class YouTubeItem(BaseModel):
    title: str
    url: str
    video_id: str
    channel: Optional[str] = None
    thumbnail_url: Optional[str] = None
    embeddable: Optional[bool] = None

class AssistantPayload(BaseModel):
    answer_markdown: str
    sources: List[SourceItem] = Field(default_factory=list)
    images: List[ImageItem] = Field(default_factory=list)
    youtube: List[YouTubeItem] = Field(default_factory=list)
    # UI helpers
    source_count: int = 0
    cite_available: bool = False
    cite_label: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    payload: AssistantPayload

# -----------------------
# Option A: Keyword Router + Option C: Escalation + Tool Budget
# -----------------------
# NOTE: avoid overly generic triggers like "how to" which match many normal questions.
_MEDIA_WORDS_RE = re.compile(r"\b(video|videos|youtube|yt|watch|tutorial|guide|how to|walkthrough)\b", re.I)
_IMAGE_WORDS_RE = re.compile(r"\b(image|images|picture|pictures|photo|photos|png|jpg|jpeg|sticker|diagram|infographic|show me)\b", re.I)

# ✅ UPDATED: web triggers are more "explicit web intent" + year-based "released in 2026"
# - remove "now" and the generic "202[0-9]" trigger because it causes accidental web routing
_WEB_WORDS_RE = re.compile(
    r"\b(latest|news|today|current|update|updated|recent|price|release|released|schedule|announcement|"
    r"source|sources|link|links|cite|citation|according to|resources|reference)\b",
    re.I,
)

# ✅ NEW: "year request" signals a time-sensitive lookup (e.g., "released in 2026")
_YEAR_WEB_INTENT_RE = re.compile(r"\b(released|release|airing|air|premiere|premiered|new|best)\b.*\b(20\d{2})\b", re.I)

_UNCERTAIN_RE = re.compile(
    r"\b(i (don't|do not) know|not sure|can't verify|cannot verify|unsure|might be|may be|could be|depends|"
    r"check online|look it up|search|verify|as of|recently)\b",
    re.I,
)

_DEFLECTION_RE = re.compile(
    r"\b(i can't|i cannot|i'm unable to|i am unable to|don't have access|do not have access|"
    r"i don't have (real\s*-?time|realtime) data|i cannot browse|i can't browse|"
    r"as an ai|i'm just an ai|cannot guarantee|no guarantees)\b",
    re.I,
)

_FACT_SEEKING_RE = re.compile(
    r"\b(who|what|when|where|which)\b|"
    r"\b(how many|how much)\b|"
    r"\b(compare|vs\.?|difference between)\b",
    re.I,
)

_YEAR_RE = re.compile(r"\b(20\d{2})\b")

def extract_years(text: str) -> List[str]:
    return _YEAR_RE.findall(text or "")

def keyword_router(message: str) -> Dict[str, bool]:
    msg = (message or "").strip()
    want_web = bool(_WEB_WORDS_RE.search(msg)) or bool(_YEAR_WEB_INTENT_RE.search(msg))
    return {
        "want_youtube": bool(_MEDIA_WORDS_RE.search(msg)),
        "want_images": bool(_IMAGE_WORDS_RE.search(msg)),
        "want_web": want_web,
    }

# ✅ NEW: smalltalk/identity detection to avoid pointless web sources
_SMALLTALK_RE = re.compile(
    r"(\bhi\b|\bhello\b|\bhey\b|\bhiya\b|\bsup\b|\bhow are you\b|\bthanks\b|\bthank you\b|\bbye\b|\bgoodbye\b|"
    r"\bwho are you\b|\bwhat are you\b|\bintroduce yourself\b|\bwhat can you do\b|\bwho am i\b)",
    re.I,
)

def is_smalltalk_or_identity(message: str) -> bool:
    msg = (message or "").strip()
    if not msg:
        return False
    return bool(_SMALLTALK_RE.search(msg))

# Follow-up intents like "elaborate more" / "apart from that" that
# suggest the user wants additional information beyond what was already given.
_MORE_DETAIL_FOLLOWUP_RE = re.compile(
    r"\b(elaborate|eloborate|more detail|more details|tell me more|expand on|apart from that|other than that|anything else)\b",
    re.I,
)

# ✅ FIX: Empty answer is NOT a signal that "web is needed"
def looks_like_needs_web(answer_md: str) -> bool:
    if not answer_md or not answer_md.strip():
        return False
    return bool(_UNCERTAIN_RE.search(answer_md))

def looks_like_deflection_or_nonanswer(answer_md: str) -> bool:
    if not answer_md or not answer_md.strip():
        return True
    text = answer_md.strip()
    if len(text) < 120:
        return True
    return bool(_DEFLECTION_RE.search(text))

def is_fact_seeking_question(message: str) -> bool:
    msg = (message or "").strip()
    if not msg:
        return False
    # Avoid escalating purely general how-to/advice prompts.
    if is_general_advice_question(msg) and not _FACTUAL_WEB_INTENT_RE.search(msg):
        return False
    return bool(_FACT_SEEKING_RE.search(msg) or _FACTUAL_WEB_INTENT_RE.search(msg))

_FACTUAL_WEB_INTENT_RE = re.compile(
    r"\b(as of|current|latest|updated|recent|today|news)\b|"
    r"\b(price|cost|salary|net\s*worth|market\s*cap|population|gdp|revenue|statistics|statistic|percent|%|how many|how much|number of)\b|"
    r"\b(is it true|true that|evidence|proof|study|research|paper)\b|"
    r"\b(released|release date|premiere|premiered|announced|launch|deadline|schedule)\b|"
    r"\b(20\d{2})\b",
    re.I,
)

_GENERAL_ADVICE_RE = re.compile(
    r"\b(how do i|how to|tips|advice|learn|improve|practice|study|start|begin|roadmap|plan)\b",
    re.I,
)

_LEARNING_REQUEST_RE = re.compile(
    r"\b(teach me|step\s*-?by\s*-?step|walk me through|tutorial|guide|documentation|docs|resources|course|learn)\b",
    re.I,
)

_WHO_IS_RE = re.compile(
    r"^\s*(who\s+is|who's|whos|who\s+was|tell\s+me\s+about)\s+(.+?)\s*[?!.]*\s*$",
    re.I,
)

def extract_who_is_subject(message: str) -> str:
    m = _WHO_IS_RE.match((message or "").strip())
    if not m:
        return ""
    subj = (m.group(2) or "").strip()
    subj = re.sub(r"\s+", " ", subj).strip()
    return subj[:120]

def is_who_is_question(message: str) -> bool:
    return bool(_WHO_IS_RE.match((message or "").strip()))

def _has_web_providers() -> bool:
    return bool((GOOGLE_API_KEY and GOOGLE_CSE_ID) or TAVILY_API_KEY)

def is_general_advice_question(message: str) -> bool:
    msg = (message or "").strip()
    if not msg:
        return False
    return bool(_GENERAL_ADVICE_RE.search(msg))

def wants_learning_resources(message: str) -> bool:
    msg = (message or "").strip()
    if not msg:
        return False
    return bool(_LEARNING_REQUEST_RE.search(msg))

def infer_need_web_sources(
    message: str,
    answer_md: str,
    user_flags: Dict[str, bool],
    model_need_web: bool,
    elapsed_sec: Optional[float] = None,
) -> Tuple[bool, str]:
    """Decide whether to fetch web sources.

    Returns: (need_web, reason)
    """
    if not _has_web_providers():
        return False, "no_providers"

    if is_smalltalk_or_identity(message):
        return False, "smalltalk_or_identity"

    # Default policy for this app: always provide web sources when providers exist.
    # Budget is enforced later in the pipeline (degrade gracefully).
    if FORCE_WEB_SOURCES:
        return True, "forced"

    if user_flags.get("want_web"):
        return True, "explicit_web_intent"

    # Option C: for learning prompts, fetch sources proactively.
    # This is treated as higher priority than the soft budget guard so that
    # educational requests like tutorials still return references/links.
    if wants_learning_resources(message):
        return True, "learning_resources"

    if model_need_web:
        return True, "model_suggested"

    if looks_like_needs_web(answer_md):
        return True, "model_uncertain"

    # If the draft answer is likely a deflection/non-answer AND the user asked a fact-seeking question,
    # escalate to web even if the model didn't explicitly say it needs web.
    if looks_like_deflection_or_nonanswer(answer_md) and is_fact_seeking_question(message):
        return True, "draft_nonanswer_fact_seeking"

    msg = (message or "").strip()
    if msg and is_fact_seeking_question(msg) and not is_general_advice_question(msg):
        return True, "factual_or_time_sensitive"

    return False, "not_needed"

def make_fallback_query(message: str, max_len: int = 120) -> str:
    if not message:
        return ""
    q = message.strip()
    q = re.sub(
        r"\b(show me|give me|find|search|look for|video|videos|youtube|yt|image|images|picture|pictures|photo|photos)\b",
        "",
        q,
        flags=re.I,
    )
    q = re.sub(r"\s+", " ", q).strip()
    if q.lower() in {"and", "or"}:
        q = ""
    return (q[:max_len] if q else message[:max_len])

def enforce_web_query_constraints(user_message: str, web_q: str) -> str:
    
    msg = (user_message or "").strip()
    q = (web_q or "").strip()

    years = extract_years(msg)
   

    # If model gave nothing, use user message
    if not q:
        q = msg[:120]

    # If user specified year(s), enforce year presence
    if years:
        if not any(y in q for y in years):
            q = f"{q} {years[0]}"

     
    q = re.sub(r"\s+", " ", q).strip()
    return q[:120]

def apply_tool_budget(
    user_flags: Dict[str, bool],
    need_web: bool,
    need_img: bool,
    need_yt: bool,
    is_first_turn: bool = False,
) -> Tuple[bool, bool, bool]:
    # On the very first turn, don't override media flags here;
    # higher-level logic in generate_cloud_structured controls behavior.
    if not is_first_turn:
        # Media only when user explicitly asked OR forced by env
        if not (user_flags.get("want_images") or user_flags.get("auto_images") or FORCE_IMAGES):
            need_img = False
        # Treat auto_youtube as a valid intent for learning prompts.
        if not (user_flags.get("want_youtube") or user_flags.get("auto_youtube") or FORCE_YOUTUBE):
            need_yt = False

    # Web is handled by "hard gate" policy in caller
    return need_web, need_img, need_yt

def normalize_markdown_spacing(text: str) -> str:
    if not text:
        return ""
    # Preserve paragraph separation: collapse 3+ blank lines into TWO, keep double newlines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Optionally keep the blank line between a title and the following paragraph
    keep_title_spacing = os.getenv("AV_KEEP_TITLE_SPACING", "1") == "1"
    if not keep_title_spacing:
        # Remove blank line between title + description (title line followed by blank, then capitalized desc)
        text = re.sub(r"([^\n])\n\n([A-Z])", r"\1\n\2", text)
    return text.strip()

def _looks_structured_answer(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    # Basic signals: multiple paragraphs or a clear list/steps.
    if s.count("\n\n") >= 2:
        return True
    if re.search(r"^\s*\d+\.|^\s*[-*•]\s+", s, flags=re.MULTILINE):
        return True
    if re.search(r"\b(step-by-step|steps|guide|roadmap)\b", s, flags=re.I):
        return True
    return False

def choose_final_answer(draft_md: str, refined_md: str) -> str:
    """Pick the best user-visible answer when we have both draft + refined outputs.

    The refined pass is mainly for web-evidence integration and tool routing.
    Sometimes it comes back shorter/less helpful; in that case keep the richer draft.
    """
    draft = cleanup_model_text(strip_plan_json_leak(draft_md))
    refined = cleanup_model_text(strip_plan_json_leak(refined_md))
    if refined and not draft:
        return refined.strip()
    if draft and not refined:
        return draft.strip()
    if not draft and not refined:
        return ""

    d_len = len(draft.strip())
    r_len = len(refined.strip())

    # If refined is significantly shorter and the draft looks structured, prefer the draft.
    if _looks_structured_answer(draft) and (r_len < max(220, int(d_len * 0.60))):
        return draft.strip()

    # Otherwise prefer refined (it may incorporate web corrections).
    return refined.strip()

def enhance_markdown_for_ui(text: str) -> str:
    """
    Light post-processing to improve readability in the UI:
    - Bold lines like "Title - details" -> "**Title** — details" (em dash)
    - After a heading ending with ":", convert following standalone lines into markdown bullets
      until another heading/section starts. Avoid double-bulleting existing list items.
    """
    if not text:
        return ""

    lines = text.splitlines()
    out_lines: List[str] = []
    in_list_after_colon = False
    plain_list_mode = os.getenv("AV_PLAIN_LISTS", "1") == "1"  # render items as plain lines (no bullet markers)

    def is_list_item(s: str) -> bool:
        return bool(re.match(r"^\s*([-*•]|\d+\.|\d+\))\s+", s))

    for i, line in enumerate(lines):
        s = line.strip()

        # Start or stop list mode based on headings
        if s.endswith(":"):
            in_list_after_colon = True
            out_lines.append(line)
            continue
        if not s:
            out_lines.append(line)
            continue

        # Normalize leading '*' bullets to '-' for consistent markdown lists
        if re.match(r"^\s*\*+\s+", line):
            line = re.sub(r"^\s*\*+\s+", "- ", line)
            s = line.strip()

        # Bold "Title - details" only when it precedes a list or within a list section
        if not is_list_item(s):
            m = re.match(r"^([A-Za-z0-9][^\-\n:]{0,80})\s*[-–—]\s+(.*)$", s)
            if m:
                apply_bold = in_list_after_colon
                if not apply_bold:
                    # Peek the next non-empty line; if it's a list item, then treat this as a title line
                    next_non_empty = ""
                    for j in range(i + 1, len(lines)):
                        nn = (lines[j] or "").strip()
                        if nn:
                            next_non_empty = nn
                            break
                    if is_list_item(next_non_empty):
                        apply_bold = True
                if apply_bold:
                    line = f"**{m.group(1).strip()}** — {m.group(2).strip()}"
                    s = line

        # Convert to list item or plain line if we're in a post-colon list section
        if in_list_after_colon and not is_list_item(s):
            # Heuristic: treat reasonably short standalone lines as items
            if len(s) <= 160:
                if plain_list_mode:
                    # As plain paragraph item (no bullet marker), keep a bit of breathing room
                    if out_lines and out_lines[-1].strip():
                        out_lines.append("")
                    line = s
                else:
                    line = f"- {s}"
            else:
                # Long line likely ends the list section
                in_list_after_colon = False

        # If we encounter another heading, end list mode
        if s.endswith(":"):
            in_list_after_colon = True

        out_lines.append(line)

    return "\n".join(out_lines).strip()
    

# -----------------------
# Moderation
# -----------------------
async def moderation_check(text: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    return await moderate_message(text, user_id=user_id, session_id=session_id)
# -----------------------
# Domain RAG 
# -----------------------
async def rag_retrieve(query: str, k: int = 4, classified_domain: Optional[str] = None) -> List[Dict[str, str]]:
    """Retrieve RAG chunks if domain is RAG-enabled, else return [].

    This is the ONLY place the domain classifier is called for RAG gating.
    If classified_domain is not passed, we classify here.
    """
    if not is_rag_available():
        return []

    # Classify if not already done
    if not classified_domain:
        try:
            classified_domain = classify_domain(query)
        except Exception:
            return []

    if not is_rag_domain(classified_domain):
        return []

    try:
        chunks = await retrieve_rag_context(query, classified_domain)
        if chunks:
            print(
                f"[RAG] injecting {len(chunks)} chunks for domain='{classified_domain}'",
                flush=True,
            )
        return chunks
    except Exception as e:
        print(f"[RAG] retrieve error (fail-soft): {e}", flush=True)
        return []

def build_rag_block(chunks: List[Dict[str, str]]) -> str:
    if not chunks:
        return ""
    lines = ["[RAG_CONTEXT] Use this context if relevant (do not mention this label):"]
    for i, ch in enumerate(chunks[:6], start=1):
        title = ch.get("title") or f"Chunk {i}"
        src = ch.get("source") or ""
        content = (ch.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"- ({i}) {title} {f'[{src}]' if src else ''}\n  {content}")
    return "\n".join(lines).strip()

# -----------------------
# Internet RAG extraction (Tavily)
# -----------------------
# NOTE: internet_rag_search_and_extract() and build_web_evidence_block()
# have been moved to app/services/internet_rag_service.py for better modularity.
# They are now imported at the top of this file.

# -----------------------
# Google web sources (CSE)
# -----------------------
async def google_web_search(query: str, num: int = 6, date_restrict: Optional[str] = None) -> List[SourceItem]:
    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID and query.strip()):
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": min(num, 10)}
    if date_restrict:
        params["dateRestrict"] = date_restrict
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        if r.status_code >= 400:
            print("WEB SEARCH ERROR:", r.status_code, r.text[:300], flush=True)
            return []
        data = r.json()

    out: List[SourceItem] = []
    for it in (data.get("items") or []):
        link = it.get("link") or ""
        if not link:
            continue
        out.append(SourceItem(
            title=it.get("title") or "Source",
            url=link,
            snippet=it.get("snippet"),
        ))
    return out

# -----------------------
# Google image search (CSE image)
# -----------------------
async def google_image_search(query: str, num: int = 4) -> List[Dict[str, str]]:
    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID and query.strip()):
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "searchType": "image",
        "num": min(num, 10),
        "safe": "active",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        if r.status_code >= 400:
            print("IMAGE SEARCH ERROR:", r.status_code, r.text[:300], flush=True)
            return []
        data = r.json()

    results: List[Dict[str, str]] = []
    for it in (data.get("items") or []):
        results.append({
            "image_url": it.get("link") or "",
            "page_url": (it.get("image", {}) or {}).get("contextLink") or "",
            "title": it.get("title") or "",
        })
    return [x for x in results if x.get("image_url")]

# -----------------------
# Helpers: displayable image filtering
# -----------------------
def _is_displayable_image_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower().strip()
    # Allow Pinterest CDN (pinimg.com) explicitly; block others from env list
    host = urlparse(u).netloc
    if host.endswith("pinimg.com"):
        pass  # allow Pinterest CDN images
    elif any(b in host for b in BLOCK_IMAGE_HOSTS):
        return False
    # Require a standard image extension
    if not re.search(r"\.(jpg|jpeg|png|webp)(\?.*)?$", u):
        return False
    return True

# -----------------------
# Supabase storage upload for images
# -----------------------
async def supabase_upload_image_from_url(image_url: str, filename_hint: str) -> Optional[ImageItem]:
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and image_url):
        return None

    async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
        img_resp = await client.get(image_url)
        if img_resp.status_code >= 400:
            return None
        content_type = img_resp.headers.get("content-type", "image/jpeg")
        img_bytes = img_resp.content

    ext = "jpg"
    if "png" in content_type:
        ext = "png"
    elif "webp" in content_type:
        ext = "webp"

    storage_path = f"{int(time.time())}_{re.sub(r'[^a-zA-Z0-9_-]+','_', filename_hint)[:40]}.{ext}"
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{storage_path}"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    async with httpx.AsyncClient(timeout=25) as client:
        up = await client.post(upload_url, headers=headers, content=img_bytes)
        if up.status_code >= 400:
            print("SUPABASE UPLOAD ERROR:", up.status_code, up.text[:300], flush=True)
            return None

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}/{storage_path}"
    return ImageItem(url=public_url, storage_path=storage_path, source_url=image_url)

# -----------------------
# YouTube search
# -----------------------
async def youtube_search(query: str, num: int = 2) -> List[YouTubeItem]:
    if not (YOUTUBE_API_KEY and query.strip()):
        return []
    # First, search videos by query
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max(2, min(num * 3, 10)),  # search a few more to find embeddable ones
        "safeSearch": "strict",
        "videoEmbeddable": "true",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(search_url, params=search_params)
        if r.status_code >= 400:
            print("YOUTUBE SEARCH ERROR:", r.status_code, r.text[:300], flush=True)
            return []
        search_data = r.json()

    ids: List[str] = []
    snippets: Dict[str, Dict[str, Any]] = {}
    for it in search_data.get("items", []):
        vid = (it.get("id") or {}).get("videoId")
        snip = it.get("snippet") or {}
        if not vid:
            continue
        ids.append(vid)
        snippets[vid] = snip
    if not ids:
        return []

    # Then, fetch embeddability status for these IDs
    videos_url = "https://www.googleapis.com/youtube/v3/videos"
    videos_params = {
        "key": YOUTUBE_API_KEY,
        "part": "status,snippet",
        "id": ",".join(ids[:50]),
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            vr = await client.get(videos_url, params=videos_params)
            if vr.status_code >= 400:
                print("YOUTUBE VIDEOS ERROR:", vr.status_code, vr.text[:300], flush=True)
                # Fallback: return basic results without embeddability filtering
                pass
            videos_data = vr.json() if vr.status_code < 400 else {"items": []}
    except Exception:
        videos_data = {"items": []}

    embeddable_ids: List[str] = []
    thumbnails: Dict[str, str] = {}
    titles: Dict[str, str] = {}
    channels: Dict[str, str] = {}
    for v in videos_data.get("items", []):
        vid = v.get("id")
        status = v.get("status") or {}
        snip = v.get("snippet") or {}
        if not vid:
            continue
        if status.get("embeddable", True):
            embeddable_ids.append(vid)
            thumb = ((snip.get("thumbnails") or {}).get("medium") or {}).get("url")
            thumbnails[vid] = thumb or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
            titles[vid] = (snip.get("title") or snippets.get(vid, {}).get("title") or "YouTube video")
            channels[vid] = (snip.get("channelTitle") or snippets.get(vid, {}).get("channelTitle") or None)

    # If no embeddable info returned, fallback to using snippets list but avoid known blocked content
    selected_ids = embeddable_ids or ids

    # Fallback: if fewer than requested, try a second search ordered by viewCount
    if len(selected_ids) < num:
        fallback_params = dict(search_params)
        fallback_params["order"] = "viewCount"
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r2 = await client.get(search_url, params=fallback_params)
                if r2.status_code < 400:
                    data2 = r2.json()
                    ids2: List[str] = []
                    snippets2: Dict[str, Dict[str, Any]] = {}
                    for it in data2.get("items", []):
                        vid2 = (it.get("id") or {}).get("videoId")
                        snip2 = it.get("snippet") or {}
                        if not vid2:
                            continue
                        ids2.append(vid2)
                        snippets2[vid2] = snip2
                    # Fetch statuses for new IDs
                    if ids2:
                        videos_params2 = {
                            "key": YOUTUBE_API_KEY,
                            "part": "status,snippet",
                            "id": ",".join(ids2[:50]),
                        }
                        async with httpx.AsyncClient(timeout=20) as client:
                            vr2 = await client.get(videos_url, params=videos_params2)
                            if vr2.status_code < 400:
                                vids2 = (vr2.json() or {}).get("items", [])
                                for v in vids2:
                                    vid = v.get("id")
                                    status = v.get("status") or {}
                                    snip = v.get("snippet") or {}
                                    if not vid:
                                        continue
                                    if status.get("embeddable", True):
                                        if vid not in selected_ids:
                                            selected_ids.append(vid)
                                        thumb = ((snip.get("thumbnails") or {}).get("medium") or {}).get("url")
                                        thumbnails[vid] = thumb or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                                        titles[vid] = (snip.get("title") or snippets2.get(vid, {}).get("title") or "YouTube video")
                                        channels[vid] = (snip.get("channelTitle") or snippets2.get(vid, {}).get("channelTitle") or None)
                                # Merge snippet fallbacks
                                for vid in ids2:
                                    if vid not in titles:
                                        titles[vid] = (snippets2.get(vid, {}) or {}).get("title") or "YouTube video"
                                    if vid not in channels:
                                        channels[vid] = (snippets2.get(vid, {}) or {}).get("channelTitle")
                                    if vid not in thumbnails:
                                        thumb = (((snippets2.get(vid, {}) or {}).get("thumbnails") or {}).get("medium") or {}).get("url")
                                        thumbnails[vid] = thumb or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
        except Exception:
            pass

    out: List[YouTubeItem] = []
    for vid in selected_ids[:max(2, num)]:
        title = titles.get(vid) or (snippets.get(vid, {}) or {}).get("title") or "YouTube video"
        channel = channels.get(vid) or (snippets.get(vid, {}) or {}).get("channelTitle")
        thumb = thumbnails.get(vid) or (((snippets.get(vid, {}) or {}).get("thumbnails") or {}).get("medium") or {}).get("url")
        if not thumb:
            thumb = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
        # Skip videos from blocked channels
        if channel and any(b.lower() in (channel or "").lower() for b in BLOCK_YT_CHANNELS):
            continue
        out.append(YouTubeItem(
            title=title,
            video_id=vid,
            url=f"https://www.youtube.com/watch?v={vid}",
            channel=channel,
            thumbnail_url=thumb,
            embeddable=True if vid in embeddable_ids else None,
        ))

    # Final guard (belt-and-suspenders) to ensure we never exceed max_results.
    return out

# -----------------------
# Helpers: extract JSON from model
# -----------------------
def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find the first balanced JSON object in the text
    s = str(text)
    n = len(s)
    for i in range(n):
        if s[i] == '{':
            depth = 0
            for j in range(i, n):
                if s[j] == '{':
                    depth += 1
                elif s[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = s[i:j+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break
    return {}

def strip_plan_json_leak(text: str) -> str:
    """Prevent the model's planning JSON from leaking into the visible answer.

    - If the text contains a plan-like JSON object with a non-empty `answer_markdown`, return that.
    - If the text is plan-like JSON (need_web_sources/web_query/etc.) but has no `answer_markdown`, return "" so
      downstream fallback messaging can kick in.
    - Otherwise, return the original text.
    """
    s = (text or "").strip()
    if not s:
        return ""

    # Fast heuristics to avoid touching normal prose.
    looks_jsonish = s.startswith("{") and s.endswith("}")
    has_plan_keys = any(
        k in s
        for k in (
            "need_web_sources",
            "need_images",
            "need_youtube",
            "web_query",
            "image_query",
            "youtube_query",
        )
    )
    has_answer_key = ("\"answer_markdown\"" in s) or ("'answer_markdown'" in s)
    if not (looks_jsonish or has_plan_keys or has_answer_key):
        return s

    # Best case: valid JSON somewhere in the output.
    try:
        obj = extract_json(s)
        if isinstance(obj, dict):
            ans = obj.get("answer_markdown")
            if isinstance(ans, str) and ans.strip():
                return ans.strip()
            # Plan-like JSON with no answer: suppress.
            if has_plan_keys and not ans:
                return ""
    except Exception:
        pass

    # Fallback: regex extraction of answer_markdown value from JSON-ish text.
    try:
        m = re.search(r"\"answer_markdown\"\s*:\s*\"(?P<v>(?:\\\\.|[^\"\\\\])*)\"", s, flags=re.S)
        if m:
            v = m.group("v")
            try:
                return json.loads('"' + v + '"').strip()
            except Exception:
                return v.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\").strip()

        m2 = re.search(r"'answer_markdown'\s*:\s*'(?P<v>(?:\\\\.|[^'\\\\])*)'", s, flags=re.S)
        if m2:
            v = m2.group("v")
            return v.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace("\\\\", "\\").strip()

        if has_plan_keys and not has_answer_key:
            return ""
    except Exception:
        pass

    return s

def safe_wrap_json(raw_text: str) -> Dict[str, Any]:
    rt = strip_plan_json_leak(raw_text)
    rt = cleanup_model_text((rt or "").strip())
    return {
        "answer_markdown": rt,
        "need_web_sources": False,
        "need_images": False,
        "need_youtube": False,
        "web_query": "",
        "image_query": "",
        "youtube_query": "",
    }

def cleanup_model_text(text: str) -> str:
    if not text:
        return ""
    # If the model echoed our JSON schema / plan, unwrap it.
    out = strip_plan_json_leak(text)
    # Remove any embedded schema/YAML-like lines the model may echo
    schema_keys = [
        r"^\s*answer_markdown\s*:\s*.*$",
        r"^\s*need_web_sources\s*:\s*.*$",
        r"^\s*need_images\s*:\s*.*$",
        r"^\s*need_youtube\s*:\s*.*$",
        r"^\s*web_query\s*:\s*.*$",
        r"^\s*image_query\s*:\s*.*$",
        r"^\s*youtube_query\s*:\s*.*$",
    ]
    for pat in schema_keys:
        out = re.sub(pat, "", out, flags=re.MULTILINE)
    # Remove inline citation tokens like [[cite:1]] or [[cite:4, 6]] (UI shows sources separately)
    out = re.sub(r"\[\[\s*cite\s*:\s*[^\]]+\]\]", "", out, flags=re.IGNORECASE)
    # Strip leaked special tokens (e.g., <|eot_id|>, <|start_header_id|>)
    out = re.sub(r"<\|.*?\|>", "", out)
    # Remove [USER] and [ASSISTANT] tags (model echoes)
    out = re.sub(r"^\s*\[(USER|ASSISTANT)\]\s*", "", out, flags=re.MULTILINE)
    # Remove all leaked internal labels and everything after them
    for label in [
        "[SOURCES]",
        "[EVIDENCE EXCERPTS]",
        "[EVIDENCE_EXCERPTS]",
        "[REVISION_RULES]",
        "[ARTICLE_CONTEXT]",
        "[RAG_CONTEXT]",
        "[ORIGINAL_ANSWER]",
    ]:
        m = re.search(r"\n" + re.escape(label), out, flags=re.IGNORECASE)
        if m:
            out = out[: m.start()]
    
    # Remove leaked instruction-style lines that sometimes appear at the start
    out = re.sub(
        r"^\s*(Remove or replace \[|Use a light touch|Focus on big-picture|Optionally include a short closing|Do not mention training).*$",
        "",
        out,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Normalize excessive blank lines
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out

def strip_meta_prompts(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"^💡?\s*Need web sources\??.*$",
        r"^\s*yes\s*$",
        r"^\s*no\s*$",
        r"^Sure!?\s*Here are some web sources.*$",
        r"^Here are some web sources.*$",
    ]

    out = text
    for p in patterns:
        out = re.sub(p, "", out, flags=re.IGNORECASE | re.MULTILINE)

    return re.sub(r"\n{3,}", "\n\n", out).strip()

# -----------------------
# Context extraction helpers for media follow-ups
# -----------------------
def extract_titles_from_answer(text: str) -> List[str]:
    """Extract likely title candidates from an assistant answer.
    Prefers bolded titles ("**Title** — details") and numbered list items
    that look like proper nouns.
    """
    titles: List[str] = []
    if not text:
        return titles

    for line in (text or "").splitlines():
        s = (line or "").strip()
        if not s:
            continue
        m_bold = re.match(r"^\*\*(.+?)\*\*\s+—\s+.*$", s)
        if m_bold:
            titles.append((m_bold.group(1) or "").strip())
            continue
        m_num = re.match(r"^\s*\d+[\.)]\s*([A-Z][A-Za-z0-9'&\- ]{2,})(?::|\s+-)\s+.*$", s)
        if m_num:
            titles.append((m_num.group(1) or "").strip())
            continue
        m_tc = re.match(r"^([A-Z][A-Za-z0-9'&\- ]{2,})\s*:\s+.*$", s)
        if m_tc:
            titles.append((m_tc.group(1) or "").strip())

    seen: set = set()
    uniq: List[str] = []
    for t in titles:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)
    return uniq

async def fetch_last_assistant_message(session_id: Optional[str]) -> str:
    """Fetch the most recent assistant message for a session from Supabase."""
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and session_id):
        return ""
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }
    url = (
        f"{SUPABASE_URL}/rest/v1/chat_messages"
        f"?session_id=eq.{session_id}&role=eq.assistant&select=content,meta&order=created_at.desc&limit=1"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers)
            if r.status_code >= 400:
                return ""
            data = r.json() or []
    except Exception:
        return ""
    if not data:
        return ""
    row = data[0] if isinstance(data[0], dict) else {}
    content = (row.get("content") or "").strip()
    meta = row.get("meta") or {}
    try:
        if isinstance(meta, str):
            meta = json.loads(meta)
    except Exception:
        meta = {}
    ans_md = (meta.get("answer_markdown") or "").strip() if isinstance(meta, dict) else ""
    return ans_md or content

async def fetch_recent_session_pairs(session_id: Optional[str], max_pairs: int = 2) -> List[Tuple[str, str]]:
    """Return up to `max_pairs` of (query, response) pairs for the session.

    Notes:
    - Supabase `queries` table uses `transcribed_text` for the user query text.
    - We still fall back to `question` for legacy schemas.
    """
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and session_id):
        return []
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    # Prefer transcribed_text; include question for backward compatibility
    q_url = (
        f"{SUPABASE_URL}/rest/v1/queries"
        f"?session_id=eq.{session_id}&select=id,transcribed_text,question,created_at&order=created_at.desc&limit={max_pairs}"
    )
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            q_resp = await client.get(q_url, headers=headers)
            if q_resp.status_code >= 400:
                return []
            q_rows = q_resp.json() or []
    except Exception:
        return []

    pairs: List[Tuple[str, str]] = []
    for qr in q_rows:
        qid = qr.get("id")
        # Use transcribed_text (current schema), fallback to question
        q_text = (qr.get("transcribed_text") or qr.get("question") or "").strip()
        if not qid:
            continue
        r_url = (
            f"{SUPABASE_URL}/rest/v1/responses"
            f"?query_id=eq.{qid}&select=response_text,content,created_at&order=created_at.desc&limit=1"
        )
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r_resp = await client.get(r_url, headers=headers)
                if r_resp.status_code >= 400:
                    continue
                r_rows = r_resp.json() or []
        except Exception:
            continue
        r_text = ""
        if r_rows:
            r = r_rows[0] if isinstance(r_rows[0], dict) else {}
            r_text = (r.get("response_text") or r.get("content") or "").strip()
        pairs.append((q_text, r_text))
    return pairs


def build_prompt(
    message: str,
    history: List[HistoryItem],
    rag_block: str = "",
    web_evidence_block: str = "",
    article_block: str = "",
    chat_mode: bool = False,
    learning_preference: Optional[str] = None,
) -> str:
    def _truncate(text: str, limit: int) -> str:
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    safe_message = _truncate(message, MAX_MESSAGE_CHARS)
    safe_article = _truncate(article_block, MAX_ARTICLE_CHARS)
    safe_rag = _truncate(rag_block, MAX_RAG_CHARS)
    safe_evidence = _truncate(web_evidence_block, MAX_EVIDENCE_CHARS)

    pref_instruction = build_learning_preference_instruction(learning_preference)

    # LLaMA-3.3 Instruct chat headers
    if chat_mode:
        # Minimal system + recent history (compact) + current user
        tone_lines = ""
        # Configure friendly tone + emoji usage without hardcoding specific emojis everywhere
        if AV_EMOJI_STYLE != "off":
            tone_lines += (
                "Adopt a warm, friendly tone. You may add a single light emoji in an opening or closing when helpful "
                "(e.g., 😊 or 🤖). Keep emoji usage minimal and skip them for formal, code, or JSON-only answers.\n"
            )
        if AV_FRIENDLY_OPENERS:
            tone_lines += (
                "If the question is casual, you may begin with a brief friendly greeting. Vary the phrasing across answers, "
                "do not repeat the same greeting, and sometimes omit it entirely. For direct academic questions, begin with the explanation.\n"
            )
        # Optional closings guidance for variety without boilerplate
        tone_lines += (
            "Optionally include one short closing line inviting follow-up (e.g., ask if more detail is needed). Vary its wording, "
            "avoid generic boilerplate, and omit closings for formal, code, or long answers.\n"
        )

        p = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are AskVox, a friendly and helpful AI assistant. "
            "When the user asks who you are or asks you to introduce yourself, explicitly say 'I'm AskVox' and give a short product-focused intro (what you can help with). "
            "Avoid generic phrases like 'I am an AI language model' and avoid mentioning training data/corpus or internal implementation details. "
            "Explain topics in a natural, human, tutor-like way. "
            "Prefer clear paragraph-style explanations with context, reasoning, and examples. "
            "Use bullet points or numbered lists only when they genuinely improve clarity "
            "(such as rankings, comparisons, or step-by-step instructions). "
            "When the request involves rankings, comparisons, or step-by-step instructions, respond with a numbered or bulleted list rather than plain paragraphs. "
            "For rankings or 'top N' queries, present a numbered list starting at 1 with ONE item per line, and include a short explanation for each item.\n"
            "Use concise paragraphs. "
            "If a line is in the form 'Title - details', render it as '**Title** — details'. "
            "Avoid giant wall-of-text paragraphs.\n"
            + (pref_instruction + "\n" if pref_instruction else "")
            + tone_lines +
            "<|eot_id|>"
        )

        # Compact recent history
        filtered_history: List[HistoryItem] = []
        seen = set()
        for h in history:
            if not h.content or not h.content.strip():
                continue
            key = (h.role, h.content.strip())
            if key in seen:
                continue
            seen.add(key)
            filtered_history.append(h)
        filtered_history = filtered_history[-4:]

        remaining_history_chars = MAX_HISTORY_CHARS
        for h in filtered_history:
            if remaining_history_chars <= 0:
                break
            role = "user" if h.role == "user" else "assistant"
            content = _truncate(h.content, 240)
            if len(content) > remaining_history_chars:
                content = _truncate(content, max(0, remaining_history_chars))
            remaining_history_chars -= len(content)
            p += (
                f"<|start_header_id|>{role}<|end_header_id|>\n"
                f"{content}<|eot_id|>"
            )

        p += (
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{safe_message}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        if len(p) > MAX_PROMPT_CHARS:
            p = p[:MAX_PROMPT_CHARS] + "..."
        print("🧠 CHAT_MODE prompt length:", len(p), flush=True)
        print("\n==== PROMPT SENT TO MODEL ====".ljust(40, "="), flush=True)
        print(p[:1200] + ("..." if len(p) > 1200 else ""), flush=True)
        print("="*40, flush=True)
        return p

    # Structured mode with JSON instruction and optional context blocks
    p = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
    )
    p += MODEL_JSON_INSTRUCTION
    if pref_instruction:
        p += "\n" + pref_instruction
    if safe_article:
        p += "\nUse the provided ARTICLE_CONTEXT as the primary source for the user's question.\n"
        p += f"\n{safe_article}\n"
    if safe_rag:
        p += f"\n{safe_rag}\n"
    if safe_evidence:
        p += f"\n{safe_evidence}\n"
    p += "<|eot_id|>"

    # Filter out empty or duplicate history entries (keep last 2 exchanges)
    filtered_history = []
    seen = set()
    for h in history:
        if not h.content or not h.content.strip():
            continue
        # If the caller already included the current user message in history,
        # avoid duplicating it (we append safe_message at the end).
        if h.role == "user" and h.content.strip() == safe_message.strip():
            continue
        key = (h.role, h.content.strip())
        if key in seen:
            continue
        seen.add(key)
        filtered_history.append(h)
    filtered_history = filtered_history[-4:]

    remaining_history_chars = MAX_HISTORY_CHARS
    for h in filtered_history:
        role = "user" if h.role == "user" else "assistant"
        if remaining_history_chars <= 0:
            break
        content = _truncate(h.content, 260)
        if len(content) > remaining_history_chars:
            content = _truncate(content, max(0, remaining_history_chars))
        remaining_history_chars -= len(content)
        p += (
            f"<|start_header_id|>{role}<|end_header_id|>\n"
            f"{content}<|eot_id|>"
        )

    p += (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{safe_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    if len(p) > MAX_PROMPT_CHARS:
        p = p[:MAX_PROMPT_CHARS] + "..."

    # Debug print for prompt sent to model (may contain user text)
    if AV_LOG_PROMPTS:
        n = max(0, AV_LOG_PROMPT_CHARS)
        print("\n==== PROMPT SENT TO MODEL ====".ljust(40, "="), flush=True)
        print(p[:n] + ("..." if len(p) > n else ""), flush=True)
        print("=" * 40, flush=True)

    return p

async def call_cloudrun(prompt: str, timeout: httpx.Timeout) -> str:
    payload = {"message": prompt}

    # Debug: show Cloud Run target and measure request time
    try:
        await _log_cloudrun_meta_once()
    except Exception:
        pass
    print("☁️ Sending request to Cloud Run:", LLAMA_CLOUDRUN_URL, flush=True)
    t0 = time.perf_counter()

    async def _post(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post(LLAMA_CLOUDRUN_URL, json=payload)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await _post(client)
            t1 = time.perf_counter()
            print(f"⏱️ CloudRun generation time: {t1 - t0:.2f}s", flush=True)
    except httpx.ReadTimeout as e:
        req_url = getattr(getattr(e, "request", None), "url", None)
        print(
            "CLOUD RUN READ TIMEOUT: retrying once",
            {"url": str(req_url) if req_url else None, "error": str(e) or repr(e)},
            flush=True,
        )
        retry_timeout = httpx.Timeout(
            connect=timeout.connect,
            read=max(timeout.read or 0, 600.0),
            write=timeout.write,
            pool=timeout.pool,
        )
        try:
            async with httpx.AsyncClient(timeout=retry_timeout) as retry_client:
                t_retry0 = time.perf_counter()
                resp = await _post(retry_client)
                t_retry1 = time.perf_counter()
                print(f"⏱️ CloudRun retry generation time: {t_retry1 - t_retry0:.2f}s", flush=True)
        except httpx.RequestError as e2:
            req_url = getattr(getattr(e2, "request", None), "url", None)
            err_type = type(e2).__name__
            err_text = str(e2) or repr(e2)
            print(
                "CLOUD RUN REQUEST ERROR:",
                {"type": err_type, "url": str(req_url) if req_url else None, "error": err_text},
                flush=True,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Cloud Run request failed ({err_type}): {err_text}",
            )
    except httpx.RequestError as e:
        req_url = getattr(getattr(e, "request", None), "url", None)
        err_type = type(e).__name__
        err_text = str(e) or repr(e)
        print(
            "CLOUD RUN REQUEST ERROR:",
            {"type": err_type, "url": str(req_url) if req_url else None, "error": err_text},
            flush=True,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Cloud Run request failed ({err_type}): {err_text}",
        )

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Cloud Run returned {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid JSON from Cloud Run")

    raw = data.get("response") or data.get("answer") or data.get("reply") or ""
    if not raw:
        raise HTTPException(status_code=502, detail="Cloud Run response missing answer field")
    return raw

async def call_runpod_job_prompt(
    prompt: str,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    domain: Optional[str] = None,
    max_wait_sec: Optional[float] = None,
) -> str:
    """
    Submit a job to RunPod `/run` and poll `/status/{id}` until COMPLETED.
    Expects `RUNPOD_API_KEY` and `RUNPOD_RUN_ENDPOINT` in env.
    """
    if not RUNPOD_RUN_ENDPOINT:
        raise HTTPException(status_code=500, detail="RUNPOD_RUN_ENDPOINT not configured")
    if not RUNPOD_API_KEY:
        raise HTTPException(status_code=500, detail="RUNPOD_API_KEY missing")

    # Make it obvious (once) whether prompt logging is enabled.
    _log_av_flags_snapshot_once()

    headers = {RUNPOD_AUTH_HEADER or "Authorization": f"Bearer {RUNPOD_API_KEY}"}
    inp: Dict[str, Any] = {
        "prompt": prompt,
        "stop": ["<|eot_id|>", "<|start_header_id|>"],
    }
    if (domain or "").strip():
        inp["domain"] = (domain or "").strip()
    # These fields are optional; they only take effect if your RunPod handler reads them.
    if isinstance(max_tokens, int) and max_tokens > 0:
        inp["max_tokens"] = max_tokens
    if isinstance(temperature, (int, float)) and float(temperature) >= 0:
        inp["temperature"] = float(temperature)
    if isinstance(top_p, (int, float)) and 0 < float(top_p) <= 1.0:
        inp["top_p"] = float(top_p)

    payload = {"input": inp}

    effective_max_wait = float(max_wait_sec) if isinstance(max_wait_sec, (int, float)) and float(max_wait_sec) > 0 else float(RUNPOD_MAX_WAIT_SEC)

    try:
        sent_domain = (inp.get("domain") or "").strip()
        print(f"🏷️ RunPod domain: {sent_domain or '(none)'}", flush=True)
        print(f"🚀 Submitting RunPod job: {RUNPOD_RUN_ENDPOINT}", flush=True)
        if AV_LOG_PROMPTS:
            n = max(0, AV_LOG_PROMPT_CHARS)
            print("\n==== PROMPT SENT TO MODEL (RUNPOD user prompt) ====".ljust(40, "="), flush=True)
            print(((prompt or "")[:n] + ("..." if len(prompt or "") > n else "")), flush=True)
            print("=" * 40, flush=True)
        # Reuse one client for submit+poll to reduce connection churn.
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)) as client:
            run_resp = await client.post(RUNPOD_RUN_ENDPOINT, json=payload, headers=headers)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"RunPod run request failed: {e}")

    if run_resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"RunPod /run returned {run_resp.status_code}: {run_resp.text[:200]}")

    try:
        run_data = run_resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid JSON from RunPod /run")

    job_id = run_data.get("id") or run_data.get("jobId") or run_data.get("job_id")
    if not job_id:
        # Some pods may return output immediately
        immediate_output = (run_data.get("output") or {}).get("response") or run_data.get("response")
        if immediate_output:
            return str(immediate_output)
        raise HTTPException(status_code=502, detail="RunPod /run response missing job id")

    # Derive status base URL
    status_base = RUNPOD_STATUS_ENDPOINT.strip() if RUNPOD_STATUS_ENDPOINT else ""
    if not status_base:
        if RUNPOD_RUN_ENDPOINT.endswith("/run"):
            status_base = RUNPOD_RUN_ENDPOINT[: -len("/run")] + "/status"
        else:
            status_base = RUNPOD_RUN_ENDPOINT.rstrip("/") + "/status"

    t0 = time.perf_counter()
    last_status = ""
    # Backoff helps when the pod is busy/queueing.
    poll_delay = max(0.2, float(RUNPOD_POLL_INTERVAL_SEC))
    poll_delay_max = max(poll_delay, 5.0)

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)) as poll_client:
        while (time.perf_counter() - t0) < effective_max_wait:
            url = f"{status_base}/{job_id}"
            if not last_status:
                print(f"⏳ Polling RunPod status: {url}", flush=True)
            try:
                st_resp = await poll_client.get(url, headers=headers)
            except httpx.RequestError as e:
                last_status = f"request_error: {e}"
                await asyncio.sleep(poll_delay)
                poll_delay = min(poll_delay_max, poll_delay * 1.25)
                continue

            if st_resp.status_code >= 400:
                last_status = f"http_{st_resp.status_code}"
                await asyncio.sleep(poll_delay)
                poll_delay = min(poll_delay_max, poll_delay * 1.15)
                continue

            try:
                st_data = st_resp.json()
            except Exception:
                last_status = "bad_json"
                await asyncio.sleep(poll_delay)
                poll_delay = min(poll_delay_max, poll_delay * 1.15)
                continue

            status = (st_data.get("status") or st_data.get("state") or "").upper()
            last_status = status or last_status
            if status:
                print(f"🔄 RunPod status: {status}", flush=True)
            if status == "COMPLETED":
                out = st_data.get("output") or {}
                if isinstance(out, dict):
                    ans = out.get("response") or out.get("answer") or out.get("reply")
                    if ans:
                        return str(ans)
                # Handle alternative shapes
                if isinstance(out, str) and out:
                    return out
                # Fallback: try top-level fields
                ans2 = st_data.get("response") or st_data.get("answer") or st_data.get("reply")
                if ans2:
                    return str(ans2)
                raise HTTPException(status_code=502, detail="RunPod status completed but no output.response")
            if status in {"FAILED", "ERROR", "CANCELLED"}:
                print(f"❌ RunPod job: {status}", flush=True)
                raise HTTPException(status_code=502, detail=f"RunPod job {status}")

            await asyncio.sleep(poll_delay)
            # Slowly backoff even on success to reduce pressure.
            poll_delay = min(poll_delay_max, poll_delay * 1.05)

    print(
        f"⏰ RunPod job timed out (last_status={last_status})",
        {"waited_sec": round(time.perf_counter() - t0, 2), "max_wait_sec": effective_max_wait},
        flush=True,
    )
    raise HTTPException(status_code=504, detail=f"RunPod job timed out (last_status={last_status})")

async def fetch_session_article_context(session_id: str) -> Dict[str, Any]:
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and session_id):
        return {}
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{SUPABASE_URL}/rest/v1/chat_sessions?id=eq.{session_id}&select=article_context&limit=1"
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code >= 400:
                return {}
            data = resp.json()
        except Exception:
            return {}
    if not data:
        return {}
    row = data[0] if isinstance(data[0], dict) else {}
    article_context = row.get("article_context") if isinstance(row, dict) else None
    return article_context if isinstance(article_context, dict) else {}

def build_article_block_from_cached(article_ctx: Dict[str, Any]) -> str:
    cached = (article_ctx or {}).get("cached_content")
    cached_at = (article_ctx or {}).get("cached_at")
    title = (article_ctx or {}).get("title")
    url = (article_ctx or {}).get("url")

    if not cached or not isinstance(cached, str):
        return ""
    if len(cached.strip()) < 200:
        return ""

    if cached_at:
        try:
            dt = datetime.fromisoformat(str(cached_at).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600
            if age_hours > 24:
                return ""
        except Exception:
            return ""

    text = re.sub(r"\n{3,}", "\n\n", cached).strip()
    if len(text) > MAX_ARTICLE_CHARS:
        text = text[:MAX_ARTICLE_CHARS] + "..."

    header = "[ARTICLE_CONTEXT]\n"
    if title:
        header += f"Title: {title}\n"
    if url:
        header += f"URL: {url}\n"
    return f"{header}\n{text}\n"

async def fetch_article_context(url: str, title: str = "") -> Tuple[str, str]:
    if not url:
        return "", ""
    jina_url = f"https://r.jina.ai/http://{url}" if not url.startswith("http") else f"https://r.jina.ai/{url}"
    timeout = httpx.Timeout(connect=8.0, read=12.0, write=8.0, pool=8.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(jina_url)
            if resp.status_code >= 400:
                return "", ""
            text = resp.text
        except Exception:
            return "", ""

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > MAX_ARTICLE_CHARS:
        text = text[:MAX_ARTICLE_CHARS] + "..."
    header = "[ARTICLE_CONTEXT]\n"
    if title:
        header += f"Title: {title}\n"
    header += f"URL: {url}\n"
    return f"{header}\n{text}\n", text

async def update_session_article_cache(
    session_id: Optional[str],
    article_context: Optional[Dict[str, Any]],
    scraped_text: str,
):
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and session_id):
        return
    if not article_context or not isinstance(article_context, dict):
        return
    if not scraped_text or len(scraped_text.strip()) < 200:
        return

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    new_ctx = dict(article_context)
    new_ctx["cached_content"] = scraped_text[:1800]
    new_ctx["cached_at"] = datetime.now(timezone.utc).isoformat()

    url = f"{SUPABASE_URL}/rest/v1/chat_sessions?id=eq.{session_id}"
    body = {"article_context": new_ctx}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.patch(url, headers=headers, json=body)
            if resp.status_code >= 400:
                print("CACHE UPDATE ERROR:", resp.status_code, resp.text[:200], flush=True)
    except Exception as e:
        print(f"CACHE UPDATE EXCEPTION: {e}", flush=True)

def repair_json_instruction(user_message: str) -> str:
    return f"""
Return ONLY valid JSON (no extra text, no markdown fences) matching this schema:

{{
  "answer_markdown": "string",
  "need_web_sources": true/false,
  "need_images": true/false,
  "need_youtube": true/false,
  "web_query": "string",
  "image_query": "string",
  "youtube_query": "string"
}}

User message: {user_message}
"""

# -----------------------
# Inline citation token validation
# -----------------------
_CITE_TOKEN_RE = re.compile(r"\[\[cite:(\d+)\]\]")

def validate_and_clean_citations(answer_md: str, sources: List[SourceItem]) -> str:
    max_n = len(sources)

    def repl(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= max_n:
            return match.group(0)
        return ""

    out = _CITE_TOKEN_RE.sub(repl, answer_md)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

# -----------------------
# CLOUD CALL
# -----------------------
async def generate_cloud_structured(
    message: str,
    history: List[HistoryItem],
    article_title: Optional[str] = None,
    article_url: Optional[str] = None,
    article_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    domain: Optional[str] = None,
    max_history: int = 3,
) -> AssistantPayload:
    t_start = time.perf_counter()
    if not (LLAMA_CLOUDRUN_URL or RUNPOD_RUN_ENDPOINT):
        raise HTTPException(status_code=500, detail="No model endpoint configured: set LLAMA_CLOUDRUN_URL or RUNPOD_RUN_ENDPOINT in .env")

    # Moderation detection
    mod = await moderation_check(message, user_id=user_id, session_id=session_id)
    if not mod.get("allowed", True):
        fallback_text = mod.get("response") or (
            "I'm not able to respond to that message. "
            "Could you try rephrasing your question?"
        )
        return AssistantPayload(
            answer_markdown=fallback_text,
            sources=[],
            images=[],
            youtube=[],
            source_count=0,
            cite_available=False,
            cite_label=None,
        )

    # Article context
    article_block = ""
    cached_block = build_article_block_from_cached(article_context or {}) if article_context else ""
    if cached_block:
        article_block = cached_block
    elif article_url:
        article_block, scraped_text = await fetch_article_context(article_url, article_title or "")
        await update_session_article_cache(
            session_id=session_id,
            article_context=article_context,
            scraped_text=scraped_text,
        )

    # Domain-gated RAG: classify once, reuse result.
    # For generic media follow-ups (e.g. "pictures of it"), classify/retrieve using the
    # prior topic from history so results stay on-topic.
    def _is_generic_media_followup_for_context(s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return True
        has_media_word = bool(_MEDIA_WORDS_RE.search(t) or _IMAGE_WORDS_RE.search(t))
        has_pronoun_ref = bool(re.search(r"\b(it|those|them|that|this)\b", t, re.I))
        has_more = bool(re.search(r"\bmore\b", t, re.I))
        if has_media_word and (has_pronoun_ref or has_more):
            return True
        if len(t) < 12 and has_media_word:
            return True
        return False

    context_seed_for_rag = message
    if _is_generic_media_followup_for_context(message):
        last_assistant_text = next(
            (
                (h.content or "").strip()
                for h in reversed(history or [])
                if getattr(h, "role", None) == "assistant" and (h.content or "").strip()
            ),
            "",
        )
        title_hint = None
        if last_assistant_text:
            try:
                titles = extract_titles_from_answer(last_assistant_text) or []
                if titles:
                    title_hint = titles[0]
            except Exception:
                title_hint = None

        # Fall back to the last non-generic user message.
        hist_user_msgs: List[str] = [
            (h.content or "").strip()
            for h in (history or [])
            if getattr(h, "role", None) == "user" and (h.content or "").strip()
        ]
        last_user_topic = ""
        for s in reversed(hist_user_msgs):
            if not _is_generic_media_followup_for_context(s):
                last_user_topic = s
                break
        context_seed_for_rag = make_fallback_query(title_hint or last_user_topic or message, max_len=120)

    classified_domain = None
    try:
        classified_domain = classify_domain(context_seed_for_rag)
        print(f"[RAG] domain classified: '{classified_domain}'", flush=True)
    except Exception as e:
        print(f"[RAG] classifier error (fail-soft): {e}", flush=True)

    rag_chunks = await rag_retrieve(context_seed_for_rag, k=4, classified_domain=classified_domain)
    rag_block = build_rag_block(rag_chunks)

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
    trimmed_history = history[-max_history:] if max_history and max_history > 0 else []
    is_first_turn = len(trimmed_history) == 0
    # If no client history provided but we have a session, keep current behavior (no fetch here)
    user_flags = keyword_router(message)
    chat_flag = not user_flags.get("want_web", False)

    # One-time snapshot so missing logs can be explained quickly.
    _log_av_flags_snapshot_once()

    learning_pref = await fetch_learning_preference(user_id)
    if AV_LOG_LEARNING_PREF and learning_pref:
        print(f"🎓 learning_preference={learning_pref}", flush=True)

    # Log the *exact* instruction string that will be injected into the prompt.
    # This is safe to log (no user content), and helps verify preference shaping.
    if AV_LOG_LEARNING_PREF:
        pref_instruction = build_learning_preference_instruction(learning_pref)
        target = "runpod" if RUNPOD_RUN_ENDPOINT else "cloudrun"
        print(
            "🧩 learning_pref_instruction_sent:",
            {
                "target": target,
                "learning_preference": learning_pref or "(none)",
                "chat_mode": chat_flag,
                "instruction": pref_instruction.strip(),
            },
            flush=True,
        )
    if AV_LOG_PROMPTS:
        n = max(0, AV_LOG_PROMPT_CHARS)
        msg_preview = (message or "")[:n]
        # Keep log single-line to avoid noisy multi-line logs.
        msg_preview = msg_preview.replace("\n", " ").replace("\r", " ")
        print(
            "🗣️ user_message_preview:",
            {"user_id": user_id or "(none)", "chars": n, "preview": msg_preview},
            flush=True,
        )

    # Tune sampling/length by preference. (Effective for RunPod; CloudRun may ignore.)
    def _gen_for_pref(pref: Optional[str], chat_mode: bool) -> Dict[str, Any]:
        p = _normalize_learning_pref(pref)
        # Structured/JSON answers need more headroom than chat-mode.
        if p == "secondary":
            return {"max_tokens": 700 if not chat_mode else 520, "temperature": 0.35, "top_p": 0.9}
        if p == "tertiary":
            return {"max_tokens": 850 if not chat_mode else 650, "temperature": 0.5, "top_p": 0.92}
        if p == "university":
            return {"max_tokens": 1200 if not chat_mode else 900, "temperature": 0.55, "top_p": 0.93}
        # leisure + default
        return {"max_tokens": 900 if not chat_mode else 720, "temperature": 0.7, "top_p": 0.95}

    tuned = _gen_for_pref(learning_pref, chat_flag)
    if AV_LOG_LEARNING_PREF:
        print(
            "🎛️ gen_tuning:",
            {"learning_preference": learning_pref or "(none)", "chat_mode": chat_flag, **tuned},
            flush=True,
        )

    # Debug: show the history window used for context (up to 4 entries)
    def _preview(s: str, n: int = 180) -> str:
        return (s or "").replace("\n", " ")[:n]
    try:
        print(
            "HISTORY_WINDOW:",
            [{"role": getattr(h, "role", None), "content": _preview(getattr(h, "content", ""))} for h in (trimmed_history or [])],
            flush=True,
        )
    except Exception:
        pass

    # Build context base from Supabase pairs when follow-up is generic
    def _is_generic_followup(s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return True
        has_media_word = bool(_MEDIA_WORDS_RE.search(t) or _IMAGE_WORDS_RE.search(t))
        has_pronoun_ref = bool(re.search(r"\b(it|those|them|that|this)\b", t, re.I))
        has_more = bool(re.search(r"\bmore\b", t, re.I))
        # Treat things like "can show me more videos?" as generic media follow-ups
        if has_media_word and (has_pronoun_ref or has_more):
            return True
        # Media-only prompts like "videos and images" => generic
        if has_media_word:
            topic_hint = re.sub(
                r"\b(show me|give me|find|search|look for|video|videos|youtube|yt|image|images|picture|pictures|photo|photos)\b",
                "",
                t,
                flags=re.I,
            )
            topic_hint = re.sub(r"\b(and|or)\b", " ", topic_hint, flags=re.I)
            topic_hint = re.sub(r"\s+", " ", topic_hint).strip().lower()
            if not topic_hint:
                return True
            # Phrases like "image to show" / "show me image" contain no topic.
            if re.fullmatch(r"(to\s+)?(show|see|watch)", topic_hint):
                return True
            if topic_hint in {"support", "to support", "help", "to help", "please"}:
                return True
            if re.fullmatch(r"(videos?|images?|pictures?|photos?)(\s+and\s+(videos?|images?|pictures?|photos?))*", t.strip().lower()):
                return True
        # Very short and mentions media words => likely generic
        if len(t) < 10 and has_media_word:
            return True
        return False

    context_query_base = None
    if _is_generic_followup(message) and session_id:
        pairs = await fetch_recent_session_pairs(session_id, max_pairs=2)
        try:
            print(
                "SUPABASE RECENT PAIRS:",
                [{"q": _preview(q), "r": _preview(r)} for (q, r) in (pairs or [])],
                flush=True,
            )
        except Exception:
            pass
        # Prefer a title extracted from last response; else last query text
        last_resp = next((p[1] for p in pairs if (p[1] or "").strip()), "")
        last_query = next((p[0] for p in pairs if (p[0] or "").strip()), "")
        title_hint = None
        # If Supabase has no response text, fall back to the last assistant message in history.
        if not last_resp:
            last_resp = next(
                (
                    (h.content or "").strip()
                    for h in reversed(trimmed_history or [])
                    if getattr(h, "role", None) == "assistant" and (h.content or "").strip()
                ),
                "",
            )
        if last_resp:
            titles = extract_titles_from_answer(last_resp)
            if titles:
                title_hint = titles[0]
        # If Supabase lacks pairs, fall back to last non-generic user message in history
        if not last_query:
            hist_user_msgs: List[str] = [
                (h.content or "").strip()
                for h in (trimmed_history or [])
                if getattr(h, "role", None) == "user" and (h.content or "").strip()
            ]
            def _is_generic(s: str) -> bool:
                t = (s or "").strip()
                if not t:
                    return True
                has_media_word = bool(_MEDIA_WORDS_RE.search(t) or _IMAGE_WORDS_RE.search(t))
                has_pronoun_ref = bool(re.search(r"\b(it|those|them|that|this)\b", t, re.I))
                if has_media_word and has_pronoun_ref:
                    return True
                if len(t) < 10 and has_media_word:
                    return True
                return False
            for s in reversed(hist_user_msgs):
                if not _is_generic(s):
                    last_query = s
                    break
            if not last_query and hist_user_msgs:
                last_query = hist_user_msgs[0]
        context_query_base = make_fallback_query(title_hint or last_query or article_title or message, max_len=120)
        try:
            print(
                "CONTEXT_BASE:",
                {"title_hint": title_hint, "last_query": _preview(last_query), "context_query_base": context_query_base},
                flush=True,
            )
        except Exception:
            pass


    # ✅ Draft answer first (no tools yet)
    if RUNPOD_RUN_ENDPOINT:
        prompt1 = build_runpod_user_prompt(
            message,
            trimmed_history,
            rag_block=rag_block,
            article_block=article_block,
            chat_mode=chat_flag,
            learning_preference=learning_pref,
        )
        raw = await call_runpod_job_prompt(
            prompt1,
            max_tokens=int(tuned.get("max_tokens") or 0) or None,
            temperature=float(tuned.get("temperature")) if tuned.get("temperature") is not None else None,
            top_p=float(tuned.get("top_p")) if tuned.get("top_p") is not None else None,
            domain=domain,
        )
    else:
        raw = await call_cloudrun(
            build_prompt(
                message,
                trimmed_history,
                rag_block=rag_block,
                article_block=article_block,
                chat_mode=chat_flag,
                learning_preference=learning_pref,
            ),
            timeout=timeout,
        )
    if not chat_flag:
        plan = extract_json(raw)
    else:
        plan = {}


    # ✅ Non-destructive fallback: wrap raw text into JSON without rewriting content
    if not plan or "answer_markdown" not in plan or not (plan.get("answer_markdown") or "").strip():
        plan = safe_wrap_json(raw)

    answer_md = (plan.get("answer_markdown") or "").strip()
    # Immediate fallback: if model didn't return JSON, show raw text
    if not answer_md and raw:
        answer_md = raw.strip()

    # Ensure plan has required keys so downstream logic stays consistent
    if not plan:
        plan = {}
    if "need_web_sources" not in plan:
        plan["need_web_sources"] = False
    if "need_images" not in plan:
        plan["need_images"] = False
    if "need_youtube" not in plan:
        plan["need_youtube"] = False
    plan["web_query"] = (plan.get("web_query") or "").strip()
    plan["image_query"] = (plan.get("image_query") or "").strip()
    plan["youtube_query"] = (plan.get("youtube_query") or "").strip()
    plan["answer_markdown"] = answer_md
    
    # -----------------------
    # Sanity checker: use fresh Google sources to refine answer (non-JSON)
    # -----------------------
    sanity_sources: List[SourceItem] = []
    sanity_q_used: str = ""
    sanity_media_boost: bool = False
    try:
        # If we're already running long (common with RunPod cold starts), skip extra passes.
        if RUNPOD_RUN_ENDPOINT and (time.perf_counter() - t_start) > RUNPOD_TOTAL_BUDGET_SEC:
            raise RuntimeError("budget_exceeded")
        current_year = datetime.now(timezone.utc).year
        msg_text = (message or "")
        years = extract_years(msg_text + " " + (answer_md or ""))
        recency_intent = bool(re.search(r"\b(now|currently|today|this year|latest|trending|airing|released|premiered)\b", msg_text, re.I))
        mentions_future = any(int(y) > MODEL_KNOWLEDGE_CUTOFF_YEAR for y in years if y.isdigit())
        fact_seeking = is_fact_seeking_question(msg_text)
        # IMPORTANT: Don't sanity-check every request just because the calendar year
        # is beyond the model cutoff (e.g., 2026 > 2023). Only do it when the user
        # explicitly implies recency or mentions future years.
        # Also sanity-check fact-seeking questions to attach credible sources;
        # keep it lightweight (no extra media auto-enablement unless time-sensitive).
        should_sanity = (recency_intent or mentions_future) or fact_seeking
        if should_sanity:
            sanity_media_boost = bool(recency_intent or mentions_future)
            sanity_q = build_sanity_web_query(message, answer_md, context_query_base=None)
            sanity_q_used = sanity_q
            # Prefer last year for recency queries; otherwise fetch broadly.
            sanity_num = 7 if (recency_intent or mentions_future) else 6
            if recency_intent or mentions_future:
                sanity_sources = await fast_google_web_search(sanity_q, num=sanity_num, date_restrict="y[1]")
                if not sanity_sources:
                    sanity_sources = await fast_google_web_search(sanity_q, num=sanity_num)
            else:
                sanity_sources = await fast_google_web_search(sanity_q, num=sanity_num)

            # Only run the extra rewrite pass when it is likely to improve correctness/utility.
            should_refine_answer = bool(
                (recency_intent or mentions_future)
                or looks_like_needs_web(answer_md)
                or looks_like_deflection_or_nonanswer(answer_md)
                or bool(_FACTUAL_WEB_INTENT_RE.search(msg_text))
            )
            if sanity_sources and should_refine_answer:
                sanity_block = build_web_evidence_block(sanity_sources, [])
                prompt2 = build_second_pass_prompt_chat(
                    message=message,
                    history=trimmed_history,
                    original_answer=answer_md,
                    web_evidence_block=sanity_block,
                    article_block=article_block,
                    learning_preference=learning_pref,
                )
                refined_raw = ""
                try:
                    if RUNPOD_RUN_ENDPOINT:
                        if (time.perf_counter() - t_start) > RUNPOD_TOTAL_BUDGET_SEC:
                            raise RuntimeError("budget_exceeded")
                        refined_raw = await call_runpod_job_prompt(
                            prompt2,
                            max_tokens=int(tuned.get("max_tokens") or 0) or None,
                            temperature=float(tuned.get("temperature")) if tuned.get("temperature") is not None else None,
                            top_p=float(tuned.get("top_p")) if tuned.get("top_p") is not None else None,
                            domain=domain,
                        )
                    else:
                        refined_raw = await call_cloudrun(prompt2, timeout=timeout)
                except Exception:
                    refined_raw = ""

                plan2 = extract_json(refined_raw)
                refined_md = (plan2.get("answer_markdown") or "").strip() if plan2 else (refined_raw or "").strip()
                if refined_md:
                    answer_md = refined_md
                    plan["answer_markdown"] = answer_md
                    print("🧩 SECOND_PASS_APPLIED (sanity)", {"query": sanity_q[:120], "len": len(answer_md)}, flush=True)
    except Exception:
        pass

    # Model suggestions (soft)
    model_need_web = bool(plan.get("need_web_sources"))
    need_img = bool(plan.get("need_images"))
    need_yt = bool(plan.get("need_youtube"))
    web_q = (plan.get("web_query") or "").strip()
    img_q = (plan.get("image_query") or "").strip()
    yt_q = (plan.get("youtube_query") or "").strip()

    # For the very first query in a session (non-smalltalk),
    # default to surfacing both images and videos unless caller disables via env.
    if is_first_turn and not is_smalltalk_or_identity(message):
        if not need_img:
            need_img = True
        if not need_yt:
            need_yt = True

    # If sanity sources were fetched for time-sensitive requests, also surface images and auto-enable YouTube for display
    if 'sanity_sources' in locals() and sanity_sources and (('sanity_media_boost' in locals() and sanity_media_boost) or (recency_intent if 'recency_intent' in locals() else False) or (mentions_future if 'mentions_future' in locals() else False)):
        # Prefer enabling images when fresh sources exist
        if not need_img:
            need_img = True
        # Auto-enable YouTube when sanity found fresh web evidence
        if not need_yt:
            need_yt = True
        if not img_q:
            # Derive a neutral image query from the latest answer/title or sanity query
            title_candidates = []
            try:
                title_candidates = extract_titles_from_answer(answer_md)
            except Exception:
                title_candidates = []
            base_iq = (title_candidates[0] if title_candidates else (sanity_q_used or (context_query_base or message)))
            # Avoid hard-coding "poster"; keep it close to the semantic topic
            img_q = str(base_iq or "").strip()[:120]

    # ✅ Deterministic keyword routing (still used for images/youtube decisions)
    user_flags = keyword_router(message)

    # Option C: for learning prompts, auto-enable web + YouTube.
    auto_learning_media = wants_learning_resources(message)
    if auto_learning_media:
        user_flags["auto_youtube"] = True
        user_flags["auto_images"] = False
        need_yt = True

    # Who-is prompts: auto-enable images (even if user didn't explicitly say "images").
    if is_who_is_question(message):
        user_flags["auto_images"] = True
        need_img = True
        subj = extract_who_is_subject(message)
        if subj:
            if not web_q:
                web_q = subj
            if not img_q:
                img_q = f"{subj} portrait"[:120]

    # ✅ Smart web decision: explicit intent OR factual/time-sensitive OR model uncertainty
    need_web, need_web_reason = infer_need_web_sources(
        message=message,
        answer_md=answer_md,
        user_flags=user_flags,
        model_need_web=model_need_web,
        elapsed_sec=(time.perf_counter() - t_start),
    )

    # Smart budget: degrade gracefully near time budget.
    # - Always attempt web sources + a web-evidence second pass (for freshness/citations).
    # - When time is tight, drop optional media, fetch fewer sources with shorter timeouts,
    #   and cap second-pass generation tokens.
    elapsed_now = (time.perf_counter() - t_start)
    remaining_now = _TOTAL_BUDGET_SEC - float(elapsed_now)
    budget_state = "ok"
    google_num = 7
    tavily_num = 6
    google_timeout = 3.5
    tavily_timeout = 4.5

    if remaining_now < _BUDGET_MEDIA_MIN_REMAIN_SEC:
        # Media is most expensive after web; drop it first.
        if not (user_flags.get("want_images") or FORCE_IMAGES):
            need_img = False
        if not (user_flags.get("want_youtube") or user_flags.get("auto_youtube") or FORCE_YOUTUBE):
            need_yt = False
        budget_state = "degraded_media"

    if remaining_now < _BUDGET_WEB_SECOND_PASS_MIN_REMAIN_SEC:
        google_num = 5
        tavily_num = 4
        google_timeout = 2.6
        tavily_timeout = 3.6
        if budget_state == "ok":
            budget_state = "degraded_second_pass"

    if remaining_now < _BUDGET_WEB_MIN_REMAIN_SEC:
        budget_state = "budget_exhausted"
        google_num = 3
        tavily_num = 0
        google_timeout = 1.8
        tavily_timeout = 0.0

    if budget_state != "ok":
        try:
            print(
                "BUDGET STATE:",
                {
                    "elapsed_sec": round(elapsed_now, 2),
                    "remaining_sec": round(remaining_now, 2),
                    "state": budget_state,
                    "google_num": google_num,
                    "tavily_num": tavily_num,
                    "google_timeout": google_timeout,
                    "tavily_timeout": tavily_timeout,
                    "need_img": need_img,
                    "need_yt": need_yt,
                },
                flush=True,
            )
        except Exception:
            pass

    # Fetch web sources only when the smart decision says we need them
    sources: List[SourceItem] = []
    evidence_chunks: List[Dict[str, str]] = []
    if not need_web:
        plan["need_web_sources"] = False
        print(
            "🔕 Skipping web sources",
            {"reason": need_web_reason, "message": message[:80]},
            flush=True,
        )
    else:
        base_for_web = (context_query_base or web_q or message)
        # For follow-ups like "elaborate more" / "apart from that",
        # bias the web query toward additional/new information beyond what was already given.
        if _MORE_DETAIL_FOLLOWUP_RE.search(message or ""):
            base_for_web = f"{base_for_web} additional details new information not already mentioned"[:180]
        web_q = enforce_web_query_constraints(message, base_for_web)

        # Fast, budget-aware web fetch.
        try:
            g_task = asyncio.create_task(fast_google_web_search(web_q, num=google_num, timeout_sec=google_timeout))
            if tavily_num > 0 and TAVILY_API_KEY:
                t_task = asyncio.create_task(fast_tavily(web_q, max_sources=tavily_num, timeout_sec=tavily_timeout))
                g_res, (tav_sources, tav_chunks) = await asyncio.gather(g_task, t_task)
            else:
                g_res = await g_task
                tav_sources, tav_chunks = ([], [])
            sources = g_res or []
            evidence_chunks = tav_chunks
            if tav_sources:
                seen = {s.url.lower(): s for s in sources if s.url}
                for s in tav_sources:
                    key = (s.url or "").lower()
                    if key and key not in seen:
                        sources.append(s)
                        seen[key] = s
        except Exception as e:
            try:
                print("WEB FETCH FAILED (fail-soft)", {"error": str(e), "web_q": web_q[:120]}, flush=True)
            except Exception:
                pass

        if not sources:
            need_web = False
            plan["need_web_sources"] = False
            print(
                "🔕 Web requested but no sources available",
                {"reason": need_web_reason, "web_q": web_q},
                flush=True,
            )
    if FORCE_IMAGES:
        need_img = True
    if FORCE_YOUTUBE:
        need_yt = True

    # Explicit user request overrides model
    if user_flags.get("want_images"):
        need_img = True
    if user_flags.get("want_youtube"):
        need_yt = True
    # Image-only follow-up: ensure YouTube is OFF when user did not ask for it
    if user_flags.get("want_images") and not user_flags.get("want_youtube"):
        need_yt = False

    # After the first turn in a session, only show images/YouTube
    # if the user explicitly asks for them (or FORCE_* is enabled).
    if not is_first_turn:
        if not (user_flags.get("want_images") or user_flags.get("auto_images")) and not FORCE_IMAGES:
            need_img = False
        if not (user_flags.get("want_youtube") or user_flags.get("auto_youtube")) and not FORCE_YOUTUBE:
            need_yt = False

    # Build base topic from the last non-generic user message or Supabase pairs
    user_msgs: List[str] = [
        (h.content or "").strip()
        for h in (trimmed_history or [])
        if getattr(h, "role", None) == "user" and (h.content or "").strip()
    ]

    def _is_generic_followup(s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return True
        # Use existing media identifiers + pronoun/more reference to detect generic follow-ups
        has_media_word = bool(_MEDIA_WORDS_RE.search(t) or _IMAGE_WORDS_RE.search(t))
        has_pronoun_ref = bool(re.search(r"\b(it|those|them|that|this)\b", t, re.I))
        has_more = bool(re.search(r"\bmore\b", t, re.I))
        if has_media_word and (has_pronoun_ref or has_more):
            return True
        if has_media_word:
            topic_hint = re.sub(
                r"\b(show me|give me|find|search|look for|video|videos|youtube|yt|image|images|picture|pictures|photo|photos)\b",
                "",
                t,
                flags=re.I,
            )
            topic_hint = re.sub(r"\b(and|or)\b", " ", topic_hint, flags=re.I)
            topic_hint = re.sub(r"\s+", " ", topic_hint).strip().lower()
            if not topic_hint:
                return True
            if re.fullmatch(r"(to\s+)?(show|see|watch)", topic_hint):
                return True
            if topic_hint in {"support", "to support", "help", "to help", "please"}:
                return True
            if re.fullmatch(r"(videos?|images?|pictures?|photos?)(\s+and\s+(videos?|images?|pictures?|photos?))*", t.strip().lower()):
                return True
        # Very short and mentions media words => likely generic
        if len(t) < 10 and has_media_word:
            return True
        return False

    base_topic_src = None
    for s in reversed(user_msgs):
        if not _is_generic_followup(s):
            base_topic_src = s
            break
    if not base_topic_src and user_msgs:
        base_topic_src = user_msgs[0]
    base_topic = make_fallback_query((context_query_base or base_topic_src or article_title or message), max_len=100)

    # web_q already enforced; prepare sensible fallback if empty
    if not web_q:
        web_q = make_fallback_query(base_topic or (article_title or message), max_len=120)

    if need_img and not img_q:
        img_q = f"{base_topic} photos"[:120]
    if need_yt and not yt_q:
        yt_q = f"{base_topic} explained"[:120]

    # Inline refinement for generic media follow-ups using current answer/title context
    if need_img or need_yt:
        # Detect generic phrases
        def _is_generic(q: str) -> bool:
            s = (q or "").strip().lower()
            if not s:
                return True
            if re.match(r"^(videos?|images?|pictures?|photos?)\b", s or "") is not None:
                return True
            if len(s) < 8:
                return True
            # Treat media-only followups like "to show photos" as generic.
            stripped = re.sub(
                r"\b(show me|give me|find|search|look for|to|video|videos|youtube|yt|image|images|picture|pictures|photo|photos)\b",
                " ",
                s,
                flags=re.I,
            )
            stripped = re.sub(r"\s+", " ", stripped).strip().lower()
            if not stripped:
                return True
            if re.fullmatch(r"(show|see|watch)", stripped):
                return True
            return False

        # Prefer a concrete topic from the current answer (e.g., show title)
        topic_title = None
        try:
            title_list = extract_titles_from_answer(answer_md)
            if title_list:
                topic_title = title_list[0]
        except Exception:
            topic_title = None
        if not topic_title:
            # Fallbacks: sanity query, base topic, or message
            topic_title = (sanity_q_used or base_topic or context_query_base or message)

        if need_img and _is_generic(img_q):
            if re.search(r"\b(movie|film|series|episode|season|anime|trailer|cast|actor|actress|tv show)\b", str(topic_title or ""), re.I):
                img_q = f"{topic_title} poster stills promotional photos cast".strip()[:120]
            else:
                img_q = f"{topic_title} diagram illustration cross section".strip()[:120]
        if need_yt and _is_generic(yt_q):
            if re.search(r"\b(movie|film|series|episode|season|anime|trailer|cast|actor|actress|tv show)\b", str(topic_title or ""), re.I):
                yt_q = f"{topic_title} official trailer clips".strip()[:120]
            else:
                yt_q = f"{topic_title} explained documentary lecture".strip()[:120]

    print(
        "ROUTED FLAGS:",
        {
            "need_web": need_web,
            "need_web_reason": need_web_reason,
            "web_q": web_q,
            "need_img": need_img,
            "img_q": img_q,
            "need_yt": need_yt,
            "yt_q": yt_q,
            "answer_len": len(answer_md),
            "user_flags": user_flags,
        },
        flush=True,
    )
    print("MEDIA QUERIES:", {"base_topic": base_topic, "img_q": img_q, "yt_q": yt_q}, flush=True)

    # Keep previously fetched sources; initialize media containers
    images: List[ImageItem] = []
    youtube: List[YouTubeItem] = []

    # Keep a copy of the best pre-web draft so we can fall back if the second pass is worse.
    draft_answer_md = answer_md

    # Merge sanity sources (if any) into payload sources later
    if 'sanity_sources' in locals() and sanity_sources:
        try:
            existing = {s.url.lower(): s for s in sources if s.url}
            for s in sanity_sources:
                key = (s.url or "").lower()
                if key and key not in existing:
                    sources.append(s)
                    existing[key] = s
        except Exception:
            pass

    # 3) Web evidence + second pass answer (only if need_web)
    # Always attempt this pass for freshness/citations; cap compute when near budget.
    if need_web and web_q:
        web_evidence_block = build_web_evidence_block(sources, evidence_chunks)

        # ✅ second pass should fail-soft (no 502)
        raw2 = ""
        try:
            if RUNPOD_RUN_ENDPOINT:
                if (time.perf_counter() - t_start) > RUNPOD_TOTAL_BUDGET_SEC:
                    raise RuntimeError("budget_exceeded")
                prompt2 = build_runpod_user_prompt(
                    message,
                    trimmed_history,
                    rag_block=rag_block,
                    web_evidence_block=web_evidence_block,
                    article_block=article_block,
                    chat_mode=False,
                    learning_preference=learning_pref,
                )
                raw2 = await call_runpod_job_prompt(
                    prompt2,
                    max_tokens=(
                        min(int(tuned.get("max_tokens") or 0) or 900, 450)
                        if (budget_state in {"degraded_second_pass", "budget_exhausted"} if 'budget_state' in locals() else False)
                        else (int(tuned.get("max_tokens") or 0) or None)
                    ),
                    temperature=(
                        0.35
                        if (budget_state in {"degraded_second_pass", "budget_exhausted"} if 'budget_state' in locals() else False)
                        else (float(tuned.get("temperature")) if tuned.get("temperature") is not None else None)
                    ),
                    top_p=(
                        0.9
                        if (budget_state in {"degraded_second_pass", "budget_exhausted"} if 'budget_state' in locals() else False)
                        else (float(tuned.get("top_p")) if tuned.get("top_p") is not None else None)
                    ),
                    domain=domain,
                )
            else:
                raw2 = await call_cloudrun(
                    build_prompt(
                        message,
                        trimmed_history,
                        rag_block=rag_block,
                        web_evidence_block=web_evidence_block,
                        article_block=article_block,
                        chat_mode=False,
                        learning_preference=learning_pref,
                    ),
                    timeout=timeout,
                )
        except HTTPException as e:
            print("SECOND PASS GENERATION FAILED:", e.detail, flush=True)
        except RuntimeError as e:
            if str(e) == "budget_exceeded":
                print("SECOND PASS SKIPPED (budget)", {"elapsed_sec": round(time.perf_counter() - t_start, 2)}, flush=True)
            else:
                print("SECOND PASS SKIPPED (runtime)", {"error": str(e)}, flush=True)

        plan2 = extract_json(raw2) if raw2 else {}

        # ✅ Non-destructive fallback for second pass
        if raw2 and (not plan2 or "answer_markdown" not in plan2 or not (plan2.get("answer_markdown") or "").strip()):
            plan2 = safe_wrap_json(raw2)

        if plan2 and (plan2.get("answer_markdown") or "").strip():
            refined_md = (plan2.get("answer_markdown") or "").strip()
            # Choose the final visible answer based on both draft and refined outputs.
            # Still keep plan2 flags/queries for routing/media decisions.
            answer_md = choose_final_answer(draft_answer_md, refined_md)

        # Soft updates from plan2
        if plan2:
            need_img = bool(plan2.get("need_images", need_img))
            need_yt = bool(plan2.get("need_youtube", need_yt))
            img_q = (plan2.get("image_query") or img_q).strip()
            yt_q = (plan2.get("youtube_query") or yt_q).strip()
            web_q2 = (plan2.get("web_query") or "").strip()
            if web_q2:
                web_q = enforce_web_query_constraints(message, web_q2)

        # ✅ Re-apply tool budget to stop model from forcing media unexpectedly
        need_web, need_img, need_yt = apply_tool_budget(user_flags, need_web, need_img, need_yt, is_first_turn=is_first_turn)

        # ✅ Explicit user intent / learning intent should win even if plan2 says need_youtube=false.
        # (Otherwise the model can accidentally disable YouTube and we never fetch results.)
        if user_flags.get("want_youtube") or user_flags.get("auto_youtube") or FORCE_YOUTUBE:
            need_yt = True
        if user_flags.get("want_images") or user_flags.get("auto_images") or FORCE_IMAGES:
            need_img = True

        # Fallbacks again
        if need_img and not img_q:
            img_q = make_fallback_query(context_query_base or base_topic or message, max_len=120)
        if need_yt and not yt_q:
            yt_q = make_fallback_query(context_query_base or base_topic or message, max_len=120)

        # Debug: show the final routed flags/queries after the second pass adjustments.
        try:
            print(
                "SECOND PASS ROUTED FLAGS:",
                {
                    "need_web": need_web,
                    "need_web_reason": need_web_reason,
                    "web_q": web_q,
                    "need_img": need_img,
                    "img_q": img_q,
                    "need_yt": need_yt,
                    "yt_q": yt_q,
                    "answer_len": len(answer_md),
                    "plan2_has": sorted(list((plan2 or {}).keys()))[:12],
                },
                flush=True,
            )
        except Exception:
            pass

    # 4) Images
    if need_img and img_q:
        # Prefer deriving multiple specific queries from curated titles
        def _is_generic(q: str) -> bool:
            s = (q or "").strip().lower()
            return not s or re.match(r"^(images?|pictures?|photos?)\b", s or "") is not None or len(s) < 8

        title_list: List[str] = []
        try:
            title_list = extract_titles_from_answer(answer_md) or []
        except Exception:
            title_list = []
        if (not title_list) and session_id:
            try:
                last_resp_text = await fetch_last_assistant_message(session_id)
                title_list = extract_titles_from_answer(last_resp_text) or []
            except Exception:
                title_list = []

        img_results: List[Dict[str, str]] = []
        if title_list and _is_generic(img_q):
            # Build per-title queries; stop when we have enough images
            queries = [
                f"{t} poster stills promotional photos cast"[:120]
                for t in title_list[:5]
            ]
            seen_urls: set = set()
            for q in queries:
                chunk = await fast_images(q, num=4)
                # Filter and de-duplicate
                for it in chunk:
                    u = it.get("image_url") or ""
                    if not _is_displayable_image_url(u):
                        continue
                    if u and u not in seen_urls:
                        img_results.append(it)
                        seen_urls.add(u)
                if len(img_results) >= 4:
                    break
            # If still thin, fall back to the original img_q
            if len(img_results) < 2:
                base_q = re.sub(r"\b(photos?|images?)\b", "", img_q, flags=re.I).strip()
                fallback_q = f"{base_q} wallpaper hd"[:120]
                more = await fast_images(fallback_q, num=6)
                more = [it for it in more if _is_displayable_image_url(it.get("image_url", ""))]
                seen_urls = {it.get("image_url") for it in img_results}
                for it in more:
                    u = it.get("image_url")
                    if u and u not in seen_urls:
                        img_results.append(it)
                        seen_urls.add(u)
        else:
            # Single-query path (non-generic or no titles)
            img_results = await fast_images(img_q, num=4)
            img_results = [it for it in img_results if _is_displayable_image_url(it.get("image_url", ""))]
            if len(img_results) < 2:
                base_q = re.sub(r"\b(photos?|images?)\b", "", img_q, flags=re.I).strip()
                fallback_q = f"{base_q} wallpaper hd"[:120]
                more = await fast_images(fallback_q, num=6)
                more = [it for it in more if _is_displayable_image_url(it.get("image_url", ""))]
                seen_urls = {it.get("image_url") for it in img_results}
                for it in more:
                    u = it.get("image_url")
                    if u and u not in seen_urls:
                        img_results.append(it)
                        seen_urls.add(u)
        print("IMAGE RESULTS:", len(img_results), flush=True)
        for it in img_results[:4]:
            if not USE_SUPABASE_STORAGE_FOR_IMAGES:
                images.append(ImageItem(
                    url=it.get("image_url") or "",
                    alt=it.get("title") or img_q,
                    source_url=it.get("page_url") or it.get("image_url"),
                ))
                continue

            # Upload concurrently for speed
            async def _upload_one(item: Dict[str, str]) -> Optional[ImageItem]:
                up = await supabase_upload_image_from_url(item.get("image_url", ""), filename_hint=img_q)
                if up:
                    up.alt = item.get("title") or img_q
                    up.source_url = item.get("page_url") or item.get("image_url")
                return up
            upload_tasks = [asyncio.create_task(_upload_one(it)) for it in img_results[:4]]
            uploaded_list = await asyncio.gather(*upload_tasks, return_exceptions=True)
            for up in uploaded_list:
                if isinstance(up, Exception) or not up:
                    continue
                images.append(up)

        images = [im for im in images if im.url]

    # 5) YouTube
    if need_yt and yt_q:
        if not YOUTUBE_API_KEY:
            print("YOUTUBE DISABLED: missing YOUTUBE_API_KEY", flush=True)
        youtube = await fast_youtube(yt_q, num=2)
        print("YOUTUBE RESULTS:", len(youtube), flush=True)
        # If none after filtering, disable embed to avoid broken UI but keep sources promotion
        if not youtube:
            need_yt = False

    # Ensure no plan/schema JSON leaks into the visible answer
    answer_md = strip_plan_json_leak(answer_md)

    # Optional: prepend a short line when user explicitly asked for media
    try:
        if answer_md and (images or youtube):
            # Only when the user explicitly asked (not when auto_youtube kicks in).
            if user_flags.get("want_youtube") or user_flags.get("want_images"):
                media_parts: List[str] = []
                if youtube:
                    media_parts.append("videos")
                if images:
                    media_parts.append("images")
                if media_parts:
                    if len(media_parts) == 2:
                        media_phrase = " and ".join(media_parts)
                    else:
                        media_phrase = media_parts[0]
                    lead_in = f"Sure, here are some {media_phrase} I found based on your request.\n\n"
                    answer_md = lead_in + answer_md
    except Exception:
        pass

    # 6) Final fallback if answer_md still empty
    if not answer_md:
        answer_md = (
            "I couldn’t generate the full response just now, but I’ve gathered the requested resources below. "
            "Try asking again or rephrasing and I’ll retry."
        )

    # 7) Remove inline citation tokens; sources are presented separately in payload.sources
    answer_md = cleanup_model_text(answer_md)
    answer_md = strip_meta_prompts(answer_md)
    answer_md = normalize_markdown_spacing(answer_md)
    answer_md = enhance_markdown_for_ui(answer_md)

    # 8) Promote media (YouTube/images) into generic sources for clickable links in UI
    try:
        if os.getenv("PROMOTE_MEDIA_TO_SOURCES", "1") == "1":
            merged: Dict[str, SourceItem] = {}
            for s in sources or []:
                if s.url:
                    merged[s.url.lower()] = s
            for y in youtube or []:
                url = y.url or (f"https://www.youtube.com/watch?v={y.video_id}" if getattr(y, 'video_id', None) else "")
                if url and url.lower() not in merged:
                    title = y.title or "YouTube video"
                    merged[url.lower()] = SourceItem(title=title, url=url, snippet=None, icon_url=y.thumbnail_url)
            for im in images or []:
                url = im.source_url or im.url
                if url and url.lower() not in merged:
                    title = im.alt or "Image"
                    merged[url.lower()] = SourceItem(title=title, url=url, snippet=None, icon_url=None)
            sources = list(merged.values())
    except Exception:
        # Never fail the request because of promotion logic
        pass

    return AssistantPayload(
        answer_markdown=answer_md,
        sources=sources,
        images=images,
        youtube=youtube,
        source_count=len(sources or []),
        cite_available=bool(sources),
        cite_label=(f"Sources ({len(sources)})" if sources else None),
    )

# -----------------------
# Persist assistant payload into chat_messages.meta (optional)
# -----------------------
async def persist_assistant_message(
    session_id: str,
    user_id: str,
    answer_md: str,
    payload: AssistantPayload,
    model_used: str = "llama2-cloudrag",
):
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and session_id):
        return

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    body = {
        "session_id": session_id,
        "user_id": user_id,
        "role": "assistant",
        "content": answer_md,
        "display_name": "AskVox",
        "meta": payload.model_dump(),
    }

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            await client.post(f"{SUPABASE_URL}/rest/v1/chat_messages", headers=headers, json=body)
        except Exception as e:
            print(f"⚠️ Failed to insert assistant chat_message: {e}", flush=True)


def detect_domain_for_message(message: str, requested_domain: Optional[str]) -> str:
    requested = (requested_domain or "").strip()
    if requested and validate_domain(requested):
        return requested
    try:
        return classify_domain(message or "")
    except Exception:
        return "general"


async def update_query_domain(query_id: str, detected_domain: str) -> None:
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and query_id and detected_domain):
        return

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    url = f"{SUPABASE_URL}/rest/v1/queries?id=eq.{query_id}"
    body = {"detected_domain": detected_domain}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.patch(url, headers=headers, json=body)
            if resp.status_code >= 400:
                print("⚠️ Failed to update query domain:", resp.status_code, resp.text[:200], flush=True)
    except Exception as e:
        print(f"⚠️ Failed to update query domain: {e}", flush=True)

# -----------------------
# ENDPOINT
# -----------------------
@router.post("/cloud_plus", response_model=ChatResponse)
async def chat_cloud_plus(req: ChatRequest, request: Request):
    from app.services.rate_limit import enforce_chat_rate_limit

    await enforce_chat_rate_limit(request, req.user_id)
    article_title = req.article_title
    article_url = req.article_url
    article_context = None

    if req.session_id:
        article_context = await fetch_session_article_context(req.session_id)
        article_title = article_title or (article_context or {}).get("title")
        article_url = article_url or (article_context or {}).get("url")

    detected_domain = detect_domain_for_message(req.message, req.domain)
    if req.query_id:
        await update_query_domain(req.query_id, detected_domain)

    payload = await generate_cloud_structured(
        req.message,
        req.history,
        article_title=article_title,
        article_url=article_url,
        article_context=article_context,
        session_id=req.session_id,
        user_id=req.user_id,
        domain=detected_domain,
        max_history=4 if req.user_id else 2,
    )

    # Persist response to Supabase 'responses' table if query_id provided
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and req.query_id:
        headers = {
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp_body = {
                    "query_id": req.query_id,
                    "response_text": payload.answer_markdown,
                    "model_used": "llama2-cloudrag",
                }
                r = await client.post(
                    f"{SUPABASE_URL}/rest/v1/responses",
                    headers=headers,
                    json=resp_body,
                )
                if r.status_code >= 400:
                    alt_body = {
                        "query_id": req.query_id,
                        "content": payload.answer_markdown,
                        "model_used": "llama2-cloudrag",
                    }
                    r2 = await client.post(
                        f"{SUPABASE_URL}/rest/v1/responses",
                        headers=headers,
                        json=alt_body,
                    )
                    if r2.status_code >= 400:
                        print(
                            "⚠️ Failed to insert response into Supabase:",
                            r.status_code,
                            r.text[:300],
                            "| alt",
                            r2.status_code,
                            r2.text[:300],
                            flush=True,
                        )
                    else:
                        print("✅ Inserted response (alt schema)", flush=True)
                else:
                    print("✅ Inserted response", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to insert response into Supabase: {e}", flush=True)

    if req.session_id and req.user_id:
        await persist_assistant_message(
            session_id=req.session_id,
            user_id=req.user_id,
            answer_md=payload.answer_markdown,
            payload=payload,
            model_used="llama2-cloudrag",
        )

    return ChatResponse(answer=payload.answer_markdown, payload=payload)
