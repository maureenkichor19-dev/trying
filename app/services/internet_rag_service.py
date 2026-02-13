"""
Enhanced Internet RAG Service with deep content extraction and semantic reranking.

Features:
- Multi-key Tavily API rotation (up to 5 keys)
- Deep content extraction via Jina Reader API
- Smart text chunking with paragraph preservation
- Semantic reranking via Jina Reranker API
- Rich evidence building for LLM context
"""

from __future__ import annotations
import os
import time
import asyncio
import httpx
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel

# Environment variables
TAVILY_API_KEYS = [
    os.getenv("TAVILY_API_KEY", "").strip(),
    os.getenv("TAVILY_API_KEY_2", "").strip(),
    os.getenv("TAVILY_API_KEY_3", "").strip(),
    os.getenv("TAVILY_API_KEY_4", "").strip(),
    os.getenv("TAVILY_API_KEY_5", "").strip(),
]
TAVILY_API_KEYS = [k for k in TAVILY_API_KEYS if k]  # Filter empty keys

JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
INTERNET_RAG_CHUNK_SIZE = int(os.getenv("INTERNET_RAG_CHUNK_SIZE", "800"))
INTERNET_RAG_TOP_K = int(os.getenv("INTERNET_RAG_TOP_K", "6"))
INTERNET_RAG_MAX_EXTRACT_URLS = int(os.getenv("INTERNET_RAG_MAX_EXTRACT_URLS", "5"))
JINA_READER_TIMEOUT = float(os.getenv("JINA_READER_TIMEOUT", "12"))
JINA_RERANKER_TIMEOUT = float(os.getenv("JINA_RERANKER_TIMEOUT", "8"))
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "3500"))

# SourceItem model - compatible with MultimodalLlamachat.py
class SourceItem(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    icon_url: Optional[str] = None

# Key rotation state
class TavilyKeyState:
    """Manages API key rotation and cooldown."""
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.current_index = 0
        self.disabled_until: Dict[int, float] = {}  # key_index -> time.time() when re-enabled
        self.cooldown_seconds = 60  # Re-enable after 60 seconds
    
    def get_current_key(self) -> Optional[str]:
        """Get the current active key, skipping disabled ones."""
        if not self.keys:
            return None
        
        # Clean up expired cooldowns
        now = time.time()
        self.disabled_until = {k: v for k, v in self.disabled_until.items() if v > now}
        
        # Find next available key
        attempts = 0
        while attempts < len(self.keys):
            if self.current_index not in self.disabled_until:
                return self.keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.keys)
            attempts += 1
        
        return None  # All keys disabled
    
    def rotate_key(self, disable_current: bool = False):
        """Rotate to next key, optionally disabling the current one."""
        if disable_current and self.keys:
            self.disabled_until[self.current_index] = time.time() + self.cooldown_seconds
        self.current_index = (self.current_index + 1) % len(self.keys)

# Global key state
_tavily_key_state = TavilyKeyState(TAVILY_API_KEYS)


async def _tavily_search_with_rotation(query: str, max_results: int = 10) -> Dict[str, Any]:
    """
    Call Tavily API with automatic key rotation on rate limits.
    
    Returns the JSON response or raises an exception.
    """
    if not _tavily_key_state.keys:
        return {"results": []}
    
    url = "https://api.tavily.com/search"
    max_retries = len(_tavily_key_state.keys)
    
    for attempt in range(max_retries):
        api_key = _tavily_key_state.get_current_key()
        if not api_key:
            print("⚠️ All Tavily API keys are rate-limited", flush=True)
            break
        
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": True,  # Get full content
            "search_depth": "advanced",
        }
        
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(url, json=payload)
            
            if resp.status_code == 200:
                return resp.json() or {}
            elif resp.status_code in (429, 403):  # Rate limit or exhausted
                print(f"🔄 Tavily key exhausted (HTTP {resp.status_code}), rotating...", flush=True)
                _tavily_key_state.rotate_key(disable_current=True)
            else:
                print(f"❌ Tavily error: {resp.status_code} - {resp.text[:200]}", flush=True)
                _tavily_key_state.rotate_key(disable_current=False)  # Try next key
        except Exception as e:
            print(f"❌ Tavily exception: {e}", flush=True)
            _tavily_key_state.rotate_key(disable_current=False)
    
    return {"results": []}


async def _extract_content_via_jina_reader(url: str) -> Optional[str]:
    """
    Extract full page content using Jina Reader API.
    Handles both JSON responses (authenticated) and plain text (unauthenticated).
    """
    reader_url = f"https://r.jina.ai/{url}"
    headers: Dict[str, str] = {
        "X-Return-Format": "markdown",
    }
    
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
        headers["Accept"] = "application/json"
    
    try:
        async with httpx.AsyncClient(timeout=JINA_READER_TIMEOUT) as client:
            resp = await client.get(reader_url, headers=headers)
        
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "")
            
            # Try JSON parsing first (authenticated requests return JSON)
            if "application/json" in content_type:
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        # Jina Reader JSON: {"code": 200, "data": {"content": "...", "title": "..."}}
                        nested = data.get("data", {})
                        if isinstance(nested, dict):
                            content = nested.get("content", "")
                        else:
                            content = data.get("content", "") or str(data.get("data", ""))
                    else:
                        content = ""
                    if content and isinstance(content, str) and len(content.strip()) > 100:
                        print(f"✅ [Jina Reader] JSON extraction for {url[:60]}: {len(content)} chars", flush=True)
                        return content.strip()
                except Exception:
                    pass
            
            # Fallback: plain text/markdown response (unauthenticated or non-JSON)
            text = resp.text.strip()
            if text and len(text) > 100:
                print(f"✅ [Jina Reader] Text extraction for {url[:60]}: {len(text)} chars", flush=True)
                return text
            
            print(f"⚠️ [Jina Reader] Response too short for {url[:60]}: {len(text)} chars", flush=True)
            return None
        else:
            print(f"⚠️ [Jina Reader] HTTP {resp.status_code} for {url[:60]}", flush=True)
            return None
    except asyncio.TimeoutError:
        print(f"⏱️ [Jina Reader] Timeout for {url[:60]}", flush=True)
        return None
    except Exception as e:
        print(f"❌ [Jina Reader] Exception for {url[:60]}: {e}", flush=True)
        return None


def _smart_chunk_text(text: str, chunk_size: int = 800, max_chunk_size: int = 1200) -> List[str]:
    """
    Split text into chunks, preserving paragraph boundaries.
    
    Args:
        text: The text to chunk
        chunk_size: Target chunk size
        max_chunk_size: Maximum chunk size before forcing split
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Split into paragraphs
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_size = len(para)
        
        # If single paragraph is too large, split by sentences
        if para_size > max_chunk_size:
            sentences = para.split(". ")
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                # Only add period if sentence doesn't already end with one
                if not sent.endswith("."):
                    sent = sent + "."
                sent_size = len(sent) + 1  # Account for space separator
                
                if current_size + sent_size > max_chunk_size:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_size = sent_size
                else:
                    current_chunk.append(sent)
                    current_size += sent_size
        else:
            # Check if adding this paragraph would exceed max size
            if current_size + para_size > max_chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            elif current_size + para_size > chunk_size and current_chunk:
                # Exceeded target size, start new chunk
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # Account for \n\n
    
    # Add remaining content
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks


async def _rerank_chunks_via_jina(
    query: str, 
    chunks: List[Dict[str, Any]], 
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Rerank chunks by semantic relevance using Jina Reranker API.
    
    Args:
        query: The user's query
        chunks: List of chunk dicts with 'content' field
        top_k: Number of top chunks to return
    
    Returns:
        Reranked list of chunks (top_k most relevant)
    """
    if not chunks or not JINA_API_KEY:
        return chunks[:top_k]
    
    reranker_url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Prepare documents for reranking
    documents = [chunk.get("content", "") for chunk in chunks]
    
    payload = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }
    
    try:
        async with httpx.AsyncClient(timeout=JINA_RERANKER_TIMEOUT) as client:
            resp = await client.post(reranker_url, json=payload, headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            
            # Reorder chunks based on reranker scores
            reranked = []
            for result in results:
                idx = result.get("index")
                if idx is not None and 0 <= idx < len(chunks):
                    reranked.append(chunks[idx])
            
            return reranked[:top_k]
        else:
            print(f"⚠️ Jina Reranker error: {resp.status_code}", flush=True)
            return chunks[:top_k]
    except asyncio.TimeoutError:
        print("⏱️ Jina Reranker timeout, using original order", flush=True)
        return chunks[:top_k]
    except Exception as e:
        print(f"❌ Jina Reranker exception: {e}, using original order", flush=True)
        return chunks[:top_k]


async def internet_rag_search_and_extract(
    query: str,
    max_sources: int = 6,
    chunk_size: int = None,
    top_k: int = None,
) -> Tuple[List[SourceItem], List[Dict[str, str]]]:
    """
    Enhanced internet RAG with deep extraction and reranking.
    
    Args:
        query: User's search query
        max_sources: Maximum number of sources to return
        chunk_size: Target chunk size (default from env)
        top_k: Number of top chunks after reranking (default from env)
    
    Returns:
        Tuple of (sources, evidence_chunks)
        - sources: List of SourceItem objects with metadata
        - evidence_chunks: List of dicts with 'source_index' and 'content'
    """
    if not query.strip():
        return [], []
    
    chunk_size = chunk_size or INTERNET_RAG_CHUNK_SIZE
    top_k = top_k or INTERNET_RAG_TOP_K
    
    # Step 1: Tavily search with raw content
    tavily_data = await _tavily_search_with_rotation(query, max_results=max_sources)
    results = tavily_data.get("results") or []
    
    if not results:
        return [], []
    
    # Step 2: Build sources and collect raw content from Tavily
    sources: List[SourceItem] = []
    tavily_raw_contents: Dict[int, str] = {}  # idx -> raw_content
    urls_needing_jina: List[Tuple[int, str, str]] = []  # URLs where Tavily didn't give enough
    
    for idx, result in enumerate(results[:max_sources], start=1):
        title = (result.get("title") or "Source").strip()
        url = (result.get("url") or "").strip()
        snippet = (result.get("content") or "").strip()
        raw_content = (result.get("raw_content") or "").strip()
        
        if not url:
            continue
        
        sources.append(SourceItem(title=title, url=url, snippet=snippet or None))
        
        # Use Tavily raw_content if it's rich enough (>= 300 chars)
        if raw_content and len(raw_content) >= 300:
            tavily_raw_contents[idx] = raw_content
            print(f"📄 [InternetRAG] Using Tavily raw_content for [{idx}] {title[:50]}: {len(raw_content)} chars", flush=True)
        elif len(urls_needing_jina) < INTERNET_RAG_MAX_EXTRACT_URLS:
            urls_needing_jina.append((idx, url, title))
    
    print(f"🔍 [InternetRAG] Tavily returned {len(results)} results | {len(tavily_raw_contents)} have rich raw_content | {len(urls_needing_jina)} need Jina extraction", flush=True)
    
    # Step 3: Build chunks from Tavily raw_content + Jina Reader extraction
    all_chunks: List[Dict[str, Any]] = []
    
    # 3a: Chunk Tavily raw_content directly
    for idx, raw_text in tavily_raw_contents.items():
        src = sources[idx - 1] if idx <= len(sources) else None
        title = src.title if src else "Source"
        url = src.url if src else ""
        text_chunks = _smart_chunk_text(raw_text, chunk_size=chunk_size)
        for chunk_text in text_chunks:
            all_chunks.append({
                "source_index": idx,
                "url": url,
                "title": title,
                "content": chunk_text,
            })
    
    # 3b: Extract remaining URLs via Jina Reader (concurrent)
    if urls_needing_jina:
        extract_tasks = [_extract_content_via_jina_reader(url) for _, url, _ in urls_needing_jina]
        extracted_contents = await asyncio.gather(*extract_tasks, return_exceptions=True)
        
        for (idx, url, title), content in zip(urls_needing_jina, extracted_contents):
            if isinstance(content, Exception) or not content:
                # Fallback to Tavily snippet
                for src in sources:
                    if src.url == url and src.snippet:
                        all_chunks.append({
                            "source_index": idx,
                            "url": url,
                            "title": title,
                            "content": src.snippet[:chunk_size],
                        })
                        break
                continue
            
            text_chunks = _smart_chunk_text(content, chunk_size=chunk_size)
            for chunk_text in text_chunks:
                all_chunks.append({
                    "source_index": idx,
                    "url": url,
                    "title": title,
                    "content": chunk_text,
                })
    
    # Fallback if nothing worked
    if not all_chunks:
        for idx, src in enumerate(sources, start=1):
            if src.snippet:
                all_chunks.append({
                    "source_index": idx,
                    "url": src.url,
                    "title": src.title,
                    "content": src.snippet[:chunk_size],
                })
    
    print(f"📦 [InternetRAG] Created {len(all_chunks)} total chunks", flush=True)
    
    # Step 4: Rerank chunks by semantic relevance
    if all_chunks and JINA_API_KEY:
        reranked_chunks = await _rerank_chunks_via_jina(query, all_chunks, top_k)
    else:
        reranked_chunks = all_chunks[:top_k]
    
    print(f"🏆 [InternetRAG] Reranked to {len(reranked_chunks)} chunks (largest: {max(len(c.get('content','')) for c in reranked_chunks) if reranked_chunks else 0} chars)", flush=True)
    
    return sources, reranked_chunks


def build_web_evidence_block(
    sources: List[SourceItem],
    evidence_chunks: List[Dict[str, str]],
    max_chars: int = None,
) -> str:
    """
    Build formatted [SOURCES] + [EVIDENCE EXCERPTS] block for LLM prompt.
    
    Args:
        sources: List of SourceItem objects
        evidence_chunks: List of evidence dicts with 'source_index' and 'content'
        max_chars: Maximum character limit (default from env)
    
    Returns:
        Formatted evidence block string
    """
    max_chars = max_chars or MAX_EVIDENCE_CHARS
    
    if not sources:
        return ""
    
    lines = ["[SOURCES]"]
    for i, src in enumerate(sources[:6], start=1):
        lines.append(f"[{i}] Title: {src.title}\n    URL: {src.url}")
    
    # Add evidence excerpts
    excerpt_lines: List[str] = []
    if evidence_chunks:
        for j, chunk in enumerate(evidence_chunks[:8], start=1):
            source_idx = chunk.get("source_index") or ""
            content = (chunk.get("content") or "").strip()
            if not source_idx or not content:
                continue
            
            # Use letter suffix for multiple excerpts from same source
            suffix = chr(ord("a") + (j - 1) % 26)
            excerpt_lines.append(f"[{source_idx}{suffix}] {content[:800]}")
    else:
        # Fallback: use source snippets
        for i, src in enumerate(sources[:6], start=1):
            if src.snippet:
                excerpt_lines.append(f"[{i}a] {src.snippet[:600]}")
            if len(excerpt_lines) >= 4:
                break
    
    if excerpt_lines:
        lines.append("\n[EVIDENCE EXCERPTS]")
        lines.extend(excerpt_lines)
    
    block = "\n".join(lines).strip()
    
    # Enforce character limit
    if len(block) > max_chars:
        block = block[:max_chars] + "..."
    
    print(f"📋 [InternetRAG] Evidence block: {len(block)} chars, {len(excerpt_lines)} excerpts", flush=True)
    
    return block
