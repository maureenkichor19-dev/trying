# Internet RAG Service

Enhanced Internet RAG implementation with deep content extraction and semantic reranking.

## Features

- **Multi-key Tavily rotation**: Automatically rotates through up to 5 Tavily API keys when rate-limited
- **Deep content extraction**: Uses Jina Reader API to extract full page content from discovered URLs
- **Smart chunking**: Intelligently splits long content while preserving paragraph boundaries
- **Semantic reranking**: Uses Jina Reranker API to rank chunks by relevance to the query
- **Rich evidence building**: Creates comprehensive evidence blocks for LLM context

## Environment Variables

### Required

- `TAVILY_API_KEY` - Primary Tavily API key for web search

### Optional

#### Additional Tavily Keys (for rotation)
- `TAVILY_API_KEY_2` - Second Tavily API key
- `TAVILY_API_KEY_3` - Third Tavily API key
- `TAVILY_API_KEY_4` - Fourth Tavily API key
- `TAVILY_API_KEY_5` - Fifth Tavily API key

#### Jina API (for deep extraction and reranking)
- `JINA_API_KEY` - Jina AI API key (for Reader and Reranker)

#### Content Processing
- `INTERNET_RAG_CHUNK_SIZE` - Target chunk size in characters (default: 800)
- `INTERNET_RAG_TOP_K` - Number of top chunks to return after reranking (default: 6)
- `INTERNET_RAG_MAX_EXTRACT_URLS` - How many URLs to deep-extract via Jina Reader (default: 5)
- `MAX_EVIDENCE_CHARS` - Maximum characters in evidence block (default: 1400)

#### Timeouts
- `JINA_READER_TIMEOUT` - Timeout for Jina Reader API per URL in seconds (default: 12)
- `JINA_RERANKER_TIMEOUT` - Timeout for Jina Reranker API in seconds (default: 8)

## API

### `internet_rag_search_and_extract(query, max_sources=6, chunk_size=None, top_k=None)`

Main function to search and extract internet content.

**Parameters:**
- `query` (str): User's search query
- `max_sources` (int): Maximum number of sources to return
- `chunk_size` (int): Target chunk size (uses env default if None)
- `top_k` (int): Number of top chunks after reranking (uses env default if None)

**Returns:**
- Tuple of `(sources, evidence_chunks)`
  - `sources`: List of SourceItem objects with metadata
  - `evidence_chunks`: List of dicts with 'source_index' and 'content'

### `build_web_evidence_block(sources, evidence_chunks, max_chars=None)`

Formats sources and evidence into a structured block for LLM prompts.

**Parameters:**
- `sources` (List[SourceItem]): List of source items
- `evidence_chunks` (List[Dict]): List of evidence chunk dicts
- `max_chars` (int): Maximum character limit (uses env default if None)

**Returns:**
- Formatted evidence block string with [SOURCES] and [EVIDENCE EXCERPTS] sections

## Architecture

### Key Rotation

The service maintains a `TavilyKeyState` object that tracks:
- Current key index
- Disabled keys with cooldown timestamps
- Automatic rotation on 429 (rate limit) or 403 (exhausted) responses

Keys are re-enabled after a 60-second cooldown period.

### Content Extraction Pipeline

1. **Tavily Search**: Query Tavily API with `include_raw_content=True` and `search_depth="advanced"`
2. **URL Discovery**: Collect top URLs from search results
3. **Deep Extraction**: Concurrently fetch full content via Jina Reader API for top N URLs
4. **Smart Chunking**: Split extracted content into semantic chunks (800-1200 chars)
5. **Reranking**: Use Jina Reranker to rank chunks by semantic relevance
6. **Evidence Building**: Format top chunks into structured evidence block

### Error Handling

- Graceful degradation: Falls back to Tavily snippets if extraction fails
- Timeout protection: Configurable timeouts for external API calls
- Concurrent extraction: Uses `asyncio.gather()` for parallel processing
- API rotation: Automatic failover between multiple Tavily keys

## Testing

Run the test suite:

```bash
python3 test_internet_rag.py
```

Tests cover:
- Import validation
- Key rotation logic
- Smart chunking algorithm
- Evidence block formatting
