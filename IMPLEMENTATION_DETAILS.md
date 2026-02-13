# Implementation Details - Multimodal Pipeline Fixes

## Code Changes Summary

### Constants Added
```python
# Line ~203: Pre-fetch timeout for time-sensitive queries
PRE_FETCH_TIMEOUT_SEC = 15.0

# Line ~785: Time signal pattern for real-time information needs
_TIME_SIGNAL_RE = re.compile(
    r"\b(latest|newest|recent|current|now|today|this year|trending|airing|released|premiered|upcoming|new)\b",
    re.I
)
```

### Functions Modified

#### 1. `fast_tavily` (Line 666)
**Before**: `timeout_sec: float = 4.5`
**After**: `timeout_sec: float = 20.0`
**Impact**: Allows complete deep extraction from Tavily API

#### 2. `build_runpod_user_prompt` (Line 367)
**Added**: Evidence usage instruction when web_evidence_block is provided
```python
if safe_evidence:
    user_context_parts.append(
        "USE THE FOLLOWING WEB EVIDENCE to provide accurate, up-to-date information. "
        "Include specific names, dates, and details from the evidence. "
        "Do NOT mention [SOURCES] or [EVIDENCE EXCERPTS] labels in your answer."
    )
    user_context_parts.append(safe_evidence)
```

#### 3. `build_second_pass_prompt_chat` (Line 507)
**Major restructure**: All context now INSIDE user turn
```python
# Before: Context was AFTER assistant header
p += "<|start_header_id|>assistant<|end_header_id|>\n"
if safe_article:
    p += "[ARTICLE_CONTEXT]\n" + safe_article + "\n\n"
# ... model treated this as its own output

# After: Context is IN user message
user_parts: List[str] = []
if safe_article:
    user_parts.append(f"[ARTICLE_CONTEXT]\n{safe_article}")
# ... then build user payload
p += "<|start_header_id|>user<|end_header_id|>\n"
p += f"{user_payload}<|eot_id|>"
p += "<|start_header_id|>assistant<|end_header_id|>\n"
```

#### 4. `cleanup_model_text` (Line 1561)
**Enhanced**: Strip all leaked internal labels
```python
# Added labels to strip (both middle and start of text):
- [EVIDENCE EXCERPTS]
- [EVIDENCE_EXCERPTS]
- [REVISION_RULES]
- [ARTICLE_CONTEXT]
- [RAG_CONTEXT]
- [ORIGINAL_ANSWER]

# Also handles labels at very start (no preceding newline)
for label in [...]:
    if out.strip().startswith(label):
        out = out.strip()[len(label):].strip()
```

#### 5. `generate_cloud_structured` (Line 2298)
**Major addition**: Pre-fetch logic (~45 lines, starting line 2600)

Key components:
1. **Detection** (Line ~2620):
   ```python
   has_time_signal = bool(_YEAR_WEB_INTENT_RE.search(msg_text)) or bool(_TIME_SIGNAL_RE.search(msg_text))
   has_factual_intent = bool(_FACTUAL_WEB_INTENT_RE.search(msg_text))
   is_who = is_who_is_question(msg_text)
   should_prefetch = has_time_signal or has_factual_intent or is_who or explicit_web or FORCE_WEB_SOURCES
   ```

2. **Fetch** (Line ~2633):
   ```python
   pre_fetched_sources, pre_fetched_chunks = await asyncio.wait_for(
       internet_rag_search_and_extract(prefetch_query, max_sources=6),
       timeout=PRE_FETCH_TIMEOUT_SEC
   )
   ```

3. **Include in first draft** (Line ~2641):
   ```python
   prompt1 = build_runpod_user_prompt(
       message,
       trimmed_history,
       rag_block=rag_block,
       web_evidence_block=pre_fetched_evidence_block,  # ✅ NOW INCLUDED!
       article_block=article_block,
       chat_mode=chat_flag,
       learning_preference=learning_pref,
   )
   ```

4. **Reuse in second pass** (Line ~2920):
   ```python
   if pre_fetched_sources and pre_fetched_chunks:
       sources = pre_fetched_sources
       evidence_chunks = pre_fetched_chunks
       print(f"♻️ [WebSources] Reusing pre-fetched evidence ({len(sources)} sources)", flush=True)
   else:
       # Fallback to new fetch
   ```

## Validation

### Test Coverage
- ✅ All 5 fixes validated with automated tests
- ✅ Syntax checking passes
- ✅ Code structure verified with AST parsing
- ✅ Pattern matching confirms all changes in place

### Edge Cases Handled
1. **Pre-fetch timeout**: Falls back gracefully, will retry in second pass
2. **No web providers**: Pre-fetch skipped, existing behavior preserved
3. **Smalltalk queries**: Pre-fetch skipped, no unnecessary API calls
4. **Generic queries**: Only prefetch when time-sensitive/factual

### Performance Considerations
1. **15s timeout**: Balances thoroughness with responsiveness
2. **Result reuse**: Prevents duplicate API calls to Tavily
3. **Selective pre-fetch**: Only for queries that need real-time data
4. **Fail-soft**: Never breaks the pipeline, always has fallback

## Migration Notes

### No Breaking Changes
- All function signatures maintained
- `web_evidence_block` parameter already existed
- Existing behavior preserved for non-time-sensitive queries
- No new dependencies required

### Configuration
No new environment variables required. Existing variables work:
- `TAVILY_API_KEY`: For web evidence
- `FORCE_WEB_SOURCES`: Forces pre-fetch for all queries
- `MODEL_KNOWLEDGE_CUTOFF_YEAR`: Used to detect future years (default: 2023)

### Deployment
Simply deploy the updated `MultimodalLlamachat.py`:
1. No database migrations needed
2. No configuration changes needed
3. No service restarts beyond normal deployment
4. Backward compatible with existing clients

## Expected Behavior Changes

### Before Fix
**Query**: "latest kdrama 2026"
1. First LLM call: No web evidence → generates from 2023 knowledge → mentions 2024 dramas
2. Check if needs web: Maybe not (answer is confident)
3. If second pass triggered: Evidence added but with label leakage

**Result**: Stale 2024 data, possible label leakage

### After Fix
**Query**: "latest kdrama 2026"
1. Pre-fetch detection: ✅ "latest" + "2026" → time-sensitive
2. Fetch evidence: Gets actual 2026 K-drama data from Tavily
3. First LLM call: WITH web evidence → generates current answer
4. Second pass: Reuses same evidence (no duplicate API call)

**Result**: Current 2026 data, clean markdown output

## Metrics to Monitor

### Success Indicators
- Increase in citations `[[cite:N]]` in responses
- Decrease in "I don't have current data" deflections
- User satisfaction for time-sensitive queries
- Reduction in label leakage reports

### Performance Metrics
- Pre-fetch hit rate (queries that trigger pre-fetch)
- Pre-fetch success rate (successful evidence retrieval)
- Evidence reuse rate (second pass reusing pre-fetched data)
- Average response time (should not significantly increase due to selective pre-fetch)

### Monitoring
Log messages added for easy tracking:
- `🌐 [PreFetch] Pre-fetching web evidence for: {query}`
- `✅ [PreFetch] Got {N} sources, {M} chunks, evidence block: {chars} chars`
- `⚠️ [PreFetch] No results for: {query}`
- `⚠️ [PreFetch] Failed (will retry in second pass): {error}`
- `♻️ [WebSources] Reusing pre-fetched evidence ({N} sources)`
