# Multimodal Pipeline Fixes - Implementation Summary

## Overview
Fixed 3 critical architectural bugs in the Internet RAG pipeline that prevented web evidence from effectively reaching the LLM.

## Bugs Fixed

### Bug 1: First draft prompt had NO web evidence
**Problem**: The LLM generated from stale 2023 knowledge instead of using real-time web evidence.

**Solution**: Pre-fetch web evidence BEFORE the first draft for time-sensitive and factual queries.
- Added detection for time signals, factual intent, and "who is" questions
- Call `internet_rag_search_and_extract` with 15s timeout before first LLM call
- Pass pre-fetched evidence to both `build_runpod_user_prompt` and `build_prompt`
- Reuse pre-fetched results in second pass to avoid duplicate API calls

### Bug 2: Second-pass prompt put evidence AFTER assistant header
**Problem**: Model treated evidence as its own output, causing label leakage.

**Solution**: Moved ALL context INSIDE the user turn (BEFORE assistant header).
- Build complete user payload with article, original answer, evidence, and revision instructions
- Everything now appears before `<|start_header_id|>assistant<|end_header_id|>`
- Model now sees context as input, not as its own prior output

### Bug 3: `cleanup_model_text` didn't strip leaked prompt fragments
**Problem**: Internal labels leaked into user-facing output.

**Solution**: Comprehensive label stripping.
- Strip `[SOURCES]`, `[EVIDENCE EXCERPTS]`, `[REVISION_RULES]`, `[ARTICLE_CONTEXT]`, `[RAG_CONTEXT]`, `[ORIGINAL_ANSWER]`
- Handle both mid-text and start-of-text occurrences
- Remove leaked instruction lines using regex patterns

## Additional Improvements

### Fix 4: Added evidence instruction in `build_runpod_user_prompt`
When web evidence is provided, explicitly instruct the model to:
- Use the evidence to provide accurate, up-to-date information
- Include specific names, dates, and details
- NOT mention internal labels in output

### Fix 5: Increased `fast_tavily` timeout
Changed default timeout from 4.5s to 20.0s to allow for complete deep extraction.

## Files Modified
- `MultimodalLlamachat.py` (all 5 fixes)

## Testing
Created comprehensive validation test (`test_multimodal_fixes.py`) that verifies:
- All code changes are in place
- Function signatures are correct
- Logic flow is properly structured
- All 5 fixes pass validation

## Expected Improvements
1. **Accurate Real-Time Data**: For queries like "latest kdrama 2026", the model now sees actual 2026 data on first pass
2. **No Label Leakage**: Clean markdown output without internal labels
3. **Better Citations**: Evidence properly integrated with [[cite:N]] tokens
4. **Faster Response**: Reusing pre-fetched data avoids duplicate API calls
5. **More Reliable**: 20s timeout ensures complete evidence extraction

## Backward Compatibility
- All function signatures unchanged (web_evidence_block was already a parameter)
- No new dependencies required
- Fails gracefully if pre-fetch times out or fails
- Existing behavior preserved for non-time-sensitive queries
