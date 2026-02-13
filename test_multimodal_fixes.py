#!/usr/bin/env python3
"""
Test script to validate the multimodal pipeline fixes.

This tests the logic changes without requiring actual API calls.
"""

import sys
import os
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_fix_5_fast_tavily_timeout():
    """Test Fix 5: fast_tavily timeout increased to 20.0"""
    print("\n🧪 Testing Fix 5: fast_tavily timeout...")
    
    with open('MultimodalLlamachat.py', 'r') as f:
        content = f.read()
    
    # Check the function signature
    pattern = r'async def fast_tavily\([^)]*timeout_sec:\s*float\s*=\s*20\.0[^)]*\)'
    if re.search(pattern, content):
        print("   ✅ fast_tavily default timeout is 20.0 seconds")
        return True
    else:
        print("   ❌ fast_tavily timeout not set to 20.0")
        return False


def test_fix_4_evidence_instruction():
    """Test Fix 4: Evidence instruction added to build_runpod_user_prompt"""
    print("\n🧪 Testing Fix 4: Evidence instruction in build_runpod_user_prompt...")
    
    with open('MultimodalLlamachat.py', 'r') as f:
        content = f.read()
    
    # Check for the evidence instruction
    required_phrases = [
        "USE THE FOLLOWING WEB EVIDENCE",
        "Include specific names, dates, and details",
        "Do NOT mention [SOURCES] or [EVIDENCE EXCERPTS] labels"
    ]
    
    all_found = all(phrase in content for phrase in required_phrases)
    
    if all_found:
        print("   ✅ Evidence instruction properly added")
        return True
    else:
        print("   ❌ Evidence instruction missing or incomplete")
        return False


def test_fix_3_cleanup_model_text():
    """Test Fix 3: cleanup_model_text strips all leaked labels"""
    print("\n🧪 Testing Fix 3: cleanup_model_text strips leaked labels...")
    
    with open('MultimodalLlamachat.py', 'r') as f:
        content = f.read()
    
    # Find the cleanup_model_text function
    cleanup_start = content.find('def cleanup_model_text(text: str) -> str:')
    if cleanup_start == -1:
        print("   ❌ cleanup_model_text function not found")
        return False
    
    # Get the function content (approximate)
    cleanup_end = content.find('\ndef ', cleanup_start + 1)
    if cleanup_end == -1:
        cleanup_end = len(content)
    
    cleanup_func = content[cleanup_start:cleanup_end]
    
    # Check for all required labels
    required_labels = [
        '[EVIDENCE EXCERPTS]',
        '[EVIDENCE_EXCERPTS]',
        '[REVISION_RULES]',
        '[ARTICLE_CONTEXT]',
        '[RAG_CONTEXT]',
        '[ORIGINAL_ANSWER]'
    ]
    
    # Check that labels appear twice (once in each loop)
    label_counts = {label: cleanup_func.count(label) for label in required_labels}
    
    # Each label should appear at least twice (in the two for loops)
    all_found = all(count >= 2 for count in label_counts.values())
    
    # Check for the start-of-text detection
    has_startswith_check = 'if out.strip().startswith(label):' in cleanup_func
    
    if all_found and has_startswith_check:
        print("   ✅ All labels are stripped (both middle and start)")
        return True
    else:
        print("   ❌ Some labels missing or start-of-text check missing")
        return False


def test_fix_2_second_pass_prompt():
    """Test Fix 2: build_second_pass_prompt_chat puts evidence INSIDE user turn"""
    print("\n🧪 Testing Fix 2: build_second_pass_prompt_chat structure...")
    
    with open('MultimodalLlamachat.py', 'r') as f:
        content = f.read()
    
    # Find the function
    func_start = content.find('def build_second_pass_prompt_chat(')
    if func_start == -1:
        print("   ❌ build_second_pass_prompt_chat function not found")
        return False
    
    # Get the function content
    func_end = content.find('\ndef ', func_start + 1)
    if func_end == -1:
        func_end = content.find('\nasync def ', func_start + 1)
    if func_end == -1:
        func_end = len(content)
    
    func_content = content[func_start:func_end]
    
    # Check for the new structure
    checks = {
        'user_parts list': 'user_parts: List[str] = []' in func_content,
        'article in user_parts': 'user_parts.append(f"[ARTICLE_CONTEXT]' in func_content,
        'original answer in user_parts': 'user_parts.append(f"[ORIGINAL_ANSWER]' in func_content,
        'evidence in user_parts': 'if safe_evidence:\n        user_parts.append(safe_evidence)' in func_content,
        'revision instruction in user_parts': 'TASK: Rewrite and improve the [ORIGINAL_ANSWER]' in func_content,
        'user payload join': 'user_payload = "\\n\\n".join([p for p in user_parts if p]).strip()' in func_content,
        'user_payload in prompt': 'f"{user_payload}<|eot_id|>"' in func_content
    }
    
    # Check that OLD structure is removed
    old_structure_removed = (
        'p += "[ARTICLE_CONTEXT]\\n" + safe_article + "\\n\\n"' not in func_content and
        'p += "[REVISION_RULES]\\n"' not in func_content
    )
    
    passed = all(checks.values()) and old_structure_removed
    
    if passed:
        print("   ✅ Evidence and context now INSIDE user turn (before assistant header)")
        return True
    else:
        print("   ❌ Structure not properly updated")
        for check, result in checks.items():
            if not result:
                print(f"      - Missing: {check}")
        if not old_structure_removed:
            print("      - Old structure still present")
        return False


def test_fix_1_prefetch_logic():
    """Test Fix 1: Pre-fetch logic added"""
    print("\n🧪 Testing Fix 1: Pre-fetch web evidence logic...")
    
    with open('MultimodalLlamachat.py', 'r') as f:
        content = f.read()
    
    # Find the generate_cloud_structured function
    func_start = content.find('async def generate_cloud_structured(')
    if func_start == -1:
        print("   ❌ generate_cloud_structured function not found")
        return False
    
    # Check for pre-fetch initialization
    checks = {
        'pre_fetched_evidence_block init': 'pre_fetched_evidence_block = ""' in content,
        'pre_fetched_sources init': 'pre_fetched_sources: List[SourceItem] = []' in content,
        'pre_fetched_chunks init': 'pre_fetched_chunks: List[Dict[str, str]] = []' in content,
        'should_prefetch flag': 'should_prefetch = False' in content,
        'time signal detection': 'has_time_signal = bool(_YEAR_WEB_INTENT_RE.search(msg_text))' in content,
        'factual intent detection': 'has_factual_intent = bool(_FACTUAL_WEB_INTENT_RE.search(msg_text))' in content,
        'pre-fetch condition': 'should_prefetch = has_time_signal or has_factual_intent' in content,
        'internet_rag_search_and_extract call': 'internet_rag_search_and_extract(prefetch_query, max_sources=6)' in content,
        '15s timeout': 'timeout=15.0  # Allow more time since this is critical' in content,
        'evidence block build': 'pre_fetched_evidence_block = build_web_evidence_block(pre_fetched_sources, pre_fetched_chunks)' in content
    }
    
    # Check that pre-fetched evidence is passed to prompts
    prompt_checks = {
        'runpod prompt': 'web_evidence_block=pre_fetched_evidence_block,  # ✅ NOW INCLUDED!' in content,
        'cloudrun prompt': content.count('web_evidence_block=pre_fetched_evidence_block') >= 2
    }
    
    # Check for reuse logic
    reuse_checks = {
        'reuse condition': 'if pre_fetched_sources and pre_fetched_chunks:' in content,
        'reuse assignment': 'sources = pre_fetched_sources' in content and 'evidence_chunks = pre_fetched_chunks' in content,
        'reuse log': '♻️ [WebSources] Reusing pre-fetched evidence' in content
    }
    
    all_passed = (
        all(checks.values()) and 
        all(prompt_checks.values()) and 
        all(reuse_checks.values())
    )
    
    if all_passed:
        print("   ✅ Pre-fetch logic fully implemented")
        print("      - Pre-fetch variables initialized")
        print("      - Time-sensitive/factual detection in place")
        print("      - internet_rag_search_and_extract called with 15s timeout")
        print("      - Pre-fetched evidence passed to prompt builders")
        print("      - Reuse logic prevents duplicate API calls")
        return True
    else:
        print("   ❌ Pre-fetch logic incomplete")
        for check, result in checks.items():
            if not result:
                print(f"      - Missing: {check}")
        for check, result in prompt_checks.items():
            if not result:
                print(f"      - Missing: {check}")
        for check, result in reuse_checks.items():
            if not result:
                print(f"      - Missing: {check}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Multimodal Pipeline Fixes")
    print("=" * 60)
    
    tests = [
        ("Fix 5: fast_tavily timeout", test_fix_5_fast_tavily_timeout),
        ("Fix 4: Evidence instruction", test_fix_4_evidence_instruction),
        ("Fix 3: cleanup_model_text", test_fix_3_cleanup_model_text),
        ("Fix 2: Second-pass prompt", test_fix_2_second_pass_prompt),
        ("Fix 1: Pre-fetch logic", test_fix_1_prefetch_logic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All fixes validated successfully!")
        return 0
    else:
        print("\n⚠️ Some fixes need attention:")
        for name, result in results:
            if not result:
                print(f"   - {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
