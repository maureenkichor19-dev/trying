#!/usr/bin/env python3
"""
Test script for internet_rag_service.py

This tests the core logic without requiring actual API calls.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work."""
    try:
        from app.services.internet_rag_service import (
            internet_rag_search_and_extract,
            build_web_evidence_block,
            SourceItem,
            TavilyKeyState,
            _smart_chunk_text,
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_tavily_key_rotation():
    """Test the key rotation logic."""
    from app.services.internet_rag_service import TavilyKeyState
    import time
    
    keys = ["key1", "key2", "key3"]
    state = TavilyKeyState(keys)
    
    # Test getting current key
    assert state.get_current_key() == "key1", "Should start with first key"
    
    # Test rotation without disabling
    state.rotate_key(disable_current=False)
    assert state.get_current_key() == "key2", "Should rotate to second key"
    
    # Test rotation with disabling (disables key2, moves to key3)
    state.rotate_key(disable_current=True)
    assert state.get_current_key() == "key3", "Should move to key3 after disabling key2"
    
    # Now disable key3 and rotate - should skip to key1 (key2 still disabled)
    state.rotate_key(disable_current=True)
    assert state.get_current_key() == "key1", "Should skip disabled key2 and key3, use key1"
    
    # Test disabling all keys
    state.rotate_key(disable_current=True)  # Disable key1
    state.rotate_key(disable_current=True)  # Try key2, it's disabled, rotate
    assert state.get_current_key() is None, "Should return None when all keys disabled"
    
    print("✅ Tavily key rotation logic works correctly")
    return True


def test_smart_chunking():
    """Test text chunking logic."""
    from app.services.internet_rag_service import _smart_chunk_text
    
    # Test empty text
    assert _smart_chunk_text("") == [], "Empty text should return empty list"
    
    # Test short text
    short_text = "This is a short paragraph."
    chunks = _smart_chunk_text(short_text, chunk_size=100)
    assert len(chunks) == 1, "Short text should be single chunk"
    assert chunks[0] == short_text, "Content should match"
    
    # Test long text that needs splitting
    long_text = "Paragraph 1.\n\n" + ("Paragraph 2 with lots of text. " * 50)
    chunks = _smart_chunk_text(long_text, chunk_size=500, max_chunk_size=800)
    assert len(chunks) > 1, "Long text should be split into multiple chunks"
    assert all(len(chunk) <= 850 for chunk in chunks), "Chunks should respect max size (with margin)"
    
    # Test that paragraphs are preserved when possible
    multi_para = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = _smart_chunk_text(multi_para, chunk_size=30)
    assert len(chunks) >= 2, "Should create multiple chunks"
    
    print("✅ Smart chunking logic works correctly")
    return True


def test_evidence_block_builder():
    """Test the evidence block formatting."""
    from app.services.internet_rag_service import build_web_evidence_block, SourceItem
    
    # Test empty sources
    assert build_web_evidence_block([], []) == "", "Empty sources should return empty string"
    
    # Test with sources only
    sources = [
        SourceItem(title="Test 1", url="https://example.com/1", snippet="Snippet 1"),
        SourceItem(title="Test 2", url="https://example.com/2", snippet="Snippet 2"),
    ]
    block = build_web_evidence_block(sources, [])
    assert "[SOURCES]" in block, "Should contain SOURCES section"
    assert "Test 1" in block, "Should contain first title"
    assert "https://example.com/1" in block, "Should contain first URL"
    
    # Test with evidence chunks
    evidence = [
        {"source_index": 1, "content": "Evidence from source 1"},
        {"source_index": 2, "content": "Evidence from source 2"},
    ]
    block = build_web_evidence_block(sources, evidence)
    assert "[EVIDENCE EXCERPTS]" in block, "Should contain EVIDENCE section"
    assert "[1a]" in block, "Should contain indexed evidence"
    assert "Evidence from source 1" in block, "Should contain evidence content"
    
    # Test character limit
    long_evidence = [{"source_index": 1, "content": "x" * 2000}]
    block = build_web_evidence_block(sources, long_evidence, max_chars=100)
    assert len(block) <= 103, "Should respect max_chars limit (with '...')"
    assert block.endswith("..."), "Should end with ellipsis when truncated"
    
    print("✅ Evidence block builder works correctly")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing internet_rag_service.py")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_tavily_key_rotation,
        test_smart_chunking,
        test_evidence_block_builder,
    ]
    
    results = []
    for test in tests:
        print(f"\n🧪 Running {test.__name__}...")
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
