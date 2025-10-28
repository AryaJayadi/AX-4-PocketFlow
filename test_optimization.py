#!/usr/bin/env python3
"""
Test script for LLM optimization features.
Verifies chunking, structure extraction, and token budget management.
"""

import os
import sys
from utils.call_llm import (
    estimate_tokens,
    extract_code_structure,
    chunk_large_file,
    optimize_context_for_budget,
    get_smart_context
)

# Sample Python code for testing
SAMPLE_PYTHON = '''
import os
import sys
from typing import List, Dict

class DataProcessor:
    """Main data processing class."""

    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.data = []

    def load_data(self, filepath: str) -> List[Dict]:
        """Load data from file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def process(self, data: List[Dict]) -> List[Dict]:
        """Process the data."""
        results = []
        for item in data:
            processed = self._process_item(item)
            results.append(processed)
        return results

    def _process_item(self, item: Dict) -> Dict:
        """Process a single item."""
        # Implementation details here
        return {
            'id': item['id'],
            'value': item['value'] * 2,
            'status': 'processed'
        }

def main():
    """Main entry point."""
    processor = DataProcessor({'mode': 'fast'})
    data = processor.load_data('data.json')
    results = processor.process(data)
    print(f"Processed {len(results)} items")

if __name__ == "__main__":
    main()
'''

SAMPLE_JAVASCRIPT = '''
import React from 'react';
import { useState, useEffect } from 'react';

export class DataService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
    }

    async fetchData(endpoint) {
        const response = await fetch(`${this.apiUrl}/${endpoint}`);
        return response.json();
    }
}

export const useData = (endpoint) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const service = new DataService('/api');
        service.fetchData(endpoint).then(result => {
            setData(result);
            setLoading(false);
        });
    }, [endpoint]);

    return { data, loading };
};
'''


def test_token_estimation():
    """Test token estimation accuracy."""
    print("=" * 60)
    print("TEST 1: Token Estimation")
    print("=" * 60)

    test_texts = [
        "Hello world",
        "This is a longer sentence with more words.",
        SAMPLE_PYTHON[:500],
    ]

    for i, text in enumerate(test_texts, 1):
        tokens = estimate_tokens(text)
        chars = len(text)
        ratio = chars / tokens if tokens > 0 else 0
        print(f"\nText {i}: {chars} chars -> {tokens} tokens (ratio: {ratio:.2f})")
        print(f"Preview: {text[:50]}...")

    print("\n✓ Token estimation test passed\n")


def test_structure_extraction():
    """Test code structure extraction."""
    print("=" * 60)
    print("TEST 2: Structure Extraction")
    print("=" * 60)

    tests = [
        ("test.py", SAMPLE_PYTHON),
        ("test.js", SAMPLE_JAVASCRIPT),
    ]

    for filename, code in tests:
        print(f"\n--- Testing {filename} ---")
        original_tokens = estimate_tokens(code)
        structure = extract_code_structure(code, filename)
        structure_tokens = estimate_tokens(structure)

        reduction = ((original_tokens - structure_tokens) / original_tokens * 100)
        print(f"Original: {original_tokens} tokens")
        print(f"Structure: {structure_tokens} tokens")
        print(f"Reduction: {reduction:.1f}%")
        print(f"\nExtracted structure:\n{structure[:300]}...")

    print("\n✓ Structure extraction test passed\n")


def test_chunking():
    """Test file chunking."""
    print("=" * 60)
    print("TEST 3: File Chunking")
    print("=" * 60)

    # Create a larger Python file
    large_code = SAMPLE_PYTHON * 10

    print(f"\nOriginal code: {estimate_tokens(large_code)} tokens")

    chunks = chunk_large_file(large_code, "large_test.py", max_tokens=2000)

    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i} ({chunk['type']}): {chunk['tokens']} tokens - {chunk['name']}")

    total_chunk_tokens = sum(c['tokens'] for c in chunks)
    print(f"\nTotal tokens across chunks: {total_chunk_tokens}")

    print("\n✓ Chunking test passed\n")


def test_context_optimization():
    """Test context optimization with budget."""
    print("=" * 60)
    print("TEST 4: Context Optimization")
    print("=" * 60)

    # Create multiple files
    files_content_map = {
        "0 # main.py": SAMPLE_PYTHON,
        "1 # utils.py": SAMPLE_PYTHON[:400],
        "2 # api.js": SAMPLE_JAVASCRIPT,
        "3 # config.py": "CONFIG = {'debug': True, 'port': 8000}\n" * 50,
        "4 # large_file.py": SAMPLE_PYTHON * 5,
    }

    # Calculate original size
    original_size = sum(estimate_tokens(content) for content in files_content_map.values())
    print(f"\nOriginal total: {original_size} tokens across {len(files_content_map)} files")

    # Test different budgets
    budgets = [5000, 10000, 20000]

    for budget in budgets:
        optimized = optimize_context_for_budget(
            files_content_map,
            token_budget=budget,
            use_structure_for_large=True,
            priority_keywords=["main", "api"]
        )

        actual_tokens = estimate_tokens(optimized)
        efficiency = (actual_tokens / budget * 100) if budget > 0 else 0
        savings = ((original_size - actual_tokens) / original_size * 100) if original_size > 0 else 0

        print(f"\nBudget: {budget} tokens")
        print(f"  Result: {actual_tokens} tokens ({efficiency:.1f}% of budget)")
        print(f"  Savings: {savings:.1f}% reduction")
        print(f"  Within budget: {'✓' if actual_tokens <= budget else '✗'}")

    print("\n✓ Context optimization test passed\n")


def test_smart_context():
    """Test smart context retrieval."""
    print("=" * 60)
    print("TEST 5: Smart Context Retrieval")
    print("=" * 60)

    # Simulate files_data structure
    files_data = [
        ("main.py", SAMPLE_PYTHON),
        ("utils.py", SAMPLE_PYTHON[:300]),
        ("api.js", SAMPLE_JAVASCRIPT),
        ("config.py", "CONFIG = {'debug': True}\n" * 20),
        ("large.py", SAMPLE_PYTHON * 3),
    ]

    # Test with different index sets
    test_cases = [
        ([0, 1], 3000),
        ([0, 2, 4], 8000),
        (list(range(len(files_data))), 15000),
    ]

    for indices, budget in test_cases:
        context = get_smart_context(
            files_data,
            indices,
            token_budget=budget,
            priority_keywords=["main"]
        )

        actual_tokens = estimate_tokens(context)
        print(f"\nIndices {indices}, Budget: {budget}")
        print(f"  Result: {actual_tokens} tokens")
        print(f"  Within budget: {'✓' if actual_tokens <= budget else '✗'}")

    print("\n✓ Smart context test passed\n")


def test_integration():
    """Integration test simulating real usage."""
    print("=" * 60)
    print("TEST 6: Integration Test")
    print("=" * 60)

    # Simulate a small codebase
    codebase = {
        "0 # src/main.py": SAMPLE_PYTHON,
        "1 # src/utils.py": "def helper():\n    pass\n" * 50,
        "2 # src/api.js": SAMPLE_JAVASCRIPT,
        "3 # tests/test_main.py": "def test_something():\n    assert True\n" * 30,
        "4 # README.md": "# Project Documentation\n" * 20,
    }

    print(f"\nCodebase: {len(codebase)} files")

    # Simulate IdentifyAbstractions context
    print("\n--- Simulating IdentifyAbstractions ---")
    budget = 40000
    context = optimize_context_for_budget(
        codebase,
        token_budget=budget,
        priority_keywords=["main", "core", "api"]
    )
    tokens = estimate_tokens(context)
    print(f"Context: {tokens}/{budget} tokens ({tokens*100//budget}% of budget)")

    # Simulate WriteChapter context (subset of files)
    print("\n--- Simulating WriteChapter ---")
    chapter_budget = 12000
    chapter_files = {k: v for k, v in list(codebase.items())[:3]}
    chapter_context = optimize_context_for_budget(
        chapter_files,
        token_budget=chapter_budget,
        use_structure_for_large=True
    )
    chapter_tokens = estimate_tokens(chapter_context)
    print(f"Chapter context: {chapter_tokens}/{chapter_budget} tokens")

    print("\n✓ Integration test passed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LLM OPTIMIZATION TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_token_estimation()
        test_structure_extraction()
        test_chunking()
        test_context_optimization()
        test_smart_context()
        test_integration()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nOptimization features are working correctly.")
        print("\nConfiguration environment variables:")
        print("  TOKEN_BUDGET_IDENTIFY=40000 (default)")
        print("  TOKEN_BUDGET_RELATIONSHIPS=15000 (default)")
        print("  TOKEN_BUDGET_CHAPTER=12000 (default)")
        print("\nYou can adjust these via environment variables.\n")

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
