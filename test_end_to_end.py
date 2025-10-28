#!/usr/bin/env python3
"""
Quick end-to-end test with a minimal code sample.
"""

import os
import tempfile
import shutil
import sys

# Create a minimal test codebase
def create_test_codebase():
    """Create a small test codebase in a temp directory."""
    tmpdir = tempfile.mkdtemp(prefix="test_codebase_")

    # Create main.py
    with open(os.path.join(tmpdir, "main.py"), "w") as f:
        f.write("""
import os
from utils import helper

class Application:
    '''Main application class.'''

    def __init__(self, config):
        self.config = config

    def run(self):
        '''Run the application.'''
        result = helper.process_data(self.config)
        return result

def main():
    app = Application({'mode': 'test'})
    app.run()

if __name__ == "__main__":
    main()
""")

    # Create utils.py
    with open(os.path.join(tmpdir, "utils.py"), "w") as f:
        f.write("""
class DataProcessor:
    '''Process data.'''

    def process(self, data):
        return data * 2

def helper_function(value):
    '''Helper function.'''
    return value + 1

def process_data(config):
    '''Process data based on config.'''
    processor = DataProcessor()
    return processor.process(config.get('value', 0))
""")

    # Create README.md
    with open(os.path.join(tmpdir, "README.md"), "w") as f:
        f.write("""# Test Project

A minimal test project for testing the tutorial generator.
""")

    return tmpdir


def test_optimization_with_real_code():
    """Test that optimization works with actual code."""
    print("Creating test codebase...")
    tmpdir = create_test_codebase()

    try:
        print(f"Test codebase created at: {tmpdir}")
        print("\nFiles created:")
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                filepath = os.path.join(root, file)
                print(f"  - {os.path.relpath(filepath, tmpdir)}")

        # Test structure extraction
        print("\n" + "="*60)
        print("Testing structure extraction...")
        print("="*60)

        from utils.call_llm import extract_code_structure, estimate_tokens

        with open(os.path.join(tmpdir, "main.py"), "r") as f:
            code = f.read()

        original_tokens = estimate_tokens(code)
        structure = extract_code_structure(code, "main.py")
        structure_tokens = estimate_tokens(structure)

        print(f"Original: {original_tokens} tokens")
        print(f"Structure: {structure_tokens} tokens")
        print(f"Reduction: {(original_tokens - structure_tokens) / original_tokens * 100:.1f}%")
        print(f"\nExtracted structure:\n{structure}")

        # Test context optimization
        print("\n" + "="*60)
        print("Testing context optimization...")
        print("="*60)

        from utils.call_llm import optimize_context_for_budget

        files_map = {}
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                filepath = os.path.join(root, file)
                with open(filepath, "r") as f:
                    content = f.read()
                relpath = os.path.relpath(filepath, tmpdir)
                files_map[f"0 # {relpath}"] = content

        optimized = optimize_context_for_budget(
            files_map,
            token_budget=5000,
            priority_keywords=["main"]
        )

        optimized_tokens = estimate_tokens(optimized)
        print(f"Optimized context: {optimized_tokens} tokens")
        print(f"Preview:\n{optimized[:500]}...")

        print("\n✓ Optimization test passed!")
        print(f"\nYou can now test with the actual system:")
        print(f"  python main.py --dir {tmpdir} --max-abstractions 3")

        return tmpdir

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None


if __name__ == "__main__":
    tmpdir = test_optimization_with_real_code()
    if tmpdir:
        print(f"\n📁 Test codebase preserved at: {tmpdir}")
        print("   Run the above command to generate a tutorial")
        print(f"   Cleanup: rm -rf {tmpdir}")
