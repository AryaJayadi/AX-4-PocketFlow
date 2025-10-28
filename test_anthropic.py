#!/usr/bin/env python3
"""
Test Anthropic Claude integration.
Verifies that the LLM provider works correctly with Anthropic's API.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_anthropic_config():
    """Test that Anthropic configuration is present."""
    print("=" * 60)
    print("TEST 1: Anthropic Configuration")
    print("=" * 60)

    provider = os.getenv("LLM_PROVIDER")
    model = os.getenv("ANTHROPIC_MODEL")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    tpm_limit = os.getenv("OPENAI_TPM_LIMIT")
    max_tokens = os.getenv("LLM_MAX_TOKENS")

    print(f"\nLLM_PROVIDER: {provider}")
    print(f"ANTHROPIC_MODEL: {model}")
    print(f"ANTHROPIC_API_KEY: {'*' * 20 + api_key[-10:] if api_key else 'NOT SET'}")
    print(f"OPENAI_TPM_LIMIT: {tpm_limit}")
    print(f"LLM_MAX_TOKENS: {max_tokens}")

    errors = []

    if provider != "ANTHROPIC":
        errors.append(f"LLM_PROVIDER should be 'ANTHROPIC', got '{provider}'")

    if not model:
        errors.append("ANTHROPIC_MODEL is not set")
    elif model not in [
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
    ]:
        print(
            f"Warning: Unusual model name '{model}' (might be correct, just checking)"
        )

    if not api_key:
        errors.append("ANTHROPIC_API_KEY is not set")
    elif not api_key.startswith("sk-ant-"):
        errors.append(
            f"ANTHROPIC_API_KEY should start with 'sk-ant-', got '{api_key[:10]}...'"
        )

    if not tpm_limit:
        errors.append("OPENAI_TPM_LIMIT is not set")
    elif int(tpm_limit) != 450000:
        print(f"Warning: OPENAI_TPM_LIMIT is {tpm_limit}, expected 450000 for Claude")

    if not max_tokens:
        errors.append("LLM_MAX_TOKENS is not set")
    elif int(max_tokens) < 1000:
        errors.append(
            f"LLM_MAX_TOKENS ({max_tokens}) seems too low (recommend 4096-8192)"
        )

    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✓ Configuration looks good!")
        return True


def test_anthropic_api():
    """Test actual API call to Anthropic."""
    print("\n" + "=" * 60)
    print("TEST 2: Anthropic API Call")
    print("=" * 60)

    try:
        from utils.call_llm import call_llm

        print("\nSending test prompt to Claude...")
        prompt = "In exactly 3 words, what is 2+2?"

        response = call_llm(prompt, use_cache=False)

        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")

        # Verify response is reasonable
        if response and len(response) > 0:
            print("\n✓ API call successful!")
            return True
        else:
            print("\n❌ API call returned empty response")
            return False

    except Exception as e:
        print(f"\n❌ API call failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_token_limits():
    """Test TPM configuration."""
    print("\n" + "=" * 60)
    print("TEST 3: TPM Configuration")
    print("=" * 60)

    tpm_limit = int(os.getenv("OPENAI_TPM_LIMIT", "0"))
    tpm_soft_pct = float(os.getenv("OPENAI_TPM_SOFT_PCT", "0.9"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))

    effective_tpm = int(tpm_limit * tpm_soft_pct)
    avg_request_tokens = max_tokens + 10000  # Assume 10k input average
    max_concurrent = effective_tpm // avg_request_tokens

    print(f"\nTPM Limit: {tpm_limit:,}")
    print(f"Soft Percentage: {tpm_soft_pct * 100:.0f}%")
    print(f"Effective TPM: {effective_tpm:,}")
    print(f"Max Tokens per Response: {max_tokens:,}")
    print(f"\nEstimated:")
    print(f"  - Average tokens per request: ~{avg_request_tokens:,}")
    print(f"  - Max concurrent requests: ~{max_concurrent}")

    if tpm_limit == 450000:
        print("\n✓ TPM limit correct for Anthropic Claude")
        return True
    else:
        print(f"\n⚠ Warning: TPM limit is {tpm_limit}, expected 400000 for Claude")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ANTHROPIC CLAUDE INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Configuration
    results.append(("Configuration", test_anthropic_config()))

    # Test 2: API Call (only if config is good)
    if results[0][1]:
        results.append(("API Call", test_anthropic_api()))
    else:
        print("\n⚠ Skipping API test due to configuration errors")
        results.append(("API Call", False))

    # Test 3: TPM Configuration
    results.append(("TPM Configuration", test_token_limits()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {test_name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n" + "=" * 60)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("=" * 60)
        print("\nYou're ready to use Anthropic Claude!")
        print("\nNext steps:")
        print("  1. Copy .env.anthropic to .env")
        print("  2. Update your API key")
        print("  3. Run: python main.py --repo URL")
        print()
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the errors above before proceeding.")
        print("\nCommon fixes:")
        print("  1. Copy .env.anthropic to .env")
        print("  2. Update ANTHROPIC_API_KEY with your actual key")
        print("  3. Verify ANTHROPIC_MODEL is correct")
        print("  4. Set OPENAI_TPM_LIMIT=400000")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
