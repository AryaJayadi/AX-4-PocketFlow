# Qwen Integration Setup Guide

This guide explains how to configure and use Alibaba Cloud's Qwen models with PocketFlow.

## Prerequisites

1. An Alibaba Cloud account
2. A DashScope API key (get it from [Model Studio](https://www.alibabacloud.com/help/en/model-studio/get-api-key))

## Configuration Steps

### 1. Get Your API Key

Visit the [Model Studio API Key page](https://www.alibabacloud.com/help/en/model-studio/get-api-key) and create an API key.

**Note:** API keys differ between regions:
- **Singapore region**: Use `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- **China (Beijing) region**: Use `https://dashscope.aliyuncs.com/compatible-mode/v1`

### 2. Update .env File

Edit your `.env` file and uncomment the Qwen configuration section:

```bash
# Qwen Model Selection
LLM_PROVIDER="QWEN"
QWEN_MODEL="qwen3-max"
QWEN_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # Singapore region
# QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"  # China (Beijing) region - uncomment for Beijing
QWEN_API_KEY="your-actual-api-key-here"
QWEN_ENABLE_THINKING="false"  # Optional: Control thinking process
OPENAI_TPM_LIMIT=200000
OPENAI_TPM_SOFT_PCT=0.9
LLM_MAX_TOKENS=16384
LLM_EXPECTED_OUTPUT_TOKENS=16384
TOKEN_BUDGET_IDENTIFY=150000
TOKEN_BUDGET_RELATIONSHIPS=100000
TOKEN_BUDGET_CHAPTER=100000
```

**Important:** Make sure to comment out other provider configurations (OpenAI, Anthropic, etc.).

### 3. Available Models

Qwen offers several models:

| Model | Description | Use Case |
|-------|-------------|----------|
| `qwen3-max` | Latest flagship model | Best performance, complex tasks |
| `qwen-plus` | Balanced model | Good performance/cost ratio |
| `qwen-turbo` | Fast model | Quick responses, simple tasks |
| `qwen-max` | Previous generation flagship | Legacy support |
| `qwen2.5-72b-instruct` | Open source model | Self-hosted deployments |

**Recommended:** Use `qwen3-max` for best results.

### 4. Test Your Configuration

Run the test script to verify your setup:

```bash
python test_qwen.py
```

This will check:
- ✓ Configuration variables
- ✓ API connectivity
- ✓ Token limit settings

### 5. Advanced Configuration

#### Enable Thinking Mode

The `QWEN_ENABLE_THINKING` parameter controls the model's thinking process:

- **Commercial models** (qwen3-max, qwen-plus, qwen-turbo): Default is `false`
- **Open source models**: Default is `true`

```bash
# Explicitly enable/disable thinking mode
QWEN_ENABLE_THINKING="false"  # Disable (recommended for commercial models)
QWEN_ENABLE_THINKING="true"   # Enable (may require streaming for open source models)
```

#### Adjust Token Limits

Qwen's token limits vary by model and your subscription tier. Update these values based on your actual limits:

```bash
OPENAI_TPM_LIMIT=200000        # Tokens per minute limit
OPENAI_TPM_SOFT_PCT=0.9        # Use 90% of limit (safety buffer)
LLM_MAX_TOKENS=16384           # Max tokens per response
```

## Troubleshooting

### Error: "Failed to connect to QWEN API"

- Check your internet connection
- Verify `QWEN_BASE_URL` is correct for your region
- Ensure no firewall is blocking the connection

### Error: "HTTP error occurred: 401"

- Your API key is invalid or expired
- Get a new key from [Model Studio](https://www.alibabacloud.com/help/en/model-studio/get-api-key)
- Make sure you're using the right key for your region

### Error: "HTTP error occurred: 429"

- You've exceeded your rate limit
- Reduce `OPENAI_TPM_LIMIT` in .env
- Wait a few minutes before retrying

### Error: "enable_thinking parameter error"

- Remove or set `QWEN_ENABLE_THINKING="false"` for commercial models
- For open source models, ensure you're using streaming or disable thinking

## API Reference

Qwen uses an OpenAI-compatible API, making it easy to integrate. The implementation automatically handles:

- ✓ Token-per-minute rate limiting
- ✓ Exponential backoff on errors
- ✓ Automatic retry with jitter
- ✓ Request caching

## Cost Considerations

Qwen pricing varies by model and region. Check the [official pricing page](https://www.alibabacloud.com/help/en/model-studio/pricing) for current rates.

**Tips to reduce costs:**
- Use `qwen-plus` or `qwen-turbo` for simpler tasks
- Enable caching with `use_cache=True` in `call_llm()`
- Reduce `LLM_MAX_TOKENS` if you don't need long responses

## Examples

### Basic Usage

```python
from utils.call_llm import call_llm

# Simple call
response = call_llm("What is 2+2?")
print(response)

# With caching enabled (default)
response = call_llm("Explain Python decorators", use_cache=True)
```

### Switch Between Models

```bash
# Use qwen3-max for complex tasks
QWEN_MODEL="qwen3-max"

# Use qwen-turbo for simple tasks
QWEN_MODEL="qwen-turbo"
```

## Support

For issues or questions:

1. Check the [Qwen documentation](https://www.alibabacloud.com/help/en/model-studio)
2. Review the test logs in `logs/llm_calls_YYYYMMDD.log`
3. Run `python test_qwen.py` to diagnose problems

## Migration from Other Providers

If you're migrating from OpenAI or Anthropic:

1. Comment out your current provider configuration
2. Uncomment and configure the Qwen section
3. Update `LLM_PROVIDER="QWEN"`
4. Run `python test_qwen.py` to verify
5. Your existing code will work without changes!

The system is designed to be provider-agnostic, so switching between providers is as simple as updating the .env file.
