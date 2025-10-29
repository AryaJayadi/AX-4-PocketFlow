from google import genai
import os
import logging
import json
import requests
from datetime import datetime
import time
import random
from collections import deque
import re
import ast
from typing import List, Dict, Tuple, Optional, Any

try:
    from dotenv import load_dotenv

    load_dotenv()  # loads variables from a .env file into os.environ
except Exception:
    pass

# ========= Token-per-minute gate (Solution #1) =========
# Env overrides:
#   OPENAI_TPM_LIMIT         -> integer TPM limit for your org/key (default 500000)
#   OPENAI_TPM_SOFT_PCT      -> soft cap fraction to leave headroom (default 0.9)
#   LLM_EXPECTED_OUTPUT_TOKENS -> expected output tokens per call (default 512)
TPM_LIMIT = int(os.getenv("OPENAI_TPM_LIMIT", "500000"))
TPM_SOFT_PCT = float(os.getenv("OPENAI_TPM_SOFT_PCT", "0.9"))
TPM_SOFT_CAP = int(TPM_LIMIT * TPM_SOFT_PCT)
EXPECTED_OUTPUT_TOKENS_DEFAULT = int(os.getenv("LLM_EXPECTED_OUTPUT_TOKENS", "512"))

# rolling 60s window of (timestamp, tokens)
_token_window = deque()
_WINDOW_SECONDS = 60


def _estimate_tokens_from_messages(
    messages, model="gpt-5-mini", expected_output=EXPECTED_OUTPUT_TOKENS_DEFAULT
):
    """
    Best-effort token estimator.
    Tries tiktoken if available, else falls back to chars/4 heuristic.
    """
    total_prompt_text = ""
    for m in messages:
        # messages are dicts: {"role": "...", "content": "..."} (or content list)
        c = m.get("content", "")
        if isinstance(c, list):
            # if user is using multi-part content, join text parts
            parts = []
            for item in c:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            c = "\n".join(parts)
        elif not isinstance(c, str):
            c = str(c)
        total_prompt_text += c

    # Try tiktoken for better accuracy
    try:
        import tiktoken

        # Model families: gpt-4/4o often map to cl100k_base / o200k_base.
        # For GPT-5 family, o200k_base is typically appropriate; fall back gracefully.
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                enc = tiktoken.get_encoding("o200k_base")
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(enc.encode(total_prompt_text))
    except Exception:
        # Heuristic: ~4 chars per token (English)
        prompt_tokens = max(1, len(total_prompt_text) // 4)

    # add expected output tokens to budget (upper bound to be safe)
    return prompt_tokens + int(expected_output)


def _admit_or_wait(
    messages,
    model="gpt-5-mini",
    expected_output=EXPECTED_OUTPUT_TOKENS_DEFAULT,
    logger=None,
):
    """
    Blocks until adding this request would keep us under TPM_SOFT_CAP within the rolling 60s window.
    - If the window is empty and we exceed the soft cap, we allow immediately (soft cap is a guideline).
    - If the estimated tokens exceed the HARD TPM limit, raise a clear error before sending.
    """
    want = _estimate_tokens_from_messages(
        messages, model=model, expected_output=expected_output
    )
    now = time.time()

    # Drop old entries
    while _token_window and now - _token_window[0][0] > _WINDOW_SECONDS:
        _token_window.popleft()

    used = sum(t for _, t in _token_window)
    if logger:
        logger.info(
            f"TPM gate: used={used}, want={want}, soft_cap={TPM_SOFT_CAP}, hard_cap={TPM_LIMIT}"
        )

    # Hard guard: one call larger than HARD limit will be rejected by the API
    if want > TPM_LIMIT:
        msg = (
            f"Estimated tokens for a single request ({want}) exceed your hard TPM limit ({TPM_LIMIT}). "
            "Reduce input size and/or set max_tokens, or split the request."
        )
        if logger:
            logger.error(msg)
        raise RuntimeError(msg)

    # While we'd exceed the SOFT cap, decide whether to wait
    while used + want > TPM_SOFT_CAP:
        # If the window is empty, waiting won't help; allow immediately (soft cap is just headroom)
        if not _token_window:
            if logger:
                logger.info(
                    "TPM gate: window empty and want exceeds soft cap; allowing request (soft cap bypass)."
                )
            break

        # Otherwise, sleep until the oldest entry slides out of the 60s window
        sleep_for = _WINDOW_SECONDS - (now - _token_window[0][0])
        sleep_for = max(0.05, sleep_for)
        if logger:
            logger.info(
                f"TPM gate sleeping {sleep_for:.2f}s (used={used}, want={want}, cap={TPM_SOFT_CAP})"
            )
        time.sleep(sleep_for)

        # Recompute after sleeping
        now = time.time()
        while _token_window and now - _token_window[0][0] > _WINDOW_SECONDS:
            _token_window.popleft()
        used = sum(t for _, t in _token_window)

    # Admit the request: record its estimated cost into the rolling window
    _token_window.append((time.time(), want))


# ========= 429 Backoff with Retry-After (Solution #2) =========
def _post_with_backoff(url, headers, json_payload, max_retries=6, logger=None):
    """
    POST with handling for 429 rate limits:
    - Respect Retry-After header if present
    - Exponential backoff + jitter
    """
    wait = 1.0
    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=json_payload)
        if resp.status_code != 429:
            return resp

        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                wait = max(wait, float(ra))
            except Exception:
                pass

        if logger:
            try:
                err_json = resp.json()
            except Exception:
                err_json = {}
            logger.warning(
                f"429 received. attempt={attempt + 1}/{max_retries}, retrying in ~{wait:.2f}s; error={err_json}"
            )

        time.sleep(wait + random.uniform(0, 0.250))  # jitter
        wait = min(wait * 2, 20.0)  # cap wait

    return resp  # return the last response (likely 429) so caller can raise


# ========= Context Optimization & Chunking (Solution #3) =========
# Intelligent code chunking and context reduction to minimize token usage
# while maintaining high-quality results


def estimate_tokens(text: str, model: str = "gpt-5-mini") -> int:
    """
    Accurately estimate tokens for a given text.
    Returns token count as integer.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                enc = tiktoken.get_encoding("o200k_base")
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~4 chars per token
        return max(1, len(text) // 4)


def extract_code_structure(code: str, file_path: str) -> str:
    """
    Extract the structural outline of code (function/class signatures, docstrings)
    without full implementations. Reduces token usage by 60-80% while preserving context.

    Args:
        code: Source code content
        file_path: File path for language detection

    Returns:
        Structured outline of the code
    """
    # Detect language from file extension
    ext = file_path.split(".")[-1].lower()

    if ext in ["py", "pyx", "pyi"]:
        return _extract_python_structure(code)
    elif ext in ["js", "jsx", "ts", "tsx"]:
        return _extract_javascript_structure(code)
    elif ext in ["java"]:
        return _extract_java_structure(code)
    elif ext in ["go"]:
        return _extract_go_structure(code)
    elif ext in ["c", "cc", "cpp", "h", "hpp"]:
        return _extract_c_structure(code)
    else:
        # For unknown types, return first 50 lines
        lines = code.split("\n")
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more lines)"
        return code


def _extract_python_structure(code: str) -> str:
    """Extract Python structure: imports, class/function signatures, docstrings."""
    try:
        tree = ast.parse(code)
        lines = code.split("\n")
        structure = []

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, "lineno"):
                    structure.append(lines[node.lineno - 1])

        if structure:
            structure.append("")  # Blank line after imports

        # Extract class and function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Class definition with docstring
                class_def = f"class {node.name}"
                if node.bases:
                    bases = ", ".join([ast.unparse(base) for base in node.bases])
                    class_def += f"({bases})"
                class_def += ":"
                structure.append(class_def)

                # Add docstring if exists
                docstring = ast.get_docstring(node)
                if docstring:
                    structure.append(f'    """{docstring}"""')

                # Add method signatures
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        args = [arg.arg for arg in item.args.args]
                        method_sig = f"    def {item.name}({', '.join(args)})"
                        # Add return annotation if exists
                        if item.returns:
                            method_sig += f" -> {ast.unparse(item.returns)}"
                        method_sig += ": ..."
                        structure.append(method_sig)

                        # Add method docstring
                        method_doc = ast.get_docstring(item)
                        if method_doc:
                            # Use first line of docstring only
                            first_line = method_doc.split("\n")[0]
                            structure.append(f'        """{first_line}"""')

                structure.append("")

            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                # Top-level function
                args = [arg.arg for arg in node.args.args]
                func_sig = f"def {node.name}({', '.join(args)})"
                if node.returns:
                    func_sig += f" -> {ast.unparse(node.returns)}"
                func_sig += ":"
                structure.append(func_sig)

                # Add docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    first_line = docstring.split("\n")[0]
                    structure.append(f'    """{first_line}"""')
                structure.append("    ...")
                structure.append("")

        result = "\n".join(structure)
        if not result.strip():
            # Fallback: return first 30 lines
            return "\n".join(code.split("\n")[:30])
        return result

    except Exception as e:
        # If parsing fails, return first 30 lines
        # Note: logger may not be initialized yet at import time
        return "\n".join(code.split("\n")[:30])


def _extract_javascript_structure(code: str) -> str:
    """Extract JavaScript/TypeScript structure using regex patterns."""
    structure = []
    lines = code.split("\n")

    # Extract imports
    for line in lines:
        if re.match(r"^\s*(import|export|require)", line.strip()):
            structure.append(line.rstrip())

    if structure:
        structure.append("")

    # Extract class definitions
    class_pattern = re.compile(
        r"^\s*(export\s+)?(class|interface|type)\s+(\w+)", re.MULTILINE
    )
    for match in class_pattern.finditer(code):
        structure.append(match.group(0))

    # Extract function definitions
    func_patterns = [
        r"^\s*(export\s+)?(async\s+)?function\s+(\w+)\s*\([^)]*\)",
        r"^\s*(const|let|var)\s+(\w+)\s*=\s*(async\s*)?\([^)]*\)\s*=>",
        r"^\s*(\w+)\s*\([^)]*\)\s*\{",  # Method definitions
    ]

    for pattern in func_patterns:
        for match in re.finditer(pattern, code, re.MULTILINE):
            line = match.group(0).rstrip()
            if not line.strip().startswith("//"):
                structure.append(line)

    result = "\n".join(structure[:50])  # Limit to 50 items
    return result if result.strip() else "\n".join(lines[:30])


def _extract_java_structure(code: str) -> str:
    """Extract Java structure."""
    structure = []
    lines = code.split("\n")

    # Extract package and imports
    for line in lines:
        if re.match(r"^\s*(package|import)\s+", line.strip()):
            structure.append(line.rstrip())

    if structure:
        structure.append("")

    # Extract class/interface/enum definitions and method signatures
    patterns = [
        r"^\s*(public|private|protected)?\s*(class|interface|enum)\s+\w+",
        r"^\s*(public|private|protected)?\s*(static\s+)?\w+\s+\w+\s*\([^)]*\)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, code, re.MULTILINE):
            structure.append(match.group(0).rstrip() + " { ... }")

    result = "\n".join(structure[:50])
    return result if result.strip() else "\n".join(lines[:30])


def _extract_go_structure(code: str) -> str:
    """Extract Go structure."""
    structure = []
    lines = code.split("\n")

    # Extract package and imports
    for line in lines:
        if re.match(r"^\s*(package|import)\s+", line.strip()):
            structure.append(line.rstrip())

    if structure:
        structure.append("")

    # Extract type definitions and function signatures
    patterns = [
        r"^\s*type\s+\w+\s+(struct|interface)",
        r"^\s*func\s+(\(\w+\s+\*?\w+\)\s+)?\w+\s*\([^)]*\)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, code, re.MULTILINE):
            structure.append(match.group(0).rstrip() + " { ... }")

    result = "\n".join(structure[:50])
    return result if result.strip() else "\n".join(lines[:30])


def _extract_c_structure(code: str) -> str:
    """Extract C/C++ structure."""
    structure = []
    lines = code.split("\n")

    # Extract includes and defines
    for line in lines:
        if re.match(r"^\s*#\s*(include|define)", line.strip()):
            structure.append(line.rstrip())

    if structure:
        structure.append("")

    # Extract function signatures, structs, classes
    patterns = [
        r"^\s*(struct|class|enum)\s+\w+",
        r"^\s*\w+[\s\*]+\w+\s*\([^)]*\)\s*[;{]",  # Function signatures
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, code, re.MULTILINE):
            line = match.group(0).rstrip()
            if "{" in line:
                line = line.replace("{", "{ ... }")
            structure.append(line)

    result = "\n".join(structure[:50])
    return result if result.strip() else "\n".join(lines[:30])


def chunk_large_file(
    code: str, file_path: str, max_tokens: int = 2000
) -> List[Dict[str, Any]]:
    """
    Split large files into semantic chunks based on code structure.

    Args:
        code: Source code content
        file_path: File path for language detection
        max_tokens: Maximum tokens per chunk

    Returns:
        List of chunks with metadata: [{"content": str, "type": str, "name": str, "tokens": int}]
    """
    ext = file_path.split(".")[-1].lower()

    if ext in ["py", "pyx", "pyi"]:
        return _chunk_python_file(code, max_tokens)
    else:
        # For other languages, use line-based chunking
        return _chunk_by_lines(code, max_tokens)


def _chunk_python_file(code: str, max_tokens: int) -> List[Dict[str, Any]]:
    """Chunk Python file by classes and functions."""
    chunks = []

    try:
        tree = ast.parse(code)
        lines = code.split("\n")

        # Get module-level imports and docstring
        imports = []
        module_docstring = ast.get_docstring(tree)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, "lineno"):
                    imports.append(lines[node.lineno - 1])

        imports_text = "\n".join(imports)

        # Process each top-level node
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                start_line = node.lineno - 1
                end_line = (
                    node.end_lineno if hasattr(node, "end_lineno") else start_line + 50
                )

                # Get the code for this node
                node_code = "\n".join(lines[start_line:end_line])
                node_tokens = estimate_tokens(node_code)

                chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"

                # If too large, get structure instead
                if node_tokens > max_tokens:
                    if isinstance(node, ast.ClassDef):
                        node_code = _extract_class_structure(node, lines)
                    else:
                        node_code = _extract_function_structure(node, lines)
                    node_tokens = estimate_tokens(node_code)

                chunks.append(
                    {
                        "content": f"{imports_text}\n\n{node_code}"
                        if imports_text
                        else node_code,
                        "type": chunk_type,
                        "name": node.name,
                        "tokens": node_tokens,
                        "priority": 1,  # Default priority
                    }
                )

        return (
            chunks
            if chunks
            else [
                {
                    "content": code[:5000],
                    "type": "module",
                    "name": "main",
                    "tokens": estimate_tokens(code[:5000]),
                    "priority": 1,
                }
            ]
        )

    except Exception as e:
        # Fallback to line-based chunking
        return _chunk_by_lines(code, max_tokens)


def _extract_class_structure(node: ast.ClassDef, lines: List[str]) -> str:
    """Extract class structure with method signatures."""
    start = node.lineno - 1
    structure = [lines[start]]  # Class definition

    docstring = ast.get_docstring(node)
    if docstring:
        structure.append(f'    """{docstring.split(chr(10))[0]}"""')

    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            args = [arg.arg for arg in item.args.args]
            method_sig = f"    def {item.name}({', '.join(args)}): ..."
            structure.append(method_sig)

    return "\n".join(structure)


def _extract_function_structure(node: ast.FunctionDef, lines: List[str]) -> str:
    """Extract function signature with docstring."""
    start = node.lineno - 1
    structure = [lines[start]]  # Function definition

    docstring = ast.get_docstring(node)
    if docstring:
        structure.append(f'    """{docstring.split(chr(10))[0]}"""')
    structure.append("    ...")

    return "\n".join(structure)


def _chunk_by_lines(code: str, max_tokens: int) -> List[Dict[str, Any]]:
    """Fallback: chunk by lines for non-Python files."""
    lines = code.split("\n")
    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line)
        if current_tokens + line_tokens > max_tokens and current_chunk:
            chunks.append(
                {
                    "content": "\n".join(current_chunk),
                    "type": "section",
                    "name": f"lines_{len(chunks) + 1}",
                    "tokens": current_tokens,
                    "priority": 1,
                }
            )
            current_chunk = [line]
            current_tokens = line_tokens
        else:
            current_chunk.append(line)
            current_tokens += line_tokens

    if current_chunk:
        chunks.append(
            {
                "content": "\n".join(current_chunk),
                "type": "section",
                "name": f"lines_{len(chunks) + 1}",
                "tokens": current_tokens,
                "priority": 1,
            }
        )

    return chunks


def optimize_context_for_budget(
    files_content_map: Dict[str, str],
    token_budget: int,
    use_structure_for_large: bool = True,
    priority_keywords: Optional[List[str]] = None,
) -> str:
    """
    Optimize file context to fit within token budget while preserving important information.

    Args:
        files_content_map: Dictionary mapping file paths to content
        token_budget: Maximum tokens allowed
        use_structure_for_large: Extract structure for files exceeding 30% of budget
        priority_keywords: Keywords to prioritize files (e.g., ["main", "core", "api"])

    Returns:
        Optimized context string fitting within budget
    """
    if not files_content_map:
        return ""

    # Calculate tokens for each file
    file_tokens = []
    for file_path, content in files_content_map.items():
        tokens = estimate_tokens(content)
        priority = 1

        # Boost priority for files with keywords
        if priority_keywords:
            for keyword in priority_keywords:
                if keyword.lower() in file_path.lower():
                    priority = 0  # Higher priority (lower number)
                    break

        file_tokens.append(
            {
                "path": file_path,
                "content": content,
                "tokens": tokens,
                "priority": priority,
            }
        )

    # Sort by priority, then by token count (smaller first)
    file_tokens.sort(key=lambda x: (x["priority"], x["tokens"]))

    # Build optimized context
    context_parts = []
    used_tokens = 0
    per_file_budget = token_budget * 0.3  # 30% of budget per file max

    for file_info in file_tokens:
        file_path = file_info["path"]
        content = file_info["content"]
        tokens = file_info["tokens"]

        # Check if we can fit the full file
        if used_tokens + tokens <= token_budget:
            context_parts.append(f"--- File: {file_path} ---\n{content}")
            used_tokens += tokens
        elif use_structure_for_large and tokens > per_file_budget:
            # Extract structure for large files
            structure = extract_code_structure(content, file_path)
            structure_tokens = estimate_tokens(structure)

            if used_tokens + structure_tokens <= token_budget:
                context_parts.append(
                    f"--- File: {file_path} (structure only) ---\n{structure}"
                )
                used_tokens += structure_tokens
            else:
                # Skip this file - exceeds budget
                pass
        else:
            # Try to fit a truncated version
            remaining_budget = token_budget - used_tokens
            if remaining_budget > 100:  # Only if we have meaningful space
                # Estimate how many lines we can fit
                lines = content.split("\n")
                approx_lines = int(remaining_budget * 4 / (len(content) / len(lines)))
                truncated = "\n".join(lines[:approx_lines])

                context_parts.append(
                    f"--- File: {file_path} (truncated) ---\n{truncated}\n... (truncated)"
                )
                used_tokens = token_budget  # Budget exhausted
                break

    result = "\n\n".join(context_parts)
    # Log optimization stats if logger is available
    try:
        logger.info(
            f"Context optimization: {len(file_tokens)} files -> {len(context_parts)} included, {used_tokens}/{token_budget} tokens used"
        )
    except NameError:
        pass  # Logger not yet initialized
    return result


def get_smart_context(
    files_data: List[Tuple[str, str]],
    indices: List[int],
    token_budget: int = 8000,
    priority_keywords: Optional[List[str]] = None,
) -> str:
    """
    Get optimized context for specific file indices with smart chunking.

    Args:
        files_data: List of (path, content) tuples
        indices: File indices to include
        token_budget: Maximum tokens for context
        priority_keywords: Keywords for file prioritization

    Returns:
        Optimized context string
    """
    # Build content map for specified indices
    files_content_map = {}
    for i in indices:
        if 0 <= i < len(files_data):
            path, content = files_data[i]
            files_content_map[f"{i} # {path}"] = content

    # Optimize and return
    return optimize_context_for_budget(
        files_content_map,
        token_budget=token_budget,
        use_structure_for_large=True,
        priority_keywords=priority_keywords,
    )


# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"


def load_cache():
    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except:
        logger.warning(f"Failed to load cache.")
    return {}


def save_cache(cache):
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f)
    except:
        logger.warning(f"Failed to save cache")


def get_llm_provider():
    provider = os.getenv("LLM_PROVIDER")
    if not provider and (os.getenv("GEMINI_PROJECT_ID") or os.getenv("GEMINI_API_KEY")):
        provider = "GEMINI"
    # if necessary, add ANTHROPIC/OPENAI
    return provider


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    """
    Call Anthropic's Claude API using their native format.
    """
    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    # Build messages for TPM admission (using OpenAI format for token estimation)
    messages_for_tpm = [{"role": "user", "content": prompt}]

    # Get max tokens from environment or use default
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))

    # ---- Solution #1: Token gate before sending ----
    _admit_or_wait(
        messages_for_tpm,
        model=model,
        expected_output=max_tokens,
        logger=logger,
    )

    # Anthropic's native API format
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        # ---- Solution #2: Backoff-aware POST ----
        response = _post_with_backoff(
            url, headers, payload, max_retries=6, logger=logger
        )

        # Log and raise if needed
        response_json = {}
        try:
            response_json = response.json()
            logger.info("RESPONSE:\n%s", json.dumps(response_json, indent=2))
        except Exception:
            logger.warning("Failed to parse JSON for logging")

        response.raise_for_status()

        # Anthropic returns content in a different format
        return response_json["content"][0]["text"]

    except requests.exceptions.HTTPError as e:
        error_message = f"HTTP error occurred: {e}"
        try:
            error_details = response.json().get("error", "No additional details")
            error_message += f" (Details: {error_details})"
        except Exception:
            pass
        raise Exception(error_message)
    except requests.exceptions.ConnectionError:
        raise Exception(
            f"Failed to connect to Anthropic API. Check your network connection."
        )
    except requests.exceptions.Timeout:
        raise Exception(f"Request to Anthropic API timed out.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"An error occurred while making the request to Anthropic: {e}")
    except (ValueError, KeyError) as e:
        raise Exception(f"Failed to parse response from Anthropic. Error: {e}")


def _call_llm_provider(prompt: str) -> str:
    """
    Call an LLM provider based on environment variables.
    Environment variables:
    - LLM_PROVIDER: "ANTHROPIC", "OPEN_AI", "OLLAMA", "XAI", etc.
    - <provider>_MODEL: Model name (e.g., ANTHROPIC_MODEL, OPEN_AI_MODEL)
    - <provider>_BASE_URL: Base URL without endpoint (not needed for ANTHROPIC)
    - <provider>_API_KEY: API key
    The endpoint /v1/chat/completions will be appended to the base URL (except for Anthropic).
    """
    logger.info(f"PROMPT: {prompt}")  # log the prompt

    try:
        from dotenv import load_dotenv

        load_dotenv()  # loads variables from a .env file into os.environ
    except Exception:
        pass

    provider = os.environ.get("LLM_PROVIDER")
    if not provider:
        raise ValueError("LLM_PROVIDER environment variable is required")

    # Special handling for Anthropic
    if provider == "ANTHROPIC":
        model = os.environ.get("ANTHROPIC_MODEL")
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not model:
            raise ValueError("ANTHROPIC_MODEL environment variable is required")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        return _call_anthropic(prompt, model, api_key)

    # OpenAI-compatible providers (OPEN_AI, OLLAMA, XAI, etc.)
    model_var = f"{provider}_MODEL"
    base_url_var = f"{provider}_BASE_URL"
    api_key_var = f"{provider}_API_KEY"

    model = os.environ.get(model_var)
    base_url = os.environ.get(base_url_var)
    api_key = os.environ.get(api_key_var, "")

    if not model:
        raise ValueError(f"{model_var} environment variable is required")
    if not base_url:
        raise ValueError(f"{base_url_var} environment variable is required")

    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build messages first (for TPM admission)
    messages = [{"role": "user", "content": prompt}]

    # ---- Solution #1: Token gate before sending ----
    # If you know your typical max output, set via env LLM_EXPECTED_OUTPUT_TOKENS
    _admit_or_wait(
        messages,
        model=model,
        expected_output=EXPECTED_OUTPUT_TOKENS_DEFAULT,
        logger=logger,
    )

    # Prepare payload; omit temperature for GPT-5* (or set to 1) to avoid unsupported_value
    payload = {
        "model": model,
        "messages": messages,
        # "max_tokens": 512,  # optional but recommended to prevent runaway outputs
    }
    if not model.startswith("gpt-5"):
        payload["temperature"] = 0.7  # safe for most non-5 models; remove if you prefer

    # Qwen-specific handling: Qwen3 models support enable_thinking parameter
    # For commercial models like qwen3-max, enable_thinking defaults to False
    # For open source models, it defaults to True, which may cause errors without streaming
    if provider == "QWEN" and model.startswith("qwen"):
        # Check if user wants to explicitly set enable_thinking via environment
        enable_thinking = os.getenv("QWEN_ENABLE_THINKING", "").lower()
        if enable_thinking == "false":
            payload["enable_thinking"] = False
        elif enable_thinking == "true":
            payload["enable_thinking"] = True
        # If not set, let the API use its default (False for commercial, True for open source)

    try:
        # ---- Solution #2: Backoff-aware POST ----
        response = _post_with_backoff(
            url, headers, payload, max_retries=6, logger=logger
        )

        # Log and raise if needed
        response_json = {}
        try:
            response_json = response.json()
            logger.info("RESPONSE:\n%s", json.dumps(response_json, indent=2))
        except Exception:
            logger.warning("Failed to parse JSON for logging")

        response.raise_for_status()
        return response_json["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as e:
        error_message = f"HTTP error occurred: {e}"
        try:
            error_details = response.json().get("error", "No additional details")
            error_message += f" (Details: {error_details})"
        except Exception:
            pass
        raise Exception(error_message)
    except requests.exceptions.ConnectionError:
        raise Exception(
            f"Failed to connect to {provider} API. Check your network connection."
        )
    except requests.exceptions.Timeout:
        raise Exception(f"Request to {provider} API timed out.")
    except requests.exceptions.RequestException as e:
        raise Exception(
            f"An error occurred while making the request to {provider}: {e}"
        )
    except ValueError:
        raise Exception(
            f"Failed to parse response as JSON from {provider}. The server might have returned an invalid response."
        )


# By default, we Google Gemini 2.5 pro, as it shows great performance for code understanding
def call_llm(prompt: str, use_cache: bool = True) -> str:
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")

    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = load_cache()
        # Return from cache if exists
        if prompt in cache:
            logger.info(f"RESPONSE: {cache[prompt]}")
            return cache[prompt]

    provider = get_llm_provider()
    if provider == "GEMINI":
        response_text = _call_llm_gemini(prompt)
    else:  # generic method using a URL that is OpenAI compatible API (Ollama, ...)
        response_text = _call_llm_provider(prompt)

    # Log the response
    logger.info(f"RESPONSE: {response_text}")

    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites
        cache = load_cache()
        # Add to cache and save
        cache[prompt] = response_text
        save_cache(cache)

    return response_text


def _call_llm_gemini(prompt: str) -> str:
    if os.getenv("GEMINI_PROJECT_ID"):
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GEMINI_PROJECT_ID"),
            location=os.getenv("GEMINI_LOCATION", "us-central1"),
        )
    elif os.getenv("GEMINI_API_KEY"):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        raise ValueError(
            "Either GEMINI_PROJECT_ID or GEMINI_API_KEY must be set in the environment"
        )
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
    response = client.models.generate_content(model=model, contents=[prompt])
    return response.text


if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
