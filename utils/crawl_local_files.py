import os
import fnmatch
import pathspec
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Set, Optional


def crawl_local_files(
    directory: str,
    include_patterns: Optional[Set[str]] = None,
    exclude_patterns: Optional[Set[str]] = None,
    max_file_size: Optional[int] = None,
    use_relative_paths: bool = True,
    max_workers: int = 8,  # OPTIMIZED: Parallel I/O
) -> Dict[str, Dict[str, str]]:
    """
    OPTIMIZED: Crawl files in parallel with progress tracking.

    Args:
        directory: Path to local directory
        include_patterns: File patterns to include (e.g. {"*.py", "*.js"})
        exclude_patterns: File patterns to exclude (e.g. {"tests/*"})
        max_file_size: Maximum file size in bytes
        use_relative_paths: Whether to use paths relative to directory
        max_workers: Number of parallel file readers (default: 8)

    Returns:
        dict: {"files": {filepath: content}}
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    # Load .gitignore
    gitignore_path = os.path.join(directory, ".gitignore")
    gitignore_spec = None
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, "r", encoding="utf-8-sig") as f:
                gitignore_patterns = f.readlines()
            gitignore_spec = pathspec.PathSpec.from_lines(
                "gitwildmatch", gitignore_patterns
            )
            print(f"✓ Loaded .gitignore from {gitignore_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not read .gitignore: {e}")

    # OPTIMIZED: Collect all candidate files first
    candidate_files = []
    for root, dirs, files in os.walk(directory):
        # Filter directories early
        excluded_dirs = set()
        for d in dirs:
            dirpath_rel = os.path.relpath(os.path.join(root, d), directory)

            if gitignore_spec and gitignore_spec.match_file(dirpath_rel):
                excluded_dirs.add(d)
                continue

            if exclude_patterns:
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(dirpath_rel, pattern) or fnmatch.fnmatch(
                        d, pattern
                    ):
                        excluded_dirs.add(d)
                        break

        for d in excluded_dirs:
            dirs.remove(d)

        for filename in files:
            filepath = os.path.join(root, filename)
            candidate_files.append(filepath)

    total_files = len(candidate_files)
    print(f"Found {total_files} candidate files")

    # OPTIMIZED: Filter and read files in parallel
    files_dict = {}
    processed_count = 0
    included_count = 0

    def process_file(filepath: str) -> tuple:
        """Process a single file and return (relpath, content, status)."""
        relpath = (
            os.path.relpath(filepath, directory) if use_relative_paths else filepath
        )

        # Check exclusions
        if gitignore_spec and gitignore_spec.match_file(relpath):
            return (relpath, None, "gitignore")

        if exclude_patterns:
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(relpath, pattern):
                    return (relpath, None, "excluded")

        # Check inclusions
        included = False
        if include_patterns:
            for pattern in include_patterns:
                if fnmatch.fnmatch(relpath, pattern):
                    included = True
                    break
        else:
            included = True

        if not included:
            return (relpath, None, "not_included")

        # Check size
        if max_file_size:
            try:
                size = os.path.getsize(filepath)
                if size > max_file_size:
                    return (relpath, None, "size_limit")
            except Exception:
                return (relpath, None, "size_error")

        # Read file
        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                content = f.read()
            return (relpath, content, "success")
        except Exception as e:
            return (relpath, None, f"read_error: {e}")

    # OPTIMIZED: Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, fp): fp for fp in candidate_files
        }

        for future in as_completed(future_to_file):
            relpath, content, status = future.result()
            processed_count += 1

            if content is not None:
                files_dict[relpath] = content
                included_count += 1

            # OPTIMIZED: Less verbose progress (every 10% or every 100 files)
            if (
                processed_count % max(1, total_files // 10) == 0
                or processed_count % 100 == 0
            ):
                percentage = int((processed_count / total_files) * 100)
                print(
                    f"Progress: {processed_count}/{total_files} ({percentage}%) - {included_count} included"
                )

    print(f"✓ Crawled {included_count}/{total_files} files")
    return {"files": files_dict}


if __name__ == "__main__":
    print("--- Testing optimized local crawler ---")
    files_data = crawl_local_files(
        "..",
        exclude_patterns={
            "*.pyc",
            "__pycache__/*",
            ".venv/*",
            ".git/*",
            "docs/*",
            "output/*",
        },
    )
    print(f"\nFound {len(files_data['files'])} files")
