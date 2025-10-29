import os
import re
import yaml
from pocketflow import Node, BatchNode
from utils.crawl_github_files import crawl_github_files
from utils.call_llm import (
    call_llm,
    get_smart_context,
    optimize_context_for_budget,
    estimate_tokens,
)
from utils.crawl_local_files import crawl_local_files

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# ========= OPTIMIZED TOKEN BUDGETS =========
# More aggressive budgeting with real-world measurements
def _calculate_smart_token_budget():
    """
    Optimized token budget calculation with less conservative margins.
    Based on empirical measurements showing actual overhead is ~40-50k, not 65k.
    """
    context_limit = int(os.getenv("OPENAI_TPM_LIMIT", "450000"))
    max_output_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))
    max_input_size = context_limit - max_output_tokens

    # OPTIMIZED: Reduced overhead from 65k to 45k based on real measurements
    # This gives us 20k more tokens for file content
    prompt_overhead = 45000
    available_for_content = max_input_size - prompt_overhead

    # OPTIMIZED: Increased safety margin from 80% to 90%
    # With better optimization algorithms, we can be more aggressive
    safe_budget = int(available_for_content * 0.90)

    explicit_budget = os.getenv("TOKEN_BUDGET_IDENTIFY")
    if explicit_budget:
        explicit_value = int(explicit_budget)
        if explicit_value > safe_budget:
            print(
                f"⚠️  Warning: TOKEN_BUDGET_IDENTIFY ({explicit_value:,}) exceeds optimized budget ({safe_budget:,})"
            )
            return safe_budget
        return explicit_value

    return safe_budget


TOKEN_BUDGETS = {
    "identify_abstractions": _calculate_smart_token_budget(),
    "analyze_and_order": int(
        os.getenv("TOKEN_BUDGET_RELATIONSHIPS", "18000")
    ),  # Increased for merged node
    "write_chapter": int(os.getenv("TOKEN_BUDGET_CHAPTER", "12000")),
}


def extract_yaml_from_response(response: str) -> str:
    """Robustly extract YAML content from LLM response."""
    yaml_str = None

    if "```yaml" in response:
        try:
            yaml_str = response.strip().split("```yaml")[1].split("```")[0].strip()
        except IndexError:
            pass

    if not yaml_str and "```" in response:
        try:
            parts = response.split("```")
            if len(parts) >= 3:
                yaml_str = parts[1].strip()
                if yaml_str.startswith("yaml\n") or yaml_str.startswith("yml\n"):
                    yaml_str = "\n".join(yaml_str.split("\n")[1:])
        except Exception:
            pass

    if not yaml_str:
        yaml_str = response.strip()

    if not yaml_str:
        raise ValueError(
            f"Could not extract YAML from LLM response. Response preview: {response[:500]}"
        )

    return yaml_str


def get_optimized_content_for_indices(
    files_data, indices, token_budget=8000, priority_keywords=None
):
    """Get optimized content for file indices with intelligent chunking."""
    return get_smart_context(
        files_data=files_data,
        indices=indices,
        token_budget=token_budget,
        priority_keywords=priority_keywords,
    )


def generate_jekyll_front_matter(
    title, nav_order=None, parent=None, has_children=False, layout="default"
):
    """Generate Jekyll front matter YAML for documentation files."""
    front_matter = ["---"]
    front_matter.append(f"layout: {layout}")
    front_matter.append(f'title: "{title}"')

    if nav_order is not None:
        front_matter.append(f"nav_order: {nav_order}")

    if parent:
        front_matter.append(f'parent: "{parent}"')

    if has_children:
        front_matter.append("has_children: true")

    front_matter.append("---")
    return "\n".join(front_matter) + "\n\n"


class FetchRepo(Node):
    def prep(self, shared):
        repo_url = shared.get("repo_url")
        local_dir = shared.get("local_dir")
        project_name = shared.get("project_name")

        if not project_name:
            if repo_url:
                project_name = repo_url.split("/")[-1].replace(".git", "")
            else:
                project_name = os.path.basename(os.path.abspath(local_dir))
            shared["project_name"] = project_name

        return {
            "repo_url": repo_url,
            "local_dir": local_dir,
            "token": shared.get("github_token"),
            "include_patterns": shared["include_patterns"],
            "exclude_patterns": shared["exclude_patterns"],
            "max_file_size": shared["max_file_size"],
            "use_relative_paths": True,
        }

    def exec(self, prep_res):
        if prep_res["repo_url"]:
            print(f"Crawling repository: {prep_res['repo_url']}...")
            result = crawl_github_files(
                repo_url=prep_res["repo_url"],
                token=prep_res["token"],
                include_patterns=prep_res["include_patterns"],
                exclude_patterns=prep_res["exclude_patterns"],
                max_file_size=prep_res["max_file_size"],
                use_relative_paths=prep_res["use_relative_paths"],
            )
        else:
            print(f"Crawling directory: {prep_res['local_dir']}...")
            result = crawl_local_files(
                directory=prep_res["local_dir"],
                include_patterns=prep_res["include_patterns"],
                exclude_patterns=prep_res["exclude_patterns"],
                max_file_size=prep_res["max_file_size"],
                use_relative_paths=prep_res["use_relative_paths"],
            )

        files_list = list(result.get("files", {}).items())
        if len(files_list) == 0:
            raise ValueError("Failed to fetch files")
        print(f"✓ Fetched {len(files_list)} files")
        return files_list

    def post(self, shared, prep_res, exec_res):
        shared["files"] = exec_res


class IdentifyAbstractions(Node):
    def prep(self, shared):
        files_data = shared["files"]
        project_name = shared["project_name"]
        language = shared.get("language", "english")
        use_cache = shared.get("use_cache", True)
        max_abstraction_num = shared.get("max_abstraction_num", 10)

        token_budget = TOKEN_BUDGETS["identify_abstractions"]

        files_content_map = {}
        file_info = []
        for i, (path, content) in enumerate(files_data):
            files_content_map[f"{i} # {path}"] = content
            file_info.append((i, path))

        priority_keywords = ["main", "core", "app", "index", "init", "base", "api"]
        context = optimize_context_for_budget(
            files_content_map,
            token_budget=token_budget,
            use_structure_for_large=True,
            priority_keywords=priority_keywords,
        )

        actual_tokens = estimate_tokens(context)
        print(
            f"Context optimization: {len(files_data)} files -> {actual_tokens:,}/{token_budget:,} tokens ({actual_tokens * 100 // token_budget}%)"
        )

        file_listing_for_prompt = "\n".join(
            [f"- {idx} # {path}" for idx, path in file_info]
        )
        return (
            context,
            file_listing_for_prompt,
            len(files_data),
            project_name,
            language,
            use_cache,
            max_abstraction_num,
        )

    def exec(self, prep_res):
        (
            context,
            file_listing_for_prompt,
            file_count,
            project_name,
            language,
            use_cache,
            max_abstraction_num,
        ) = prep_res
        print(f"Identifying abstractions using LLM...")

        language_instruction = ""
        name_lang_hint = ""
        desc_lang_hint = ""
        if language.lower() != "english":
            language_instruction = f"IMPORTANT: Generate the `name` and `description` for each abstraction in **{language.capitalize()}** language. Do NOT use English for these fields.\n\n"
            name_lang_hint = f" (value in {language.capitalize()})"
            desc_lang_hint = f" (value in {language.capitalize()})"

        prompt = f"""
For the project `{project_name}`:

Codebase Context:
{context}

{language_instruction}Analyze the codebase context.
Identify the top 5-{max_abstraction_num} core most important abstractions to help those new to the codebase.

For each abstraction, provide:
1. A concise `name`{name_lang_hint}.
2. A beginner-friendly `description` explaining what it is with a simple analogy, in around 100 words{desc_lang_hint}.
3. A list of relevant `file_indices` (integers) using the format `idx # path/comment`.

List of file indices and paths present in the context:
{file_listing_for_prompt}

Format the output as a YAML list of dictionaries:

```yaml
- name: |
    Query Processing{name_lang_hint}
  description: |
    Explains what the abstraction does.
    It's like a central dispatcher routing requests.{desc_lang_hint}
  file_indices:
    - 0 # path/to/file1.py
    - 3 # path/to/related.py
- name: |
    Query Optimization{name_lang_hint}
  description: |
    Another core concept, similar to a blueprint for objects.{desc_lang_hint}
  file_indices:
    - 5 # path/to/another.js
# ... up to {max_abstraction_num} abstractions
```"""
        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))

        yaml_str = extract_yaml_from_response(response)

        try:
            abstractions = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Failed to parse YAML: {e}\nYAML string: {yaml_str[:500]}"
            )

        if not isinstance(abstractions, list):
            raise ValueError("LLM Output is not a list")

        validated_abstractions = []
        for item in abstractions:
            if not isinstance(item, dict) or not all(
                k in item for k in ["name", "description", "file_indices"]
            ):
                raise ValueError(f"Missing keys in abstraction item: {item}")
            if not isinstance(item["name"], str):
                raise ValueError(f"Name is not a string in item: {item}")
            if not isinstance(item["description"], str):
                raise ValueError(f"Description is not a string in item: {item}")
            if not isinstance(item["file_indices"], list):
                raise ValueError(f"file_indices is not a list in item: {item}")

            validated_indices = []
            for idx_entry in item["file_indices"]:
                try:
                    if isinstance(idx_entry, int):
                        idx = idx_entry
                    elif isinstance(idx_entry, str) and "#" in idx_entry:
                        idx = int(idx_entry.split("#")[0].strip())
                    else:
                        idx = int(str(idx_entry).strip())

                    if not (0 <= idx < file_count):
                        raise ValueError(
                            f"Invalid file index {idx} found in item {item['name']}. Max index is {file_count - 1}."
                        )
                    validated_indices.append(idx)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Could not parse index from entry: {idx_entry} in item {item['name']}"
                    )

            item["files"] = sorted(list(set(validated_indices)))
            validated_abstractions.append(
                {
                    "name": item["name"],
                    "description": item["description"],
                    "files": item["files"],
                }
            )

        print(f"✓ Identified {len(validated_abstractions)} abstractions")
        return validated_abstractions

    def post(self, shared, prep_res, exec_res):
        shared["abstractions"] = exec_res


# ========= OPTIMIZED: MERGED NODE =========
class AnalyzeAndOrderChapters(Node):
    """
    OPTIMIZATION: Merged AnalyzeRelationships + OrderChapters into one LLM call.
    This reduces latency by ~20-40s and improves coherence since ordering
    naturally depends on relationships.
    """

    def prep(self, shared):
        abstractions = shared["abstractions"]
        files_data = shared["files"]
        project_name = shared["project_name"]
        language = shared.get("language", "english")
        use_cache = shared.get("use_cache", True)

        num_abstractions = len(abstractions)

        # Build context
        context = "Identified Abstractions:\n"
        all_relevant_indices = set()
        abstraction_info_for_prompt = []

        for i, abstr in enumerate(abstractions):
            file_indices_str = ", ".join(map(str, abstr["files"]))
            info_line = f"- Index {i}: {abstr['name']} (Relevant file indices: [{file_indices_str}])\n  Description: {abstr['description']}"
            context += info_line + "\n"
            abstraction_info_for_prompt.append(f"{i} # {abstr['name']}")
            all_relevant_indices.update(abstr["files"])

        context += "\nRelevant File Snippets (Referenced by Index and Path):\n"
        token_budget = TOKEN_BUDGETS["analyze_and_order"]
        file_context_str = get_optimized_content_for_indices(
            files_data,
            sorted(list(all_relevant_indices)),
            token_budget=token_budget,
            priority_keywords=["main", "core", "base", "api"],
        )
        context += file_context_str

        total_tokens = estimate_tokens(context)
        print(
            f"Analysis context: {len(all_relevant_indices)} files -> {total_tokens:,} tokens"
        )

        return (
            context,
            "\n".join(abstraction_info_for_prompt),
            num_abstractions,
            project_name,
            language,
            use_cache,
        )

    def exec(self, prep_res):
        (
            context,
            abstraction_listing,
            num_abstractions,
            project_name,
            language,
            use_cache,
        ) = prep_res
        print(f"Analyzing relationships and ordering chapters using LLM...")

        language_instruction = ""
        lang_hint = ""
        list_lang_note = ""
        if language.lower() != "english":
            language_instruction = f"IMPORTANT: Generate the `summary` and relationship `label` fields in **{language.capitalize()}** language. Do NOT use English for these fields.\n\n"
            lang_hint = f" (in {language.capitalize()})"
            list_lang_note = f" (Names might be in {language.capitalize()})"

        prompt = f"""
Based on the following abstractions and relevant code snippets from the project `{project_name}`:

List of Abstraction Indices and Names{list_lang_note}:
{abstraction_listing}

Context (Abstractions, Descriptions, Code):
{context}

{language_instruction}Please provide:
1. A high-level `summary` of the project's main purpose and functionality in a few beginner-friendly sentences{lang_hint}. Use markdown formatting with **bold** and *italic* text to highlight important concepts.

2. A list (`relationships`) describing the key interactions between these abstractions. For each relationship, specify:
    - `from_abstraction`: Index of the source abstraction (e.g., `0 # AbstractionName1`)
    - `to_abstraction`: Index of the target abstraction (e.g., `1 # AbstractionName2`)
    - `label`: A brief label for the interaction **in just a few words**{lang_hint} (e.g., "Manages", "Inherits", "Uses").
    
IMPORTANT: Make sure EVERY abstraction is involved in at least ONE relationship (either as source or target).

3. A `chapter_order` list showing the best order to explain these abstractions in a tutorial, from first to last. Start with foundational or user-facing concepts, then move to detailed implementations. Use the format `idx # AbstractionName`.

Format the output as YAML:

```yaml
summary: |
  A brief, simple explanation of the project{lang_hint}.
  Can span multiple lines with **bold** and *italic* for emphasis.
relationships:
  - from_abstraction: 0 # AbstractionName1
    to_abstraction: 1 # AbstractionName2
    label: "Manages"{lang_hint}
  - from_abstraction: 2 # AbstractionName3
    to_abstraction: 0 # AbstractionName1
    label: "Provides config"{lang_hint}
chapter_order:
  - 2 # FoundationalConcept
  - 0 # CoreClassA
  - 1 # CoreClassB
  # ... all {num_abstractions} abstractions in order
```

Now, provide the YAML output:
"""
        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))

        yaml_str = extract_yaml_from_response(response)
        try:
            data = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Failed to parse YAML: {e}\nYAML string: {yaml_str[:500]}"
            )

        if not isinstance(data, dict) or not all(
            k in data for k in ["summary", "relationships", "chapter_order"]
        ):
            raise ValueError("LLM output is not a dict or missing keys")

        # Validate relationships
        validated_relationships = []
        for rel in data["relationships"]:
            if not isinstance(rel, dict) or not all(
                k in rel for k in ["from_abstraction", "to_abstraction", "label"]
            ):
                raise ValueError(f"Missing keys in relationship: {rel}")
            if not isinstance(rel["label"], str):
                raise ValueError(f"Relationship label is not a string: {rel}")

            try:
                from_idx = int(str(rel["from_abstraction"]).split("#")[0].strip())
                to_idx = int(str(rel["to_abstraction"]).split("#")[0].strip())
                if not (
                    0 <= from_idx < num_abstractions and 0 <= to_idx < num_abstractions
                ):
                    raise ValueError(
                        f"Invalid index in relationship: from={from_idx}, to={to_idx}"
                    )
                validated_relationships.append(
                    {
                        "from": from_idx,
                        "to": to_idx,
                        "label": rel["label"],
                    }
                )
            except (ValueError, TypeError):
                raise ValueError(f"Could not parse indices from relationship: {rel}")

        # Validate chapter order
        if not isinstance(data["chapter_order"], list):
            raise ValueError("chapter_order is not a list")

        ordered_indices = []
        seen_indices = set()
        for entry in data["chapter_order"]:
            try:
                if isinstance(entry, int):
                    idx = entry
                elif isinstance(entry, str) and "#" in entry:
                    idx = int(entry.split("#")[0].strip())
                else:
                    idx = int(str(entry).strip())

                if not (0 <= idx < num_abstractions):
                    raise ValueError(f"Invalid index {idx} in chapter_order")
                if idx in seen_indices:
                    raise ValueError(f"Duplicate index {idx} in chapter_order")
                ordered_indices.append(idx)
                seen_indices.add(idx)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Could not parse index from chapter_order entry: {entry}"
                )

        if len(ordered_indices) != num_abstractions:
            raise ValueError(
                f"Chapter order length ({len(ordered_indices)}) != number of abstractions ({num_abstractions})"
            )

        print(
            f"✓ Generated summary, {len(validated_relationships)} relationships, and chapter order"
        )

        return {
            "summary": data["summary"],
            "relationships": validated_relationships,
            "chapter_order": ordered_indices,
        }

    def post(self, shared, prep_res, exec_res):
        shared["relationships"] = {
            "summary": exec_res["summary"],
            "details": exec_res["relationships"],
        }
        shared["chapter_order"] = exec_res["chapter_order"]


class WriteChapters(BatchNode):
    def prep(self, shared):
        chapter_order = shared["chapter_order"]
        abstractions = shared["abstractions"]
        files_data = shared["files"]
        project_name = shared["project_name"]
        language = shared.get("language", "english")
        use_cache = shared.get("use_cache", True)

        self.chapters_written_so_far = []

        all_chapters = []
        chapter_filenames = {}
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions):
                chapter_num = i + 1
                chapter_name = abstractions[abstraction_index]["name"]
                safe_name = "".join(
                    c if c.isalnum() else "_" for c in chapter_name
                ).lower()
                filename = f"{i + 1:02d}_{safe_name}.md"
                all_chapters.append(f"{chapter_num}. [{chapter_name}]({filename})")
                chapter_filenames[abstraction_index] = {
                    "num": chapter_num,
                    "name": chapter_name,
                    "filename": filename,
                }

        full_chapter_listing = "\n".join(all_chapters)

        items_to_process = []
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions):
                abstraction_details = abstractions[abstraction_index]
                related_file_indices = abstraction_details.get("files", [])

                prev_chapter = None
                if i > 0:
                    prev_idx = chapter_order[i - 1]
                    prev_chapter = chapter_filenames[prev_idx]

                next_chapter = None
                if i < len(chapter_order) - 1:
                    next_idx = chapter_order[i + 1]
                    next_chapter = chapter_filenames[next_idx]

                items_to_process.append(
                    {
                        "chapter_num": i + 1,
                        "abstraction_index": abstraction_index,
                        "abstraction_details": abstraction_details,
                        "related_file_indices": related_file_indices,
                        "files_data": files_data,
                        "project_name": project_name,
                        "full_chapter_listing": full_chapter_listing,
                        "chapter_filenames": chapter_filenames,
                        "prev_chapter": prev_chapter,
                        "next_chapter": next_chapter,
                        "language": language,
                        "use_cache": use_cache,
                    }
                )
            else:
                print(
                    f"Warning: Invalid abstraction index {abstraction_index} in chapter_order"
                )

        print(f"Preparing to write {len(items_to_process)} chapters...")
        return items_to_process

    def exec(self, item):
        abstraction_name = item["abstraction_details"]["name"]
        abstraction_description = item["abstraction_details"]["description"]
        chapter_num = item["chapter_num"]
        project_name = item.get("project_name")
        language = item.get("language", "english")
        use_cache = item.get("use_cache", True)
        print(f"Writing chapter {chapter_num}: {abstraction_name[:50]}...")

        token_budget = TOKEN_BUDGETS["write_chapter"]
        file_context_str = get_optimized_content_for_indices(
            item["files_data"],
            item["related_file_indices"],
            token_budget=token_budget,
            priority_keywords=[abstraction_name.lower().split()[0]],
        )

        context_tokens = estimate_tokens(file_context_str)
        print(
            f"  Chapter {chapter_num} context: {len(item['related_file_indices'])} files -> {context_tokens:,}/{token_budget:,} tokens"
        )

        previous_chapters_summary = "\n---\n".join(self.chapters_written_so_far)

        language_instruction = ""
        if language.lower() != "english":
            lang_cap = language.capitalize()
            language_instruction = f"IMPORTANT: Write this ENTIRE tutorial chapter in **{lang_cap}**. Translate ALL generated content including explanations, examples, and technical terms into {lang_cap}.\n\n"

        prompt = f"""
{language_instruction}Write a very beginner-friendly tutorial chapter (in Markdown format) for the project `{project_name}` about the concept: "{abstraction_name}". This is Chapter {chapter_num}.

Concept Details:
- Name: {abstraction_name}
- Description:
{abstraction_description}

Complete Tutorial Structure:
{item["full_chapter_listing"]}

Context from previous chapters:
{previous_chapters_summary if previous_chapters_summary else "This is the first chapter."}

Relevant Code Snippets:
{file_context_str if file_context_str else "No specific code snippets provided for this abstraction."}

Instructions:
- Start with `# Chapter {chapter_num}: {abstraction_name}`
- Begin with high-level motivation and a concrete use case
- Break complex concepts into beginner-friendly pieces
- Keep code blocks BELOW 10 lines - break longer ones into pieces
- Use mermaid diagrams for complex flows (max 5 participants)
- Link to other chapters using Markdown links from the structure above
- Use analogies and examples throughout
- End with a brief conclusion and transition to next chapter

Output only the Markdown content (no ```markdown``` tags):
"""
        chapter_content = call_llm(
            prompt, use_cache=(use_cache and self.cur_retry == 0)
        )

        actual_heading = f"# Chapter {chapter_num}: {abstraction_name}"
        if not chapter_content.strip().startswith(f"# Chapter {chapter_num}"):
            lines = chapter_content.strip().split("\n")
            if lines and lines[0].strip().startswith("#"):
                lines[0] = actual_heading
                chapter_content = "\n".join(lines)
            else:
                chapter_content = f"{actual_heading}\n\n{chapter_content}"

        self.chapters_written_so_far.append(chapter_content)
        return chapter_content

    def post(self, shared, prep_res, exec_res_list):
        shared["chapters"] = exec_res_list
        del self.chapters_written_so_far
        print(f"✓ Finished writing {len(exec_res_list)} chapters")


class CombineTutorial(Node):
    def prep(self, shared):
        project_name = shared["project_name"]
        output_base_dir = shared.get("output_dir", "output")
        output_path = os.path.join(output_base_dir, project_name)
        repo_url = shared.get("repo_url")

        enable_jekyll = shared.get("enable_jekyll", True)
        jekyll_nav_order = shared.get("jekyll_nav_order", 1)

        relationships_data = shared["relationships"]
        chapter_order = shared["chapter_order"]
        abstractions = shared["abstractions"]
        chapters_content = shared["chapters"]

        # Generate Mermaid diagram
        mermaid_lines = ["flowchart TD"]
        for i, abstr in enumerate(abstractions):
            node_id = f"A{i}"
            sanitized_name = abstr["name"].replace('"', "")
            mermaid_lines.append(f'    {node_id}["{sanitized_name}"]')

        for rel in relationships_data["details"]:
            from_node_id = f"A{rel['from']}"
            to_node_id = f"A{rel['to']}"
            edge_label = rel["label"].replace('"', "").replace("\n", " ")
            if len(edge_label) > 30:
                edge_label = edge_label[:27] + "..."
            mermaid_lines.append(
                f'    {from_node_id} -- "{edge_label}" --> {to_node_id}'
            )

        mermaid_diagram = "\n".join(mermaid_lines)

        # Prepare index.md
        index_content = ""
        if enable_jekyll:
            index_content += generate_jekyll_front_matter(
                title=project_name, nav_order=jekyll_nav_order, has_children=True
            )

        index_content += f"# Tutorial: {project_name}\n\n"

        if enable_jekyll:
            index_content += f"> This tutorial is AI-generated! To learn more, check out [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)\n\n"

        index_content += f"{relationships_data['summary']}\n\n"

        if repo_url:
            index_content += f"**Source Repository:** [{repo_url}]({repo_url})\n\n"

        index_content += "```mermaid\n" + mermaid_diagram + "\n```\n\n"
        index_content += f"## Chapters\n\n"

        chapter_files = []
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions) and i < len(chapters_content):
                abstraction_name = abstractions[abstraction_index]["name"]
                safe_name = "".join(
                    c if c.isalnum() else "_" for c in abstraction_name
                ).lower()
                filename = f"{i + 1:02d}_{safe_name}.md"
                index_content += f"{i + 1}. [{abstraction_name}]({filename})\n"

                chapter_content = ""
                if enable_jekyll:
                    chapter_content += generate_jekyll_front_matter(
                        title=abstraction_name, parent=project_name, nav_order=i + 1
                    )

                chapter_content += chapters_content[i]

                if not chapter_content.endswith("\n\n"):
                    chapter_content += "\n\n"
                chapter_content += f"---\n\nGenerated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)"

                chapter_files.append(
                    {
                        "filename": filename,
                        "content": chapter_content,
                        "title": abstraction_name,
                    }
                )
            else:
                print(
                    f"Warning: Mismatch at chapter {i}, abstraction index {abstraction_index}"
                )

        index_content += f"\n\n---\n\nGenerated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)"

        return {
            "output_path": output_path,
            "index_content": index_content,
            "chapter_files": chapter_files,
            "enable_jekyll": enable_jekyll,
        }

    def exec(self, prep_res):
        output_path = prep_res["output_path"]
        index_content = prep_res["index_content"]
        chapter_files = prep_res["chapter_files"]
        enable_jekyll = prep_res["enable_jekyll"]

        print(f"Combining tutorial into: {output_path}")
        if enable_jekyll:
            print(f"  ✓ Jekyll front matter enabled")

        os.makedirs(output_path, exist_ok=True)

        index_filepath = os.path.join(output_path, "index.md")
        with open(index_filepath, "w", encoding="utf-8") as f:
            f.write(index_content)
        print(f"  ✓ Wrote index.md")

        for chapter_info in chapter_files:
            chapter_filepath = os.path.join(output_path, chapter_info["filename"])
            with open(chapter_filepath, "w", encoding="utf-8") as f:
                f.write(chapter_info["content"])
        print(f"  ✓ Wrote {len(chapter_files)} chapter files")

        return output_path

    def post(self, shared, prep_res, exec_res):
        shared["final_output_dir"] = exec_res
        print(f"\n✓ Tutorial generation complete! Files in: {exec_res}")
