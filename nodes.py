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
        max_abstraction_num = shared.get("max_abstraction_num", 100)
        doc_mode = shared.get("doc_mode", "developer")

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
            doc_mode,
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
            doc_mode,
        ) = prep_res
        print(f"Identifying abstractions using LLM (mode: {doc_mode})...")

        language_instruction = ""
        name_lang_hint = ""
        desc_lang_hint = ""
        if language.lower() != "english":
            language_instruction = f"IMPORTANT: Generate the `name` and `description` for each abstraction in **{language.capitalize()}** language. Do NOT use English for these fields.\n\n"
            name_lang_hint = f" (value in {language.capitalize()})"
            desc_lang_hint = f" (value in {language.capitalize()})"

        # MODE-SPECIFIC INSTRUCTIONS
        if doc_mode == "developer":
            mode_instruction = """
Focus on TECHNICAL abstractions that developers need to understand:
- Core classes, modules, and architectural components
- Important design patterns and data structures
- Key algorithms and processing logic
- API endpoints and interfaces
- Configuration and initialization systems

Descriptions should be technical and include:
- What the component does at a code level
- Its role in the architecture
- Key methods or functions it provides
"""
        else:  # user mode
            mode_instruction = """
Focus on BUSINESS abstractions that end-users and stakeholders need to understand:
- User-facing features and workflows
- Main application functionalities
- Business processes and logic flows
- Data models from a business perspective
- Integration points with external systems

Descriptions should be non-technical and include:
- What the feature does for the user
- Real-world use cases and benefits
- How it fits into the overall product
- Simple analogies to familiar concepts (avoid code-level details)
"""

        prompt = f"""
For the project `{project_name}`:

Codebase Context:
{context}

{language_instruction}DOCUMENTATION MODE: {doc_mode.upper()}

{mode_instruction}

Analyze the codebase context and identify the top 5-{max_abstraction_num} most important abstractions based on the {doc_mode} perspective.

For each abstraction, provide:
1. A concise `name`{name_lang_hint}.
2. A {"beginner-friendly" if doc_mode == "user" else "clear, technical"} `description` explaining what it is, in around 100-150 words{desc_lang_hint}.
   {"- Use simple analogies and avoid technical jargon" if doc_mode == "user" else "- Include technical details and implementation context"}
   {"- Focus on user benefits and business value" if doc_mode == "user" else "- Focus on architectural role and code structure"}
3. A list of relevant `file_indices` (integers) using the format `idx # path/comment`.

List of file indices and paths present in the context:
{file_listing_for_prompt}

Format the output as a YAML list of dictionaries:

```yaml
- name: |
    {"User Authentication System" if doc_mode == "user" else "Authentication Service Layer"}{name_lang_hint}
  description: |
    {"This is how users securely access the application. Think of it like a digital doorman that checks ID cards - it verifies who you are and grants access to your account. The system ensures your data stays private and only you can access your information." if doc_mode == "user" else "Core authentication service implementing JWT-based token management. Handles user login, session management, and token validation. Uses bcrypt for password hashing and implements refresh token rotation for security."}{desc_lang_hint}
  file_indices:
    - 0 # path/to/auth_service.py
    - 3 # path/to/jwt_handler.py
- name: |
    {"Payment Processing" if doc_mode == "user" else "Payment Gateway Integration Layer"}{name_lang_hint}
  description: |
    {"Handles all payment transactions securely. Like a digital cashier, it processes your credit card or payment method, ensures the transaction is safe, and confirms your purchase. It works with various payment providers to give you flexibility in how you pay." if doc_mode == "user" else "Abstraction layer for third-party payment gateways (Stripe, PayPal). Implements retry logic, webhook handling, and transaction state management. Ensures PCI compliance through tokenization."}{desc_lang_hint}
  file_indices:
    - 5 # path/to/payment_gateway.js
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

        print(
            f"✓ Identified {len(validated_abstractions)} abstractions ({doc_mode} mode)"
        )
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
        doc_mode = shared.get("doc_mode", "developer")

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
            doc_mode,
        )

    def exec(self, prep_res):
        (
            context,
            abstraction_listing,
            num_abstractions,
            project_name,
            language,
            use_cache,
            doc_mode,
        ) = prep_res
        print(
            f"Analyzing relationships and ordering chapters using LLM (mode: {doc_mode})..."
        )

        language_instruction = ""
        lang_hint = ""
        list_lang_note = ""
        if language.lower() != "english":
            language_instruction = f"IMPORTANT: Generate the `summary` and relationship `label` fields in **{language.capitalize()}** language. Do NOT use English for these fields.\n\n"
            lang_hint = f" (in {language.capitalize()})"
            list_lang_note = f" (Names might be in {language.capitalize()})"

        # MODE-SPECIFIC INSTRUCTIONS
        if doc_mode == "developer":
            summary_instruction = """
Write a technical `summary` of the project that:
- Explains the overall architecture and technical approach
- Highlights key technologies, frameworks, and design patterns used
- Describes the main components and how they interact
- Mentions notable technical decisions or innovations
Target audience: Software developers who want to understand the codebase structure.
"""
            relationship_instruction = """
For `relationships`, focus on TECHNICAL connections:
- Code dependencies (imports, inheritance, composition)
- Data flow between components
- API calls and service interactions
- Configuration and initialization order
Use technical labels like: "Implements", "Extends", "Depends on", "Initializes", "Calls", "Provides data to"
"""
            ordering_instruction = """
For `chapter_order`, arrange topics in a TECHNICAL learning path:
1. Start with core infrastructure and base classes
2. Move to main service/business logic layers
3. Then specialized features and extensions
4. End with configuration, utilities, and integration layers
This helps developers build up from foundational code to higher-level features.
"""
        else:  # user mode
            summary_instruction = """
Write a business-focused `summary` of the project that:
- Explains what the product does and who it's for
- Highlights key features and benefits for end-users
- Describes the main use cases and workflows
- Mentions the business value and goals of the project
Target audience: End-users, product managers, and non-technical stakeholders.
"""
            relationship_instruction = """
For `relationships`, focus on BUSINESS/WORKFLOW connections:
- How features work together in user workflows
- Which features enable or support other features
- Business process dependencies
- User journey connections
Use business labels like: "Enables", "Supports", "Feeds into", "Requires", "Enhances", "Complements"
"""
            ordering_instruction = """
For `chapter_order`, arrange topics in a USER-CENTRIC learning path:
1. Start with core user-facing features (what users see first)
2. Move to supporting features that enhance the core experience
3. Then advanced features and workflows
4. End with administrative or configuration features
This helps users understand the product from their perspective and natural usage flow.
"""

        prompt = f"""
Based on the following abstractions and relevant code snippets from the project `{project_name}`:

DOCUMENTATION MODE: {doc_mode.upper()}

List of Abstraction Indices and Names{list_lang_note}:
{abstraction_listing}

Context (Abstractions, Descriptions, Code):
{context}

{language_instruction}Please provide:

1. PROJECT SUMMARY:
{summary_instruction}
Use markdown formatting with **bold** and *italic* text to highlight important concepts.

2. RELATIONSHIPS:
{relationship_instruction}
    - `from_abstraction`: Index of the source abstraction (e.g., `0 # AbstractionName1`)
    - `to_abstraction`: Index of the target abstraction (e.g., `1 # AbstractionName2`)
    - `label`: A brief label for the interaction **in just a few words**{lang_hint}
    
IMPORTANT: Make sure EVERY abstraction is involved in at least ONE relationship (either as source or target).

3. CHAPTER ORDER:
{ordering_instruction}
Use the format `idx # AbstractionName` for all {num_abstractions} abstractions.

Format the output as YAML:

```yaml
summary: |
  {"**ProjectX** is a web application that helps small businesses manage their inventory and sales. It provides an intuitive interface for tracking products, processing orders, and generating reports. The system *streamlines operations* by automating routine tasks and providing real-time insights." if doc_mode == "user" else "**ProjectX** is built using a microservices architecture with Node.js and React. It implements *event-driven patterns* for scalability and uses PostgreSQL for data persistence. The system follows **Domain-Driven Design** principles with clear separation between API, business logic, and data layers."}{lang_hint}
relationships:
  - from_abstraction: 0 # {"User Dashboard" if doc_mode == "user" else "AuthService"}
    to_abstraction: 1 # {"Order Processing" if doc_mode == "user" else "UserRepository"}
    label: "{"Displays" if doc_mode == "user" else "Queries"}"{lang_hint}
  - from_abstraction: 2 # {"Inventory Management" if doc_mode == "user" else "OrderController"}
    to_abstraction: 0 # {"User Dashboard" if doc_mode == "user" else "AuthService"}
    label: "{"Updates" if doc_mode == "user" else "Authenticates with"}"{lang_hint}
chapter_order:
  - {"0 # User Dashboard" if doc_mode == "user" else "2 # CoreModule"}
  - {"1 # Order Processing" if doc_mode == "user" else "0 # AuthService"}
  - {"2 # Inventory Management" if doc_mode == "user" else "1 # UserRepository"}
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
            f"✓ Generated summary, {len(validated_relationships)} relationships, and chapter order ({doc_mode} mode)"
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
        doc_mode = shared.get("doc_mode", "developer")

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
                        "doc_mode": doc_mode,
                    }
                )
            else:
                print(
                    f"Warning: Invalid abstraction index {abstraction_index} in chapter_order"
                )

        print(
            f"Preparing to write {len(items_to_process)} chapters ({doc_mode} mode)..."
        )
        return items_to_process

    def exec(self, item):
        abstraction_name = item["abstraction_details"]["name"]
        abstraction_description = item["abstraction_details"]["description"]
        chapter_num = item["chapter_num"]
        project_name = item.get("project_name")
        language = item.get("language", "english")
        use_cache = item.get("use_cache", True)
        doc_mode = item.get("doc_mode", "developer")

        print(
            f"Writing chapter {chapter_num} ({doc_mode} mode): {abstraction_name[:50]}..."
        )

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

        # MODE-SPECIFIC INSTRUCTIONS
        if doc_mode == "developer":
            mode_specific_instructions = """
DEVELOPER MODE - Technical Documentation:

Structure your chapter with these sections:
1. **Overview**: Technical introduction to the component
2. **Architecture & Design**: How it's structured and why
3. **Key Components**: Main classes, functions, or modules with code examples
4. **Implementation Details**: 
   - Code snippets showing important methods (keep each snippet under 15 lines)
   - Explanation of algorithms or logic
   - Design patterns used
5. **API Reference**: Key functions/methods with parameters and return types
6. **Usage Examples**: Practical code examples showing how to use this component
7. **Integration Points**: How this connects to other parts of the codebase
8. **Technical Considerations**: Performance, security, or scalability notes

Style Guidelines:
- Use technical terminology appropriately
- Include code snippets with syntax highlighting
- Show actual file paths and line numbers when referencing code
- Explain WHY certain technical decisions were made
- Use mermaid diagrams for complex architectural flows (see examples below)
- Link to related technical chapters
- Keep code examples focused and well-commented

Example code block format:
```python
# From src/auth/jwt_handler.py
def generate_token(user_id: str, expires_in: int = 3600) -> str:
    \"\"\"Generate JWT token for authenticated user.
    
    Args:
        user_id: Unique identifier for the user
        expires_in: Token expiration time in seconds
    
    Returns:
        Signed JWT token string
    \"\"\"
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(seconds=expires_in)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
```

Mermaid Diagram Examples (Mermaid 11.6.0 Compatible):

**1. Sequence Diagram** - For API calls, service interactions, authentication flows:
```mermaid
sequenceDiagram
    participant Client
    participant AuthService
    participant Database
    participant JWTHandler
    
    Client->>AuthService: POST /login
    activate AuthService
    AuthService->>Database: Verify credentials
    activate Database
    Database-->>AuthService: User data
    deactivate Database
    AuthService->>JWTHandler: Generate token
    activate JWTHandler
    JWTHandler-->>AuthService: JWT token
    deactivate JWTHandler
    AuthService-->>Client: 200 OK + token
    deactivate AuthService
```

**2. Flowchart** - For algorithms, decision logic, processing flows:
```mermaid
flowchart TD
    Start([Request Received]) --> ValidateInput{Valid Input?}
    ValidateInput -->|No| ReturnError[Return 400 Error]
    ValidateInput -->|Yes| CheckAuth{Authenticated?}
    CheckAuth -->|No| ReturnUnauth[Return 401 Unauthorized]
    CheckAuth -->|Yes| ProcessData[Process Request]
    ProcessData --> CheckDB{Data Exists?}
    CheckDB -->|No| CreateNew[Create New Record]
    CheckDB -->|Yes| UpdateExisting[Update Existing]
    CreateNew --> SaveDB[(Save to Database)]
    UpdateExisting --> SaveDB
    SaveDB --> ReturnSuccess[Return 200 Success]
    ReturnError --> End([End])
    ReturnUnauth --> End
    ReturnSuccess --> End
```

**3. Class Diagram** - For object-oriented architecture, inheritance, relationships:
```mermaid
classDiagram
    class BaseModel {
        +UUID id
        +DateTime created_at
        +DateTime updated_at
        +save() void
        +delete() void
        +to_dict() dict
    }
    
    class User {
        +String email
        +String password_hash
        +String role
        +authenticate(password) bool
        +generate_token() string
    }
    
    class Order {
        +UUID user_id
        +Decimal total_amount
        +String status
        +List~OrderItem~ items
        +calculate_total() Decimal
        +process_payment() bool
    }
    
    class OrderItem {
        +UUID product_id
        +Integer quantity
        +Decimal price
        +get_subtotal() Decimal
    }
    
    BaseModel <|-- User
    BaseModel <|-- Order
    BaseModel <|-- OrderItem
    User "1" --> "0..*" Order : places
    Order "1" --> "1..*" OrderItem : contains
```

**4. State Diagram** - For state machines, workflow states, lifecycle management:
```mermaid
stateDiagram-v2
    [*] --> Draft
    Draft --> PendingReview : submit()
    PendingReview --> Approved : approve()
    PendingReview --> Rejected : reject()
    PendingReview --> Draft : request_changes()
    Rejected --> Draft : revise()
    Approved --> Published : publish()
    Published --> Archived : archive()
    Archived --> [*]
    
    Draft : Entry: initialize_data()
    Draft : Do: allow_edits()
    PendingReview : Entry: notify_reviewers()
    Approved : Entry: log_approval()
    Published : Entry: make_public()
```

**5. Entity Relationship Diagram** - For database schema, data models:
```mermaid
erDiagram
    USER ||--o{ ORDER : places
    USER {
        uuid id PK
        string email UK
        string password_hash
        string role
        datetime created_at
    }
    
    ORDER ||--|{ ORDER_ITEM : contains
    ORDER {
        uuid id PK
        uuid user_id FK
        decimal total_amount
        string status
        datetime created_at
    }
    
    ORDER_ITEM }o--|| PRODUCT : references
    ORDER_ITEM {
        uuid id PK
        uuid order_id FK
        uuid product_id FK
        integer quantity
        decimal price
    }
    
    PRODUCT {
        uuid id PK
        string name
        string description
        decimal price
        integer stock
    }
```

**6. Architecture Diagram (C4 Style)** - For system architecture, component relationships:
```mermaid
graph TB
    subgraph "Client Layer"
        WebApp[Web Application]
        MobileApp[Mobile App]
    end
    
    subgraph "API Gateway"
        Gateway[API Gateway<br/>Rate Limiting, Auth]
    end
    
    subgraph "Service Layer"
        AuthSvc[Auth Service<br/>JWT, OAuth]
        OrderSvc[Order Service<br/>Business Logic]
        PaymentSvc[Payment Service<br/>Stripe Integration]
    end
    
    subgraph "Data Layer"
        PostgreSQL[(PostgreSQL<br/>User & Order Data)]
        Redis[(Redis<br/>Cache & Sessions)]
        S3[(S3<br/>File Storage)]
    end
    
    WebApp --> Gateway
    MobileApp --> Gateway
    Gateway --> AuthSvc
    Gateway --> OrderSvc
    OrderSvc --> PaymentSvc
    AuthSvc --> PostgreSQL
    AuthSvc --> Redis
    OrderSvc --> PostgreSQL
    OrderSvc --> Redis
    PaymentSvc --> S3
    
    style AuthSvc fill:#e1f5ff
    style OrderSvc fill:#e1f5ff
    style PaymentSvc fill:#e1f5ff
    style PostgreSQL fill:#ffe1e1
    style Redis fill:#ffe1e1
```

**7. Timeline Diagram** - For deployment pipelines, version history, event sequences:
```mermaid
timeline
    title Development Pipeline
    section Development
        Feature Branch : Code Changes
                      : Unit Tests
                      : Code Review
    section Testing
        Merge to Main : Integration Tests
                     : Security Scan
                     : Performance Tests
    section Staging
        Deploy Staging : End-to-End Tests
                      : UAT Testing
    section Production
        Deploy Prod : Health Check
                   : Monitor Metrics
                   : Rollback Ready
```

**8. Git Graph** - For branching strategy, release management:
```mermaid
gitgraph
    commit id: "Initial commit"
    branch develop
    checkout develop
    commit id: "Add auth module"
    branch feature/payment
    checkout feature/payment
    commit id: "Implement payment API"
    commit id: "Add payment tests"
    checkout develop
    merge feature/payment
    commit id: "Update docs"
    checkout main
    merge develop tag: "v1.0.0"
    checkout develop
    commit id: "Start v1.1 features"
```

**Diagram Selection Guidelines:**
- **Sequence Diagrams**: API flows, authentication, multi-service interactions
- **Flowcharts**: Algorithms, decision trees, processing logic
- **Class Diagrams**: OOP architecture, design patterns, inheritance
- **State Diagrams**: Lifecycle management, workflow states, state machines
- **ER Diagrams**: Database schema, data relationships
- **Architecture Diagrams**: System overview, component interactions
- **Timeline**: CI/CD pipelines, deployment processes
- **Git Graph**: Release strategy, branching workflows

Choose the diagram type that best illustrates the technical concept being explained.
"""
        else:  # user mode
            mode_specific_instructions = """
USER MODE - Business-Focused Documentation:

Structure your chapter with these sections:
1. **What It Does**: Simple explanation of the feature's purpose
2. **Why It Matters**: Business value and benefits for users
3. **How It Works**: Step-by-step user workflow (NO CODE)
4. **Key Features**: Main capabilities and functionalities
5. **Use Cases**: Real-world scenarios where this feature shines
   - Provide 2-3 concrete examples with user stories
6. **How Features Connect**: Explain relationships to other features naturally
7. **Tips & Best Practices**: Helpful advice for getting the most value
8. **Common Questions**: Address typical user concerns

Style Guidelines:
- Use simple, non-technical language
- Focus on USER ACTIONS and OUTCOMES, not code
- Use analogies and metaphors to explain concepts
- Include user workflow diagrams (not technical architecture)
- Show screenshots or UI mockups if describing interfaces (use placeholders)
- Explain benefits at each step
- Use mermaid diagrams for USER JOURNEYS
- Link to related feature chapters

Example user workflow format:
**Signing Up for an Account:**

1. **Visit the signup page** - Click "Create Account" on the homepage
2. **Enter your information** - Provide your email, name, and create a password
3. **Verify your email** - Check your inbox for a confirmation link
4. **Complete your profile** - Add additional details about your business
5. **Start using the platform** - You're ready to go!

Mermaid diagram example for user journey:
```mermaid
flowchart LR
    A[New User Visits] --> B[Creates Account]
    B --> C[Verifies Email]
    C --> D[Completes Profile]
    D --> E[Accesses Dashboard]
    E --> F[Starts Using Features]
    
    style A fill:#e1f5ff
    style F fill:#c8e6c9
```

Use case example:
**Use Case: Small Retail Store Owner**

Sarah owns a boutique clothing store and needs to track her inventory. Using the Inventory Management feature:
- She quickly adds new products when shipments arrive
- The system automatically updates stock levels as items sell
- She receives alerts when items are running low
- She can view sales trends to make better purchasing decisions

This saves Sarah 5+ hours per week and helps prevent stockouts of popular items.
"""

        prompt = f"""
{language_instruction}DOCUMENTATION MODE: {doc_mode.upper()}

Write a {"beginner-friendly" if doc_mode == "user" else "comprehensive technical"} tutorial chapter (in Markdown format) for the project `{project_name}` about: "{abstraction_name}". This is Chapter {chapter_num}.

Concept Details:
- Name: {abstraction_name}
- Description:
{abstraction_description}

Complete Tutorial Structure:
{item["full_chapter_listing"]}

Context from previous chapters:
{previous_chapters_summary if previous_chapters_summary else "This is the first chapter."}

{"Relevant Code Snippets:" if doc_mode == "developer" else "Relevant Technical Context (translate to user perspective):"}
{file_context_str if file_context_str else f"No specific {'code' if doc_mode == 'developer' else 'technical'} snippets provided for this abstraction."}

{mode_specific_instructions}

General Instructions:
- Start with `# Chapter {chapter_num}: {abstraction_name}`
- Begin with {" high-level technical motivation" if doc_mode == "developer" else "why users care about this feature"}
- {"Keep code blocks under 15 lines - break longer ones into focused pieces" if doc_mode == "developer" else "NO CODE BLOCKS - focus on user actions and outcomes"}
- Use mermaid diagrams {"for technical flows (max 5-7 participants)" if doc_mode == "developer" else "for user journeys (keep simple and clear)"}
- Link to other chapters using Markdown links from the structure above
- Use {"technical examples and actual code references" if doc_mode == "developer" else "analogies, user stories, and real-world scenarios"}
- End with a brief {"technical summary" if doc_mode == "developer" else "conclusion about user benefits"} and transition to next chapter

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
        doc_mode = shared.get("doc_mode", "developer")
        print(f"✓ Finished writing {len(exec_res_list)} chapters ({doc_mode} mode)")


class CombineTutorial(Node):
    def prep(self, shared):
        project_name = shared["project_name"]
        output_base_dir = shared.get("output_dir", "output")
        doc_mode = shared.get("doc_mode", "developer")

        # Add doc_mode suffix to output directory
        output_path = os.path.join(output_base_dir, f"{project_name}_{doc_mode}")
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
                title=f"{project_name} ({doc_mode.capitalize()} Guide)",
                nav_order=jekyll_nav_order,
                has_children=True,
            )

        doc_type_label = (
            "Developer Documentation" if doc_mode == "developer" else "User Guide"
        )
        index_content += f"# {doc_type_label}: {project_name}\n\n"

        if doc_mode == "user":
            index_content += "📘 **This guide is designed for end-users and stakeholders.** It focuses on features, workflows, and business value without technical jargon.\n\n"
        else:
            index_content += "🔧 **This guide is designed for developers.** It focuses on technical architecture, code structure, and implementation details.\n\n"

        index_content += f"{relationships_data['summary']}\n\n"

        if repo_url:
            index_content += f"**Source Repository:** [{repo_url}]({repo_url})\n\n"

        index_content += f"## {doc_type_label} Structure\n\n"
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
                        title=abstraction_name,
                        parent=f"{project_name} ({doc_mode.capitalize()} Guide)",
                        nav_order=i + 1,
                    )

                chapter_content += chapters_content[i]

                if not chapter_content.endswith("\n\n"):
                    chapter_content += "\n\n"

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

        return {
            "output_path": output_path,
            "index_content": index_content,
            "chapter_files": chapter_files,
            "enable_jekyll": enable_jekyll,
            "doc_mode": doc_mode,
        }

    def exec(self, prep_res):
        output_path = prep_res["output_path"]
        index_content = prep_res["index_content"]
        chapter_files = prep_res["chapter_files"]
        enable_jekyll = prep_res["enable_jekyll"]
        doc_mode = prep_res["doc_mode"]

        print(f"Combining {doc_mode} documentation into: {output_path}")
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
        doc_mode = shared.get("doc_mode", "developer")
        print(
            f"\n✓ {doc_mode.capitalize()} documentation generation complete! Files in: {exec_res}"
        )
