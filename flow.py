from pocketflow import Flow

# Import optimized node classes
from nodes import (
    FetchRepo,
    IdentifyAbstractions,
    AnalyzeAndOrderChapters,  # OPTIMIZED: Merged node
    WriteChapters,
    CombineTutorial,
)


def create_tutorial_flow():
    """
    Creates and returns the optimized codebase tutorial generation flow.

    OPTIMIZATION: Reduced from 4 sequential LLM calls to 3 by merging
    AnalyzeRelationships + OrderChapters into one call.
    """
    # Instantiate nodes
    fetch_repo = FetchRepo()
    identify_abstractions = IdentifyAbstractions(max_retries=5, wait=20)
    analyze_and_order = AnalyzeAndOrderChapters(
        max_retries=5, wait=20
    )  # OPTIMIZED: Single merged node
    write_chapters = WriteChapters(max_retries=5, wait=20)
    combine_tutorial = CombineTutorial()

    # OPTIMIZED: Shorter pipeline (4 nodes instead of 5)
    fetch_repo >> identify_abstractions
    identify_abstractions >> analyze_and_order  # Single step instead of 2
    analyze_and_order >> write_chapters
    write_chapters >> combine_tutorial

    tutorial_flow = Flow(start=fetch_repo)
    return tutorial_flow
