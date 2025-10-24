"""
Query Planner for the PDF Knowledge Graph System.

Uses a small LLM to decompose a user's question into a multi-step
execution plan that can be executed by the query orchestrator.
"""

import json
import os
from typing import Dict, Any

from pdfkg import llm_stats

# LLM imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def _get_planner_llm_client():
    """Initializes and returns the LLM client for the planner."""
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai is required for the Query Planner.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured.")

    # Always use the small, fast model for planning
    model_name = "gemini-2.5-flash"
    return genai.GenerativeModel(model_name)


def _build_planner_prompt(user_question: str) -> str:
    """Builds the prompt for the query planner LLM."""

    return f"""You are an expert query planner for a hybrid Knowledge Graph system.
Your task is to decompose a user's question into a step-by-step execution plan in JSON format.

The system has two tools to retrieve information:
1. `vector_search(query: str)`: Performs semantic search over text chunks. Use this for "what is" or "how to" questions, or when looking for concepts and definitions.
2. `graph_search(start_nodes: list, traverse_edges: list, end_nodes: list)`: Performs a structural search on the knowledge graph. Use this for questions about relationships, locations, or lists of items (e.g., "what figures are in section X?", "list all components").

- `start_nodes` and `end_nodes` are lists of node definitions, e.g., `[{{"type": "section", "label_contains": "safety"}}]`.
- `traverse_edges` are the relationships to follow, e.g., `[{{"type": "CONTAINS"}}]`.

The plan can have one or more steps. The output of a step can be used as input for a subsequent step using a variable, e.g., `{{"variable": "step1_output"}}`.

Analyze the user's question and generate the most efficient plan to answer it.

**User Question:**
"{user_question}"

**Execution Plan JSON:**
"""


def generate_query_plan(user_question: str) -> Dict[str, Any]:
    """
    Uses a small LLM to generate an execution plan from a user question.

    Args:
        user_question: The question asked by the user.

    Returns:
        A dictionary representing the JSON execution plan.
    """
    llm_client = _get_planner_llm_client()
    prompt = _build_planner_prompt(user_question)

    try:
        response = llm_client.generate_content(prompt)
        llm_stats.record_call("gemini-flash", "planning", "query_planner")
        response_text = response.text

        # Clean and parse the JSON response
        if "```json" in response_text:
            response_text = response_text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in response_text:
            response_text = response_text.split("```", 1)[1].split("```", 1)[0]

        plan = json.loads(response_text.strip())
        print(f"🧠 Generated Query Plan:\n{json.dumps(plan, indent=2)}")
        return plan

    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Error parsing query plan from LLM: {e}")
        # Fallback to a simple vector search plan
        return {
            "plan_type": "FALLBACK",
            "steps": [
                {
                    "step": 1,
                    "action": "vector_search",
                    "params": {
                        "query": user_question
                    }
                }
            ]
        }
    except Exception as e:
        print(f"❌ An unexpected error occurred during query planning: {e}")
        # Fallback to a simple vector search plan
        return {
            "plan_type": "FALLBACK",
            "steps": [
                {
                    "step": 1,
                    "action": "vector_search",
                    "params": {
                        "query": user_question
                    }
                }
            ]
        }

