"""
Mistral AI helpers for answer generation.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


def generate_answer_mistral(question: str, context_chunks: list[dict]) -> str:
    """
    Generate answer using Mistral AI.

    Args:
        question: User question.
        context_chunks: Retrieved chunks with metadata.

    Returns:
        Generated answer.
    """
    if not MISTRAL_AVAILABLE:
        raise RuntimeError(
            "mistralai package not installed. Install with: pip install mistralai"
        )

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set in .env")

    # Get model name from environment or use default
    model_name = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

    # Initialize Mistral client
    client = Mistral(api_key=api_key)

    # Build context
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Chunk {i}] (Section: {chunk['section_id']}, Page: {chunk['page']})\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = f"""You are a helpful assistant answering questions about a technical manual.

Question: {question}

Context from the manual:
{context}

Instructions:
- Answer the question based ONLY on the provided context
- Be specific and cite section/page references when possible
- If the context doesn't contain enough information to answer, say so
- Keep your answer concise and technical

Answer:"""

    # Generate response
    response = client.chat.complete(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content
